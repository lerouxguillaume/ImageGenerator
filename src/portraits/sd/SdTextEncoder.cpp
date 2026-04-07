#include "SdTextEncoder.hpp"
#include "SdUtils.hpp"
#include "../../managers/Logger.hpp"

namespace sd {

// Helper: copy ORT output tensor (fp16 or fp32) into a float32 vector.
static std::vector<float> tensorToFloat(Ort::Value& tensor) {
    auto info = tensor.GetTensorTypeAndShapeInfo();
    size_t n  = info.GetElementCount();
    std::vector<float> out(n);
    if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        const auto* p = tensor.GetTensorData<Ort::Float16_t>();
        for (size_t k = 0; k < n; ++k) out[k] = static_cast<float>(p[k]);
    } else {
        const auto* p = tensor.GetTensorData<float>();
        std::copy(p, p + n, out.begin());
    }
    return out;
}

// ── Weighted prompt helpers ───────────────────────────────────────────────────

struct WeightedSegment { std::string text; float weight = 1.0f; };

// Parse A1111-style weighted prompt into flat segments.
// "(dark knight:1.4), warrior" → [{"dark knight", 1.4}, {", warrior", 1.0}]
// Bare "(text)" without a colon gets a default 1.1 boost (A1111 convention).
static std::vector<WeightedSegment> parseWeightedPrompt(const std::string& prompt) {
    std::vector<WeightedSegment> segs;
    std::string plain;

    for (size_t i = 0; i < prompt.size(); ) {
        if (prompt[i] != '(') { plain += prompt[i++]; continue; }

        if (!plain.empty()) { segs.push_back({plain, 1.0f}); plain.clear(); }

        // Find matching ')'
        int    depth = 1;
        size_t j     = i + 1;
        while (j < prompt.size() && depth > 0) {
            if      (prompt[j] == '(') ++depth;
            else if (prompt[j] == ')') --depth;
            ++j;
        }

        std::string inner = prompt.substr(i + 1, j - i - 2);
        float weight = 1.1f;

        auto colon = inner.rfind(':');
        if (colon != std::string::npos) {
            try {
                float w = std::stof(inner.substr(colon + 1));
                weight  = w;
                inner   = inner.substr(0, colon);
            } catch (...) {}
        }
        segs.push_back({inner, weight});
        i = j;
    }
    if (!plain.empty()) segs.push_back({plain, 1.0f});
    return segs;
}

static bool anyWeighted(const std::vector<WeightedSegment>& segs) {
    for (const auto& s : segs)
        if (std::abs(s.weight - 1.0f) >= 1e-4f) return true;
    return false;
}

static std::string stripToPlain(const std::vector<WeightedSegment>& segs) {
    std::string result;
    for (const auto& s : segs) result += s.text;
    return result;
}

// Count content tokens (BOS at [0], stop at first EOS=49407).
static int countContentTokens(ClipTokenizer& tokenizer, const std::string& text) {
    constexpr int64_t EOS = 49407;
    auto ids = tokenizer.encode(text);
    int n = 0;
    for (size_t k = 1; k < ids.size(); ++k) {
        if (ids[k] == EOS) break;
        ++n;
    }
    return n;
}

// Build per-token weight vector [seq_len=77].
// Position 0 (BOS) and trailing EOS/pad positions stay at 1.0.
static std::vector<float> buildTokenWeights(
    const std::vector<WeightedSegment>& segs,
    ClipTokenizer& tokenizer,
    int seq_len = 77)
{
    std::vector<float> weights(seq_len, 1.0f);
    int pos = 1;  // skip BOS at position 0
    for (const auto& seg : segs) {
        if (pos >= seq_len - 1) break;
        int n = countContentTokens(tokenizer, seg.text);
        for (int t = 0; t < n && pos < seq_len - 1; ++t, ++pos)
            weights[pos] = seg.weight;
    }
    return weights;
}

// Scale hidden states in-place: embedding[seq * dim + d] *= weight[seq].
static void applyTokenWeights(std::vector<float>& embedding,
                               const std::vector<float>& weights,
                               int seq_len, int dim) {
    int weighted_count = 0;
    for (int s = 0; s < seq_len; ++s) {
        if (std::abs(weights[s] - 1.0f) < 1e-4f) continue;
        ++weighted_count;
        for (int d = 0; d < dim; ++d)
            embedding[s * dim + d] *= weights[s];
    }
    Logger::info("  token weights applied to " + std::to_string(weighted_count) + " position(s)");
}

// ── SD 1.5 single-encoder ─────────────────────────────────────────────────────

std::vector<float> encodeText(const std::string& prompt,
                              ClipTokenizer& tokenizer,
                              GenerationContext& ctx,
                              std::vector<int64_t>& out_shape) {
    auto segs     = parseWeightedPrompt(prompt);
    bool weighted = anyWeighted(segs);
    const std::string plain = weighted ? stripToPlain(segs) : prompt;

    auto token_ids = tokenizer.encode(plain);
    Logger::info("encodeText: " + std::to_string(token_ids.size()) + " tokens"
                 + (weighted ? "  [weighted]" : "")
                 + "  prompt=\"" + prompt.substr(0, 80) + (prompt.size() > 80 ? "..." : "") + "\"");

    auto tEnc = Clock::now();
    std::vector<int64_t> token_shape = {1, 77};
    Ort::Value token_tensor = Ort::Value::CreateTensor<int64_t>(
        ctx.memory_info, token_ids.data(), token_ids.size(),
        token_shape.data(), token_shape.size());

    const char* in_names[]  = {ctx.te_input.c_str()};
    const char* out_names[] = {ctx.te_output.c_str()};
    auto te_out = ctx.text_encoder.Run(Ort::RunOptions{nullptr},
                                       in_names, &token_tensor, 1, out_names, 1);
    Logger::info("  text encoder run: " + fmtMs(tEnc));

    auto shape_info = te_out.front().GetTensorTypeAndShapeInfo();
    auto shape      = shape_info.GetShape();
    out_shape       = std::vector<int64_t>(shape.begin(), shape.end());

    auto embedding = tensorToFloat(te_out.front());
    const int dim  = static_cast<int>(out_shape[2]);

    if (weighted) {
        auto token_weights = buildTokenWeights(segs, tokenizer);
        applyTokenWeights(embedding, token_weights, 77, dim);
    }

    {
        float mn = 1e9f, mx = -1e9f, sum = 0.0f;
        for (float v : embedding) { mn = std::min(mn, v); mx = std::max(mx, v); sum += v; }
        Logger::info("  embedding stats: min=" + std::to_string(mn)
                     + "  max=" + std::to_string(mx)
                     + "  mean=" + std::to_string(sum / static_cast<float>(embedding.size())));
    }

    return embedding;
}

// ── SDXL dual-encoder ────────────────────────────────────────────────────────

std::vector<float> encodeTextSDXL(const std::string& prompt,
                                  ClipTokenizer& tokenizer,
                                  GenerationContext& ctx,
                                  std::vector<int64_t>& out_shape,
                                  std::vector<float>& out_pooled) {
    auto segs     = parseWeightedPrompt(prompt);
    bool weighted = anyWeighted(segs);
    const std::string plain = weighted ? stripToPlain(segs) : prompt;

    auto token_ids = tokenizer.encode(plain);
    Logger::info("encodeTextSDXL: " + std::to_string(token_ids.size()) + " tokens"
                 + (weighted ? "  [weighted]" : "")
                 + "  prompt=\"" + prompt.substr(0, 80) + (prompt.size() > 80 ? "..." : "") + "\"");

    // Precompute per-token weights once; applied to both encoders below.
    std::vector<float> token_weights;
    if (weighted)
        token_weights = buildTokenWeights(segs, tokenizer);

    std::vector<int64_t> token_shape = {1, 77};

    // ── Encoder 1: CLIP-L → (1, 77, 768) ─────────────────────────────────────
    Ort::Value tokens1 = Ort::Value::CreateTensor<int64_t>(
        ctx.memory_info, token_ids.data(), token_ids.size(),
        token_shape.data(), token_shape.size());
    const char* enc1_in[]  = {ctx.te_input.c_str()};
    const char* enc1_out[] = {ctx.te_output.c_str()};
    auto tEnc1 = Clock::now();
    auto enc1_result = ctx.text_encoder.Run(Ort::RunOptions{nullptr},
                                            enc1_in, &tokens1, 1, enc1_out, 1);
    Logger::info("  CLIP-L run: " + fmtMs(tEnc1));
    auto shape1 = enc1_result.front().GetTensorTypeAndShapeInfo().GetShape();
    auto emb1   = tensorToFloat(enc1_result.front());
    const int dim1 = static_cast<int>(shape1[2]); // 768

    // ── Encoder 2: OpenCLIP-G → (1, 77, 1280) + pooled (1, 1280) ─────────────
    Ort::Value tokens2 = Ort::Value::CreateTensor<int64_t>(
        ctx.memory_info, token_ids.data(), token_ids.size(),
        token_shape.data(), token_shape.size());
    const char* enc2_in[]  = {ctx.te2_input.c_str()};
    const char* enc2_out[] = {ctx.te2_output.c_str(), ctx.te2_pooled.c_str()};
    auto tEnc2 = Clock::now();
    auto enc2_result = ctx.text_encoder_2.Run(Ort::RunOptions{nullptr},
                                              enc2_in, &tokens2, 1, enc2_out, 2);
    Logger::info("  OpenCLIP-G run: " + fmtMs(tEnc2));
    auto shape2 = enc2_result[0].GetTensorTypeAndShapeInfo().GetShape();
    auto emb2   = tensorToFloat(enc2_result[0]);
    const int dim2 = static_cast<int>(shape2[2]); // 1280

    out_pooled = tensorToFloat(enc2_result[1]);

    // Apply weights to both encoder outputs before concatenation.
    if (weighted) {
        Logger::info("  applying weights: CLIP-L");
        applyTokenWeights(emb1, token_weights, 77, dim1);
        Logger::info("  applying weights: OpenCLIP-G");
        applyTokenWeights(emb2, token_weights, 77, dim2);
    }

    // ── Concatenate along embedding dim → (1, 77, dim1+dim2) ─────────────────
    const int seq_len   = 77;
    const int dim_total = dim1 + dim2;
    std::vector<float> combined(seq_len * dim_total);
    for (int s = 0; s < seq_len; ++s) {
        std::copy(emb1.begin() + s * dim1, emb1.begin() + (s + 1) * dim1,
                  combined.begin() + s * dim_total);
        std::copy(emb2.begin() + s * dim2, emb2.begin() + (s + 1) * dim2,
                  combined.begin() + s * dim_total + dim1);
    }
    out_shape = {1, seq_len, dim_total};
    {
        float mn = 1e9f, mx = -1e9f, sum = 0.0f;
        for (float v : combined) { mn = std::min(mn, v); mx = std::max(mx, v); sum += v; }
        Logger::info("  SDXL embedding [1," + std::to_string(seq_len) + "," + std::to_string(dim_total) + "]"
                     + "  pooled:[1," + std::to_string(out_pooled.size()) + "]"
                     + "  min=" + std::to_string(mn)
                     + "  max=" + std::to_string(mx)
                     + "  mean=" + std::to_string(sum / static_cast<float>(combined.size())));
    }
    return combined;
}

} // namespace sd
