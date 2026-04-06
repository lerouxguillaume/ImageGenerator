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

// ── SD 1.5 single-encoder ─────────────────────────────────────────────────────

std::vector<float> encodeText(const std::string& prompt,
                              ClipTokenizer& tokenizer,
                              GenerationContext& ctx,
                              std::vector<int64_t>& out_shape) {
    auto token_ids = tokenizer.encode(prompt);
    Logger::info("encodeText: " + std::to_string(token_ids.size()) + " tokens"
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
    std::string shape_str = "[";
    for (size_t k = 0; k < shape.size(); ++k)
        shape_str += std::to_string(shape[k]) + (k + 1 < shape.size() ? ", " : "]");
    Logger::info("  embedding shape: " + shape_str);

    auto embedding = tensorToFloat(te_out.front());
    {
        float mn = 1e9f, mx = -1e9f, sum = 0.0f;
        for (float v : embedding) { mn = std::min(mn, v); mx = std::max(mx, v); sum += v; }
        Logger::info("  embedding stats: min=" + std::to_string(mn)
                     + "  max=" + std::to_string(mx)
                     + "  mean=" + std::to_string(sum / static_cast<float>(embedding.size())));
    }

    out_shape = std::vector<int64_t>(shape.begin(), shape.end());
    return embedding;
}

// ── SDXL dual-encoder ────────────────────────────────────────────────────────

std::vector<float> encodeTextSDXL(const std::string& prompt,
                                  ClipTokenizer& tokenizer,
                                  GenerationContext& ctx,
                                  std::vector<int64_t>& out_shape,
                                  std::vector<float>& out_pooled) {
    auto token_ids = tokenizer.encode(prompt);
    Logger::info("encodeTextSDXL: " + std::to_string(token_ids.size()) + " tokens"
                 + "  prompt=\"" + prompt.substr(0, 80) + (prompt.size() > 80 ? "..." : "") + "\"");

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