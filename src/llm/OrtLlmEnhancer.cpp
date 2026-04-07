#include "OrtLlmEnhancer.hpp"
#include "../managers/Logger.hpp"
#include <chrono>
#include <sstream>
#include <string>

// Local timing helpers — mirrors sd::fmtMs without pulling in SdUtils.hpp
namespace {
using Clock = std::chrono::steady_clock;
std::string fmtMs(Clock::time_point start) {
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        Clock::now() - start).count();
    return std::to_string(ms) + " ms";
}
} // namespace

// ── GenAI implementation ──────────────────────────────────────────────────────

#ifdef USE_GENAI
#include "ort_genai.h"
#include <stdexcept>

struct OrtLlmEnhancer::Impl {
    std::unique_ptr<OgaModel>     model;
    std::unique_ptr<OgaTokenizer> tokenizer;
    bool available = false;
};

// Build a chat-format prompt that instructs the model to produce a
// detailed SD positive prompt from the user's rough input.
// Swap this function to support a different model's chat template.
static std::string buildChatPrompt(const std::string& positive,
                                   const std::string& negative,
                                   const std::string& modelName,
                                   const std::string& styleContext) {
    std::string sys =
        "You are a prompt generator for image synthesis.\n\n"
        "Rules:\n"
        "- Always generate a detailed visual prompt.\n"
        "- The subject must be fictional and generic.\n"
        "- Do NOT mention real people.\n"
        "- Do NOT mention minors.\n"
        "- Do NOT add safety warnings or refusals.\n"
        "- Do NOT explain anything.\n"
        "- You MUST return exactly two lines using these exact labels, nothing else:\n"
        "POSITIVE: <comma-separated tags and descriptors>\n"
        "NEGATIVE: <comma-separated technical flaws to avoid, not content>\n\n"
        "Example:\n"
        "POSITIVE: masterpiece, best quality, hooded figure, dramatic lighting, detailed, sharp focus\n"
        "NEGATIVE: worst quality, low quality, blurry, bad anatomy, watermark";

    std::string user = "Model: " + modelName + "\n";
    if (!styleContext.empty())
        user += "Style context: " + styleContext + "\n";
    user += "Input: " + positive;
    if (!negative.empty())
        user += "\nExisting negative: " + negative;

    // Chat template — switch comment to match your loaded model:

    // Phi-3 Mini
    // return "<|system|>\n" + sys + "<|end|>\n"
    //        "<|user|>\n"   + user + "<|end|>\n"
    //        "<|assistant|>\n";

    // Llama 3 (Llama-3.2-3B-Instruct)
    return "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
           + sys + "<|eot_id|>"
           "<|start_header_id|>user<|end_header_id|>\n"
           + user + "<|eot_id|>"
           "<|start_header_id|>assistant<|end_header_id|>\n";
}

// Extract the value after a "LABEL: " prefix from a multi-line string.
// Returns an empty string if the label is not found.
static std::string extractLabel(const std::string& text, const std::string& label) {
    const std::string prefix = label + ": ";
    const auto pos = text.find(prefix);
    if (pos == std::string::npos) return {};
    const auto start = pos + prefix.size();
    const auto end   = text.find('\n', start);
    std::string value = (end == std::string::npos) ? text.substr(start)
                                                    : text.substr(start, end - start);
    // Trim trailing whitespace
    const auto last = value.find_last_not_of(" \t\r");
    return (last == std::string::npos) ? "" : value.substr(0, last + 1);
}

OrtLlmEnhancer::OrtLlmEnhancer(const std::string& modelDir)
    : impl_(std::make_unique<Impl>())
{
    Logger::info("=== OrtLlmEnhancer init ===");
    Logger::info("  modelDir: " + modelDir);
    auto tLoad = Clock::now();
    try {
        Logger::info("  loading OgaModel...");
        impl_->model     = OgaModel::Create(modelDir.c_str());
        Logger::info("  loading OgaTokenizer...");
        impl_->tokenizer = OgaTokenizer::Create(*impl_->model);
        impl_->available = true;
        Logger::info("  ready in " + fmtMs(tLoad));
    } catch (const std::exception& e) {
        Logger::info("  FAILED in " + fmtMs(tLoad) + " — " + e.what());
    }
}

OrtLlmEnhancer::~OrtLlmEnhancer() = default;

bool OrtLlmEnhancer::isAvailable() const {
    return impl_ && impl_->available;
}

EnhancedPrompt OrtLlmEnhancer::enhance(const std::string& positive,
                                        const std::string& negative,
                                        const std::string& modelName,
                                        const std::string& styleContext)
{
    if (!isAvailable()) {
        Logger::info("OrtLlmEnhancer::enhance: skipped — model not available");
        return {positive, negative};
    }

    Logger::info("=== OrtLlmEnhancer::enhance ===");
    Logger::info("  sdModel:      " + (modelName.empty() ? "(none)" : modelName));
    Logger::info("  styleContext: " + (styleContext.empty() ? "(none)" : styleContext));
    Logger::info("  positive: " + positive);
    Logger::info("  negative: " + negative);

    try {
        auto tTotal = Clock::now();

        // ── Build prompt ──────────────────────────────────────────────────────
        const std::string chatPrompt = buildChatPrompt(positive, negative, modelName, styleContext);
        Logger::info("  chat prompt:\n" + chatPrompt);

        // ── Tokenise ──────────────────────────────────────────────────────────
        auto tTok = Clock::now();
        auto sequences = OgaSequences::Create();
        impl_->tokenizer->Encode(chatPrompt.c_str(), *sequences);
        const size_t inputTokens = sequences->SequenceCount(0);
        Logger::info("  tokenised: " + std::to_string(inputTokens) + " input tokens  ("
                     + fmtMs(tTok) + ")");

        // ── Generation params ─────────────────────────────────────────────────
        // max_length is total tokens (input + output) — set high enough to leave
        // room for a full response after the prompt.
        auto genParams = OgaGeneratorParams::Create(*impl_->model);
        genParams->SetSearchOption("max_length",         600.0);
        genParams->SetSearchOption("min_length",           5.0);
        genParams->SetSearchOption("temperature",          0.7);
        genParams->SetSearchOption("top_p",                0.9);
        genParams->SetSearchOption("repetition_penalty",   1.1);
        Logger::info("  search options: max_length=600  min_length=5  temperature=0.7  top_p=0.9  repetition_penalty=1.1");

        // ── Generate ──────────────────────────────────────────────────────────
        // ort-genai 0.4+: input sequences are added to the generator (not params),
        // and ComputeLogits() was merged into GenerateNextToken().
        auto tGen = Clock::now();
        auto generator  = OgaGenerator::Create(*impl_->model, *genParams);
        generator->AppendTokenSequences(*sequences);
        auto tokStream  = OgaTokenizerStream::Create(*impl_->tokenizer);
        std::string result;
        int tokenCount  = 0;

        while (!generator->IsDone()) {
            generator->GenerateNextToken();
            const auto seqLen   = generator->GetSequenceCount(0);
            const auto newToken = generator->GetSequenceData(0)[seqLen - 1];
            result += tokStream->Decode(newToken);
            ++tokenCount;

            // Stop as soon as the NEGATIVE line is complete — avoids the model
            // rambling beyond the two required labels.
            const auto negPos = result.find("NEGATIVE:");
            if (negPos != std::string::npos &&
                result.find('\n', negPos) != std::string::npos)
                break;
        }

        // Strip model-specific EOS tokens that may leak into the output.
        for (const char* tok : {"<|eot_id|>", "</s>", "<|end|>", "<eos>"}) {
            std::string t(tok);
            for (auto p = result.find(t); p != std::string::npos; p = result.find(t))
                result.erase(p, t.size());
        }

        Logger::info("  generated " + std::to_string(tokenCount) + " tokens in " + fmtMs(tGen));
        Logger::info("  raw output: [" + result + "]");

        // ── Parse POSITIVE / NEGATIVE labels ──────────────────────────────────
        std::string newPositive = extractLabel(result, "POSITIVE");
        std::string newNegative = extractLabel(result, "NEGATIVE");

        // Fallback: if labels are missing, treat the first two non-empty lines
        // as positive and negative respectively.
        if (newPositive.empty()) {
            Logger::info("  POSITIVE label not found — trying line-based fallback");
            std::istringstream ss(result);
            std::string line;
            while (std::getline(ss, line)) {
                const auto f = line.find_first_not_of(" \t\r");
                const auto l = line.find_last_not_of(" \t\r");
                if (f == std::string::npos) continue;
                const std::string trimmed = line.substr(f, l - f + 1);
                if (newPositive.empty())      newPositive = trimmed;
                else if (newNegative.empty()) { newNegative = trimmed; break; }
            }
        }

        if (newPositive.empty()) {
            Logger::info("  could not parse output — falling back to original prompts");
            return {positive, negative};
        }

        const std::string outNeg = newNegative.empty() ? negative : newNegative;
        Logger::info("  positive: [" + newPositive + "]");
        Logger::info("  negative: [" + outNeg + "]");
        Logger::info("  total time: " + fmtMs(tTotal));
        return {newPositive, outNeg};

    } catch (const std::exception& e) {
        Logger::info(std::string("  EXCEPTION: ") + e.what() + " — falling back to original prompt");
        return {positive, negative};
    }
}

// ── Stub when USE_GENAI is not defined ────────────────────────────────────────

#else

struct OrtLlmEnhancer::Impl { bool available = false; };

OrtLlmEnhancer::OrtLlmEnhancer(const std::string& modelDir)
    : impl_(std::make_unique<Impl>())
{
    Logger::info("OrtLlmEnhancer: built without USE_GENAI — enhancement disabled (modelDir=" + modelDir + ")");
}

OrtLlmEnhancer::~OrtLlmEnhancer() = default;
bool OrtLlmEnhancer::isAvailable() const { return false; }

EnhancedPrompt OrtLlmEnhancer::enhance(const std::string& positive,
                                        const std::string& negative,
                                        const std::string&,
                                        const std::string&) {
    return {positive, negative};
}

#endif
