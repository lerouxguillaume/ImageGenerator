#include "OrtLlmEnhancer.hpp"
#include "../managers/Logger.hpp"
#include <chrono>
#include <cctype>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>

// ── GenAI implementation ──────────────────────────────────────────────────────

#ifdef USE_GENAI
#include "ort_genai.h"
#include <stdexcept>

// Local timing helpers — mirrors sd::fmtMs without pulling in SdUtils.hpp
namespace {
using Clock = std::chrono::steady_clock;
std::string fmtMs(Clock::time_point start) {
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        Clock::now() - start).count();
    return std::to_string(ms) + " ms";
}
} // namespace

struct OrtLlmEnhancer::Impl {
    std::unique_ptr<OgaModel>     model;
    std::unique_ptr<OgaTokenizer> tokenizer;
    bool available = false;
};

// ── Streaming JSON completion tracker ────────────────────────────────────────
//
// Fed one character at a time from the token stream. Signals when the
// outermost JSON object is structurally complete so generation can stop
// immediately — before the model has a chance to drift into trailing text.
//
// State transitions (non-string context):
//   '{' → depth++; started = true
//   '}' → depth--; if depth == 0 → done = true, returns true
//   '"' → enter string context
// State transitions (string context):
//   '\' → arm escape
//   any after escape → disarm escape (character consumed)
//   '"' (unescaped) → leave string context
//
// Everything inside strings is invisible to the depth counter, so
//   "a knight in {fantasy armor}"  does NOT close the object.
struct JsonStreamTracker {
    int  depth    = 0;
    bool inString = false;
    bool escape   = false;
    bool done     = false;

    // Returns true exactly once: when the outermost '}' is written.
    bool feed(char c) {
        if (done) return false;

        if (escape) {
            escape = false;
            return false;
        }
        if (inString) {
            if (c == '\\') escape = true;
            else if (c == '"') inString = false;
            return false;
        }

        // Structural character (outside strings).
        if      (c == '"') { inString = true; }
        else if (c == '{') { ++depth; }
        else if (c == '}') {
            if (--depth == 0) { done = true; return true; }
        }
        return false;
    }

    bool isComplete() const { return done; }
};

// ── Prompt helpers ────────────────────────────────────────────────────────────

static std::string modelTypeGuidance(ModelType model) {
    if (model == ModelType::SDXL)
        return "STYLE: SDXL — write natural-language descriptive sentences forming a coherent scene. "
               "Order: subject → attributes → environment → composition → lighting → style → quality. "
               "Avoid keyword spam; maintain flow and coherence.";
    return "STYLE: SD 1.5 — write concise comma-separated tags. Order: subject → attributes → "
               "environment → composition → lighting → style → quality. "
               "Use strong visual keywords; avoid long sentences.";
}

static std::string strengthGuidance(float strength) {
    if (strength <= 0.3f)
        return "STRENGTH: minimal — fix grammar and add 1-2 quality tags only.";
    if (strength <= 0.7f)
        return "STRENGTH: moderate — improve detail, lighting, and composition "
               "while keeping the subject and intent unchanged.";
    return "STRENGTH: strong — substantially enhance style, atmosphere, and "
           "technical quality; preserve the core subject and intent.";
}

static std::string trimCopy(const std::string& value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

static int countWords(const std::string& text) {
    int  words   = 0;
    bool inWord  = false;
    for (const unsigned char ch : text) {
        if (std::isalnum(ch)) {
            if (!inWord) ++words;
            inWord = true;
        } else {
            inWord = false;
        }
    }
    return words;
}

static int countCommaGroups(const std::string& text) {
    int groups = 0;
    bool hasContent = false;
    for (char ch : text) {
        if (ch == ',') {
            if (hasContent) ++groups;
            hasContent = false;
            continue;
        }
        if (!std::isspace(static_cast<unsigned char>(ch)))
            hasContent = true;
    }
    if (hasContent) ++groups;
    return groups;
}

static bool isMinimalPrompt(const std::string& prompt) {
    const std::string trimmed = trimCopy(prompt);
    if (trimmed.empty()) return false;
    const int words = countWords(trimmed);
    const int groups = countCommaGroups(trimmed);
    return words <= 6 && groups <= 3;
}

// Build a Llama 3 single-turn chat prompt.
//
// The assistant turn is pre-filled with '{' so the model continues directly
// inside a JSON object — it cannot emit a preamble even if it tries.
//
// The prompt also shows one VALID and one INVALID example so the model has
// a concrete negative signal against wrapping the output in prose or markdown.
static std::string buildTransformPrompt(const std::string& prompt,
                                        const std::string& instruction,
                                        ModelType          model,
                                        float              strength) {
    const bool isSDXL = (model == ModelType::SDXL);
    const bool minimalPrompt = isMinimalPrompt(prompt);

    const std::string validExample = isSDXL
        ? "{\"prompt\":\"A detailed portrait of a woman with soft studio lighting, sharp focus on "
          "facial features, a clean neutral background, rendered in photorealistic high quality.\","
          "\"negative_prompt\":\"worst quality, low quality, blurry, bad anatomy, extra limbs, "
          "deformed face, watermark, text\"}"
        : "{\"prompt\":\"portrait of a woman, detailed face, soft studio lighting, clean background, "
          "sharp focus, masterpiece, best quality, highly detailed\","
          "\"negative_prompt\":\"worst quality, low quality, blurry, bad anatomy, extra limbs, "
          "deformed, watermark, text, signature\"}";

    const std::string sys =
        "You are a JSON generator function for Stable Diffusion prompts. "
        "You do NOT chat, explain, or reason. "
        "You output exactly one JSON object and stop immediately.\n\n"
        + modelTypeGuidance(model) + "\n"
        + strengthGuidance(strength) + "\n\n"
        "PROMPT STRUCTURE — cover these components in order when relevant:\n"
        "1. SUBJECT — who or what is depicted\n"
        "2. ATTRIBUTES — appearance, clothing, pose, expression\n"
        "3. ENVIRONMENT — setting, background, time of day\n"
        "4. COMPOSITION — framing, camera angle, shot type (e.g. close-up, wide shot)\n"
        "5. LIGHTING — cinematic, soft, dramatic, golden hour, etc.\n"
        "6. STYLE — realism, anime, oil painting, photographic, etc.\n"
        "7. QUALITY TAGS — masterpiece, best quality, highly detailed, sharp focus\n\n"
        "NEGATIVE PROMPT RULES:\n"
        "- Always include base negatives: worst quality, low quality, blurry, bad anatomy.\n"
        "- Add context-aware negatives (e.g. extra limbs for characters, "
          "blurry background for landscapes, watermark, text).\n\n"
        "RULES:\n"
        "- Preserve the subject and original intent exactly.\n"
        "- Do not add elements unrelated to the subject or instruction.\n"
        "- Do not remove key elements from the original prompt.\n"
        "- Do not repeat the same words.\n"
        "- Do not turn the prompt into an explanation or description of intent.\n\n"
        + (minimalPrompt
            ? std::string(
                "MINIMAL SUBJECT RULES:\n"
                "- The original prompt is intentionally simple.\n"
                "- Keep the result simple and literal.\n"
                "- Do not invent a room, scenery, furniture, people, props, camera drama, or narrative context.\n"
                "- If extra detail is needed, limit it to material, color, surface finish, and lighting consistency.\n"
                "- Prefer a plain background and straightforward framing unless explicitly requested otherwise.\n\n")
            : std::string())
        + "STRICT COMPLETION RULE:\n"
        "- Output exactly one JSON object.\n"
        "- Stop immediately after the closing brace.\n"
        "- No markdown, no code fences, no explanations, no text before or after.\n\n"
        "VALID output (the only acceptable form):\n"
        + validExample + "\n\n"
        "INVALID output (will be rejected — do NOT produce this):\n"
        "Here is the improved prompt: "
        "{\"prompt\":\"...\",\"negative_prompt\":\"...\"}";

    const std::string effectiveInstruction =
        instruction.empty() ? "Improve the prompt quality and detail" : instruction;

    const std::string user =
        "ORIGINAL PROMPT: " + prompt +
        "\nINSTRUCTION: " + effectiveInstruction;

    return "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
           + sys + "<|eot_id|>"
           "<|start_header_id|>user<|end_header_id|>\n"
           + user + "<|eot_id|>"
           "<|start_header_id|>assistant<|end_header_id|>\n"
           "{";   // assistant turn pre-filled: model must continue from inside the JSON object
}

// ── Extraction and validation ─────────────────────────────────────────────────

// Forward balanced-brace extractor.
// Starting at startPos (must be '{'), walks forward counting depth while
// respecting quoted strings and escape sequences.  Returns the complete
// substring [startPos, matching '}'] or nullopt if the object is unclosed.
static std::optional<std::string> extractJsonObject(const std::string& s, size_t startPos) {
    int  depth    = 0;
    bool inString = false;
    bool escape   = false;
    for (size_t i = startPos; i < s.size(); ++i) {
        const char c = s[i];
        if (escape)                { escape = false; continue; }
        if (c == '\\' && inString) { escape = true;  continue; }
        if (c == '"')              { inString = !inString; continue; }
        if (inString)              { continue; }
        if      (c == '{')         { ++depth; }
        else if (c == '}') {
            if (--depth == 0)
                return s.substr(startPos, i - startPos + 1);
        }
    }
    return std::nullopt;
}

// Strict schema validator.
// Requires:
//   • exactly two keys: "prompt" and "negative_prompt"
//   • both must be non-empty strings
//   • no extra keys allowed
static std::optional<LLMResponse> validateSchema(const std::string& candidate) {
    try {
        const auto j = nlohmann::json::parse(candidate);
        if (!j.is_object())                       return std::nullopt;
        if (j.size() != 2)                        return std::nullopt;
        if (!j.contains("prompt"))                return std::nullopt;
        if (!j.contains("negative_prompt"))       return std::nullopt;
        if (!j.at("prompt").is_string())          return std::nullopt;
        if (!j.at("negative_prompt").is_string()) return std::nullopt;
        const std::string p  = j.at("prompt").get<std::string>();
        const std::string np = j.at("negative_prompt").get<std::string>();
        if (p.empty() || np.empty())              return std::nullopt;
        return LLMResponse{p, np};
    } catch (...) {
        return std::nullopt;
    }
}

// Scan raw for JSON objects and return the FIRST one that passes strict schema
// validation.  "First" is correct here: with the '{' prefill the model cannot
// emit preamble before the object, so the first valid object is the intended
// output.  Returning immediately also avoids being fooled by any trailing
// garbage the model may have appended before EOS.
static std::optional<LLMResponse> extractFirstValidResponse(const std::string& raw) {
    for (size_t i = 0; i < raw.size(); ++i) {
        if (raw[i] != '{') continue;
        const auto candidate = extractJsonObject(raw, i);
        if (!candidate) continue;
        if (auto r = validateSchema(*candidate))
            return r;  // first valid object wins — do not scan further
    }
    return std::nullopt;
}

// ── Constructor / destructor ──────────────────────────────────────────────────

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

LLMResponse OrtLlmEnhancer::transform(const LLMRequest& req)
{
    const std::string defaultNeg =
        "worst quality, low quality, blurry, bad anatomy, watermark, text, signature";

    if (!isAvailable()) {
        Logger::info("OrtLlmEnhancer::transform: skipped — model not available");
        return {req.prompt, defaultNeg};
    }

    const std::string modelLabel =
        (req.model == ModelType::SDXL) ? "SDXL" : "SD1.5";

    Logger::info("=== OrtLlmEnhancer::transform ===");
    Logger::info("  model:       " + modelLabel);
    Logger::info("  strength:    " + std::to_string(req.strength));
    Logger::info("  instruction: " + (req.instruction.empty() ? "(none)" : req.instruction));
    Logger::info("  prompt:      " + req.prompt);

    try {
        auto tTotal = Clock::now();

        // ── Build prompt (assistant turn pre-filled with '{') ─────────────────
        const std::string chatPrompt =
            buildTransformPrompt(req.prompt, req.instruction, req.model, req.strength);
        Logger::info("  chat prompt:\n" + chatPrompt);

        // ── Tokenise ──────────────────────────────────────────────────────────
        auto tTok = Clock::now();
        auto sequences = OgaSequences::Create();
        impl_->tokenizer->Encode(chatPrompt.c_str(), *sequences);
        const size_t inputTokens = sequences->SequenceCount(0);
        Logger::info("  tokenised: " + std::to_string(inputTokens) + " input tokens  ("
                     + fmtMs(tTok) + ")");

        // ── Generation params ─────────────────────────────────────────────────
        //
        // temperature=0.15: near-greedy sampling — structural tokens (quotes,
        //   colons, commas, braces) are highly deterministic; content inside
        //   values still varies enough for prompt quality.
        // No min_length: forcing tokens past natural EOS can corrupt a clean
        //   JSON close.
        auto genParams = OgaGeneratorParams::Create(*impl_->model);
        genParams->SetSearchOption("max_length",         800.0);
        genParams->SetSearchOption("temperature",          0.15);
        genParams->SetSearchOption("top_p",                0.9);
        genParams->SetSearchOption("repetition_penalty",   1.05);

        // ── Generate with streaming completion detection ───────────────────────
        //
        // The tracker is fed one character at a time from the decoded token
        // stream.  It signals the instant the outermost JSON object is closed
        // so we break before the model can emit any trailing text.
        //
        // The prefilled '{' that ends the chat prompt counts as the first
        // character of the object — it is fed to the tracker before the loop
        // so the tracker starts at depth=1.
        //
        // Fallback: if no valid completion is detected, the loop runs to natural
        // EOS (IsDone()) or max_length; extractFirstValidResponse() then does a
        // post-hoc scan of whatever was collected.
        auto tGen      = Clock::now();
        auto generator = OgaGenerator::Create(*impl_->model, *genParams);
        generator->AppendTokenSequences(*sequences);
        auto tokStream = OgaTokenizerStream::Create(*impl_->tokenizer);

        JsonStreamTracker tracker;
        tracker.feed('{');  // account for the '{' pre-filled in the prompt

        std::string result;
        int  tokenCount         = 0;
        bool completedInStream  = false;

        while (!generator->IsDone()) {
            generator->GenerateNextToken();
            const auto seqLen   = generator->GetSequenceCount(0);
            const auto newToken = generator->GetSequenceData(0)[seqLen - 1];
            const std::string decoded = tokStream->Decode(newToken);
            result += decoded;
            ++tokenCount;

            // Feed every character of this (possibly multi-char) token.
            // If the tracker fires we break immediately — any characters decoded
            // in the same token beyond the closing '}' are harmless trailing
            // bytes that extractJsonObject() will ignore via depth counting.
            for (const char c : decoded) {
                if (tracker.feed(c)) {
                    completedInStream = true;
                    break;
                }
            }
            if (completedInStream) break;
        }

        const std::string stopReason = completedInStream ? "JSON complete" : "EOS/max_length";
        Logger::info("  generated " + std::to_string(tokenCount) + " tokens  stop=" + stopReason
                     + "  (" + fmtMs(tGen) + ")");

        // Strip model-specific EOS tokens that may appear on EOS/max_length paths.
        for (const char* tok : {"<|eot_id|>", "</s>", "<|end|>", "<eos>"}) {
            const std::string t(tok);
            for (auto p = result.find(t); p != std::string::npos; p = result.find(t))
                result.erase(p, t.size());
        }

        // Re-attach the prefilled '{' — the generation loop only captures tokens
        // produced after it, so the opening brace must be prepended.
        const std::string full = "{" + result;
        Logger::info("  raw output: [" + full + "]");

        // ── Extract and validate ───────────────────────────────────────────────
        if (auto parsed = extractFirstValidResponse(full)) {
            Logger::info("  parsed prompt:    [" + parsed->prompt + "]");
            Logger::info("  parsed negative:  [" + parsed->negative_prompt + "]");
            Logger::info("  total time: " + fmtMs(tTotal));
            return *parsed;
        }

        Logger::info("  JSON extraction failed — falling back to original prompt");
        return {req.prompt, defaultNeg};

    } catch (const std::exception& e) {
        Logger::info(std::string("  EXCEPTION: ") + e.what() + " — falling back to original prompt");
        return {req.prompt, defaultNeg};
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

LLMResponse OrtLlmEnhancer::transform(const LLMRequest& req) {
    return {req.prompt, "worst quality, low quality, blurry, bad anatomy, watermark, text, signature"};
}

#endif
