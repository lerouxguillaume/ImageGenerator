#pragma once
#include "../enum/enums.hpp"
#include <string>

// Stateless prompt transformation request / response.
// Used by transform() to apply an instruction to a prompt without conversation history.
struct LLMRequest {
    std::string prompt;
    std::string instruction; // e.g. "make it cinematic"; empty → generic quality improvement
    ModelType   model;       // SD15 or SDXL — drives output style guidance
    float       strength;    // 0.0–1.0: how strongly to apply the transformation
};

struct LLMResponse {
    std::string prompt;
    std::string negative_prompt;
};

// Pure interface for prompt enhancement. Implementations may use an LLM,
// a rules-based expander, or be a no-op. The controller holds one instance
// and calls enhance() on a background thread; swap implementations via config.
class IPromptEnhancer {
public:
    virtual ~IPromptEnhancer() = default;

    // Stateless prompt transformation: applies an instruction (e.g. "make it cinematic")
    // to a prompt and returns a transformed prompt + negative prompt as JSON.
    // No conversation history is maintained; each call is independent.
    // Called from a background thread — must be thread-safe.
    virtual LLMResponse transform(const LLMRequest& req) = 0;

    // Returns true if this enhancer has a model loaded and can do real work.
    virtual bool isAvailable() const = 0;
};
