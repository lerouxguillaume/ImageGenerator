#pragma once
#include <string>

struct EnhancedPrompt {
    std::string positive;
    std::string negative;
};

// Pure interface for prompt enhancement. Implementations may use an LLM,
// a rules-based expander, or be a no-op. The controller holds one instance
// and calls enhance() on a background thread; swap implementations via config.
class IPromptEnhancer {
public:
    virtual ~IPromptEnhancer() = default;

    // Enhance positive and negative prompts.
    // styleContext: plain-English style description or example prompt for this model;
    //   empty string → generic enhancement with no model-specific guidance.
    // Called from a background thread — must be thread-safe.
    virtual EnhancedPrompt enhance(const std::string& positive,
                                   const std::string& negative,
                                   const std::string& modelName,
                                   const std::string& styleContext) = 0;

    // Returns true if this enhancer has a model loaded and can do real work.
    virtual bool isAvailable() const = 0;
};
