#pragma once
#include "IPromptEnhancer.hpp"

// Passthrough enhancer used when enhancement is disabled or no model is loaded.
class NullPromptEnhancer final : public IPromptEnhancer {
public:
    EnhancedPrompt enhance(const std::string& positive,
                           const std::string& negative,
                           const std::string&,
                           const std::string&) override {
        return {positive, negative};
    }
    bool isAvailable() const override { return false; }
};
