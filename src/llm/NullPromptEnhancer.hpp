#pragma once
#include "IPromptEnhancer.hpp"

// Passthrough enhancer used when enhancement is disabled or no model is loaded.
class NullPromptEnhancer final : public IPromptEnhancer {
public:
    LLMResponse transform(const LLMRequest& req) override {
        return {req.prompt, "worst quality, low quality, blurry, bad anatomy, watermark"};
    }

    bool isAvailable() const override { return false; }
};
