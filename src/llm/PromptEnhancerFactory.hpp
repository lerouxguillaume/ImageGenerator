#pragma once
#include "IPromptEnhancer.hpp"
#include "NullPromptEnhancer.hpp"
#include "OrtLlmEnhancer.hpp"
#include <memory>
#include <string>

class PromptEnhancerFactory {
public:
    // Returns an OrtLlmEnhancer when enabled and modelDir is non-empty,
    // NullPromptEnhancer otherwise. Callers should check isAvailable() on
    // the result to know whether a real model was loaded.
    static std::unique_ptr<IPromptEnhancer> create(bool enabled,
                                                    const std::string& modelDir) {
        if (enabled && !modelDir.empty())
            return std::make_unique<OrtLlmEnhancer>(modelDir);
        return std::make_unique<NullPromptEnhancer>();
    }
};
