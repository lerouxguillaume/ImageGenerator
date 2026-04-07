#pragma once
#include "IPromptEnhancer.hpp"
#include <memory>
#include <string>

// Prompt enhancer backed by a local ONNX-Runtime-GenAI language model.
// Compile with -DUSE_GENAI to enable; without it this class is a no-op stub.
// Recommended model: Phi-3 Mini 4K Instruct (ONNX, cpu-int4-rtn-block-32).
class OrtLlmEnhancer final : public IPromptEnhancer {
public:
    explicit OrtLlmEnhancer(const std::string& modelDir);
    ~OrtLlmEnhancer() override;

    // Thread-safe: called from a background thread by the controller.
    EnhancedPrompt enhance(const std::string& positive,
                           const std::string& negative,
                           const std::string& modelName,
                           const std::string& styleContext) override;

    bool isAvailable() const override;

private:
    // PImpl keeps ort_genai.h out of this header so the rest of the project
    // does not need to see it (and compiles cleanly without USE_GENAI).
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
