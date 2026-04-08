#pragma once
#include "SdTypes.hpp"
#include "../../config/AppConfig.hpp"
#include <string>
#include <vector>

namespace sd {

// Reads <modelDir>/model.json; falls back to SD 1.5 defaults if absent.
ModelConfig loadModelConfig(const std::string& modelDir);

// Creates and returns a fully initialised GenerationContext for cfg.
// Loads text encoder(s), UNet (GPU + CPU fallback), and VAE decoder.
// If loras is non-empty, each ONNX model is read into memory, LoRA weights
// are applied in-place, and the session is created from the patched bytes.
GenerationContext loadModels(const ModelConfig&           cfg,
                             const std::string&           modelDir,
                             const std::vector<LoraEntry>& loras = {});

std::string mapLoraKeyToOnnx(const std::string& key);
} // namespace sd