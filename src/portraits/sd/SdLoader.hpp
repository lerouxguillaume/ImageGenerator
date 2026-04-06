#pragma once
#include "SdTypes.hpp"
#include <string>

namespace sd {

// Reads <modelDir>/model.json; falls back to SD 1.5 defaults if absent.
ModelConfig loadModelConfig(const std::string& modelDir);

// Creates and returns a fully initialised GenerationContext for cfg.
// Loads text encoder(s), UNet (GPU + CPU fallback), and VAE decoder.
GenerationContext loadModels(const ModelConfig& cfg, const std::string& modelDir);

} // namespace sd