#pragma once
#include "SdTypes.hpp"
#include "../../config/AppConfig.hpp"
#include <string>
#include <vector>

namespace sd {

// Reads <modelDir>/model.json; falls back to SD 1.5 defaults if absent.
ModelConfig loadModelConfig(const std::string& modelDir);

// Creates and returns a fully initialised ModelInstance for cfg.
// Loads text encoder(s), UNet (GPU + CPU fallback), and VAE decoder.
// No-LoRA path: sessions are created directly from the .onnx file path so
// ORT can memory-map the file and resolve .onnx.data natively.
// LoRA path: external-data tensor metadata is parsed once per .onnx file,
// LoRA deltas are applied to matched base weights, and the merged tensors are
// injected via SessionOptions::AddExternalInitializers before session creation.
// ORT still loads all non-patched weights natively from .onnx.data.
ModelInstance loadModels(const ModelConfig&            cfg,
                         const std::string&            modelDir,
                         const std::vector<LoraEntry>& loras = {});

} // namespace sd