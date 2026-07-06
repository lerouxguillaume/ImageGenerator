#pragma once
#include "SdTypes.hpp"
#include "../../config/AppConfig.hpp"
#include <filesystem>
#include <string>
#include <vector>

namespace sd {

// Absolute path to the directory containing the running executable. Bundled
// resources (the tokenizer vocab/merges under models/) are resolved against
// this rather than the process working directory: cwd is not reliably the
// executable dir on all platforms — e.g. when running from a VirtualBox shared
// folder, where chdir into the share fails and cwd stays elsewhere. Falls back
// to the current working directory if the executable path cannot be determined.
std::filesystem::path resourceDir();

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