#pragma once
#include "SdOnnxPatcher.hpp"
#include "../../config/AppConfig.hpp"
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

namespace sd {

// ── LoraOverrides ─────────────────────────────────────────────────────────────
// Owns the merged-weight tensors produced by buildLoraOverrides().
// Must remain alive for the entire duration of Ort::Session construction;
// it can be destroyed immediately afterwards.

struct LoraOverrides {
    std::vector<std::string>           names;     // ONNX initializer names (for AddExternalInitializers)
    std::vector<Ort::Value>            values;    // non-owning views into the backing buffers below

    // Backing storage: one entry per override in names/values order.
    // fp16 models → fp16Bufs; fp32 models → fp32Bufs.
    std::vector<std::vector<uint16_t>> fp16Bufs;
    std::vector<std::vector<float>>    fp32Bufs;

    Ort::MemoryInfo memInfo;

    LoraOverrides();
    bool empty() const noexcept { return names.empty(); }
};

// Reads base weights for all LoRA-matched tensors from bundle.dataPath,
// applies the LoRA delta (W_merged = W_base + scale * alpha/rank * up @ down),
// and returns a LoraOverrides ready to pass to SessionOptions::AddExternalInitializers.
//
// Only tensors that have at least one LoRA match are included in the result.
// Tensors with no LoRA match are left for ORT to load natively from .onnx.data.
LoraOverrides buildLoraOverrides(const OnnxModelBundle&          bundle,
                                 const OnnxExternalIndex&        extIndex,
                                 const OnnxExternalSuffixIndex&  extSuffixIndex,
                                 const std::vector<LoraEntry>&   loras);

} // namespace sd
