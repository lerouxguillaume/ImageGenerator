#pragma once
#include "SdOnnxPatcher.hpp"
#include "../../config/AppConfig.hpp"
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

namespace sd {

// ── LegacyLoraOverrides ───────────────────────────────────────────────────────
// Deprecated: superseded by LoraInjector / LoraOverrides in LoraInjector.hpp.
// Kept for reference; buildLoraOverrides() still compiles but is no longer
// called by SdLoader.  Do not use in new code.

struct LegacyLoraOverrides {
    std::vector<std::string>           names;     // ONNX initializer names (for AddExternalInitializers)
    std::vector<Ort::Value>            values;    // non-owning views into the backing buffers below

    // Backing storage: one entry per override in names/values order.
    // fp16 models → fp16Bufs; fp32 models → fp32Bufs.
    std::vector<std::vector<uint16_t>> fp16Bufs;
    std::vector<std::vector<float>>    fp32Bufs;

    Ort::MemoryInfo memInfo;

    LegacyLoraOverrides();
    bool empty() const noexcept { return names.empty(); }
};

// Deprecated — see LoraInjector.hpp.
LegacyLoraOverrides buildLoraOverrides(const OnnxModelBundle&          bundle,
                                 const OnnxExternalIndex&        extIndex,
                                 const OnnxExternalSuffixIndex&  extSuffixIndex,
                                 const std::vector<LoraEntry>&   loras);

} // namespace sd
