#pragma once
#include "SdSafetensors.hpp"
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace sd {

// Location and metadata of one ONNX initializer that carries inline raw_data.
struct TensorIndex {
    size_t               rawDataOffset = 0;   // byte offset of raw_data payload in the ONNX binary
    size_t               rawDataLength = 0;   // byte count of raw_data payload
    std::vector<int64_t> shape;
    int32_t              dtype = 0;           // 1 = float32, 10 = float16
};

// Key: normalised ONNX initializer name (all '.' replaced with '_').
// Value: location + metadata for in-place patching.
using OnnxTensorIndex = std::map<std::string, TensorIndex>;

// Single-pass parse of the ONNX protobuf binary.
// Returns an index of all initializers with inline raw_data.
// Initializers that reference external data files are skipped and logged.
// Throws std::runtime_error on structural parse errors.
OnnxTensorIndex parseTensorIndex(const std::vector<uint8_t>& onnxBytes);

// Applies LoRA deltas from `lora` to the mutable ONNX bytes using the pre-built index.
//   W_merged = W_base + (userScale * alpha / rank) * (lora_up @ lora_down)
// Supports both fp32 and fp16 base weights; computation is always done in fp32.
// Returns the number of initializers patched (0 is not an error).
int applyLoraToBytes(std::vector<uint8_t>&  onnxBytes,
                     const OnnxTensorIndex& index,
                     const SafetensorsMap&  lora,
                     float                  userScale);

} // namespace sd
