#pragma once
#include "SdSafetensors.hpp"
#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace sd {

// Location and metadata of one ONNX initializer that carries inline raw_data.
struct TensorIndex {
    size_t               rawDataOffset = 0;   // byte offset of raw_data payload in the ONNX binary
    size_t               rawDataLength = 0;   // byte count of raw_data payload
    std::vector<int64_t> shape;
    int32_t              dtype = 0;           // 1 = float32, 10 = float16
};

// Key: normalised ONNX initializer name (all '.' and '/' replaced with '_').
// Value: location + metadata for in-place patching.
using OnnxTensorIndex = std::map<std::string, TensorIndex>;

// Key: every '_'-boundary suffix of a normalised ONNX name.
// Value: pointer into the OnnxTensorIndex that was passed to buildSuffixIndex().
// The OnnxTensorIndex must outlive this map.
using OnnxSuffixIndex = std::unordered_map<std::string, const TensorIndex*>;

// Single-pass parse of the ONNX protobuf binary.
// Returns an index of all initializers with inline raw_data.
// Initializers that reference external data files are skipped and logged.
// Throws std::runtime_error on structural parse errors.
OnnxTensorIndex parseTensorIndex(const std::vector<uint8_t>& onnxBytes);

// Builds a secondary lookup table keyed by every '_'-boundary suffix of each
// normalised ONNX name.  Enables O(1) matching by suffix instead of a linear scan.
// The returned map holds raw pointers into `index`; `index` must outlive it.
OnnxSuffixIndex buildSuffixIndex(const OnnxTensorIndex& index);

// Returns a pointer to the ONNX TensorIndex that best matches the given LoRA
// base name (kohya prefix already stripped, e.g. "down_blocks_0_attn1_to_q").
// Tries "<loraBase>_weight" first, then "<loraBase>_bias".
// Returns nullptr if no match is found.
const TensorIndex* matchLoraKey(const OnnxSuffixIndex& suffixIndex,
                                const std::string&     loraBase);

// Applies LoRA deltas from `lora` to the mutable ONNX bytes using the pre-built suffix index.
//   W_merged = W_base + (userScale * alpha / rank) * (lora_up @ lora_down)
// Supports both fp32 and fp16 base weights; computation is always done in fp32.
// Returns the number of initializers patched (0 is not an error).
int applyLoraToBytes(std::vector<uint8_t>&   onnxBytes,
                     const OnnxSuffixIndex&  suffixIndex,
                     const SafetensorsMap&   lora,
                     float                   userScale);

} // namespace sd
