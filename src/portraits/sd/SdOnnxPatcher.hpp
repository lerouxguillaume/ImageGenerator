#pragma once
#include "SdSafetensors.hpp"
#include <cstdint>
#include <map>
#include <memory>
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

using OnnxTensorIndexIt = OnnxTensorIndex::const_iterator;

struct SuffixEntry {
    OnnxTensorIndexIt it;
    size_t            suffixLen;
};

using OnnxSuffixIndex = std::unordered_map<std::string, std::vector<SuffixEntry>>;

// Single-pass parse of the ONNX protobuf binary.
// Returns an index of all initializers with inline raw_data.
// Initializers that reference external data files are skipped and logged.
// Throws std::runtime_error on structural parse errors.
OnnxTensorIndex parseTensorIndex(const std::vector<uint8_t>& onnxBytes);

// Builds a secondary lookup table keyed by every '_'-boundary suffix of each
// normalised ONNX name.  Enables O(1) matching by suffix instead of a linear scan.
// The returned map holds raw pointers into `index`; `index` must outlive it.
OnnxSuffixIndex buildSuffixIndex(const OnnxTensorIndex& index);

// One LoRA layer entry: down, up, and optional alpha scalar.
struct LoraLayer {
    const SafeTensor* down  = nullptr;
    const SafeTensor* up    = nullptr;
    float             alpha = 0.0f;  // 0 → default to rank
};

// Result of grouping a SafetensorsMap into (down, up, alpha) triplets.
struct ParsedLora {
    std::map<std::string, LoraLayer> layers;  // key = base layer name (prefix stripped)
};

// Groups safetensors keys from `lora` into ParsedLora by stripping the kohya
// prefix (lora_unet_, lora_te_, lora_te2_) and suffixes (.lora_down.weight, etc.).
ParsedLora parseLoraLayers(const SafetensorsMap& lora);

// Computes delta = effectiveScale * (lora_up @ lora_down), entirely in fp32.
// lora_up:   [out_feat, rank]
// lora_down: [rank, in_feat]
// Returns a flat [out_feat * in_feat] vector.
std::vector<float> computeLoraDelta(const SafeTensor& up,
                                    const SafeTensor& down,
                                    float             effectiveScale);

// Result of applyLoraToBytes: a newly allocated patched buffer and the patch count.
struct PatchResult {
    std::shared_ptr<std::vector<uint8_t>> bytes;
    int                                   patchCount = 0;
};

// Applies LoRA deltas to a copy of `onnxBytes`; the input buffer is never mutated.
//   W_merged = W_base + (userScale * alpha / rank) * (lora_up @ lora_down)
// Supports both fp32 and fp16 base weights; computation is always done in fp32.
// Returns a new buffer with the patches applied, plus the number of initializers patched.
PatchResult applyLoraToBytes(std::shared_ptr<const std::vector<uint8_t>> onnxBytes,
                              const OnnxSuffixIndex&                      suffixIndex,
                              const SafetensorsMap&                       lora,
                              float                                       userScale);

} // namespace sd
