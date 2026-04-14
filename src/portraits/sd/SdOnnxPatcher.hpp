#pragma once
#include "SdSafetensors.hpp"
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace sd {

// ── Model bundle ──────────────────────────────────────────────────────────────
// Treats an ONNX model as a single atomic unit: the graph file (.onnx) and its
// optional external-data companion (.onnx.data).  All loader code works through
// bundles so the layout is never guessed from heuristics.

struct OnnxModelBundle {
    std::filesystem::path onnxPath;  // always present
    std::filesystem::path dataPath;  // empty iff the model is fully inline

    bool hasExternalData() const noexcept { return !dataPath.empty(); }
};

// Probes the filesystem to detect the model layout.
// Sets dataPath = onnxPath + ".data" when that companion file exists.
// Throws std::runtime_error if onnxPath itself does not exist.
OnnxModelBundle resolveBundle(const std::filesystem::path& onnxPath);

// Returns a fully-inline ONNX byte buffer suitable for Session(bytes, size).
//   • Inline bundle  → reads onnxPath and returns the bytes as-is.
//   • External bundle → reads dataPath, resolves every external TensorProto
//     (offset + length), moves tensor bytes into raw_data, clears external_data
//     references, and returns a self-contained buffer.
// Throws std::runtime_error on any I/O or format error.
// The input bundle is never mutated.
std::vector<uint8_t> materializeBundle(const OnnxModelBundle& bundle);

// ── Tensor index ──────────────────────────────────────────────────────────────

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

// Single-pass parse of a fully-inline ONNX byte buffer.
// Returns an index of all initializers that carry inline raw_data.
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

// ── External-data tensor index ────────────────────────────────────────────────
// Used by the AddExternalInitializers LoRA path.  Parsing is done once per
// model file; matching reuses the same suffix-index machinery as the inline path.

struct ExternalTensorMeta {
    std::string          onnxName;    // exact initializer name as stored in the ONNX graph
    std::vector<int64_t> shape;
    int32_t              dtype       = 0;   // 1=float32, 10=float16
    int64_t              dataOffset  = 0;   // byte offset inside .onnx.data
    int64_t              dataLength  = 0;   // byte count inside .onnx.data
};

// Key: normalised ONNX initializer name (dots/slashes → '_') for LoRA key matching.
using OnnxExternalIndex   = std::map<std::string, ExternalTensorMeta>;
using OnnxExternalIndexIt = OnnxExternalIndex::const_iterator;

struct ExternalSuffixEntry {
    OnnxExternalIndexIt it;
    size_t              suffixLen;
};
using OnnxExternalSuffixIndex = std::unordered_map<std::string, std::vector<ExternalSuffixEntry>>;

// Parses the .onnx file and returns metadata for all external-data initializers.
// Returns an empty map for a fully-inline model.
// Throws std::runtime_error on I/O or format errors.
OnnxExternalIndex parseExternalIndex(const OnnxModelBundle& bundle);

// Builds a suffix lookup table over an OnnxExternalIndex (same semantics as buildSuffixIndex).
OnnxExternalSuffixIndex buildExternalSuffixIndex(const OnnxExternalIndex& index);

} // namespace sd
