#pragma once
#include "SdSafetensors.hpp"
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace sd {

// ── Model bundle ──────────────────────────────────────────────────────────────
// Treats an ONNX model as a single atomic unit: the graph file (.onnx) and its
// optional external-data companion (.onnx.data).

struct OnnxModelBundle {
    std::filesystem::path onnxPath;  // always present
    std::filesystem::path dataPath;  // empty iff there is no .onnx.data companion

    bool hasExternalData() const noexcept { return !dataPath.empty(); }
};

// Probes the filesystem to detect the model layout.
// Sets dataPath = onnxPath + ".data" when that companion file exists.
// Throws std::runtime_error if onnxPath itself does not exist.
OnnxModelBundle resolveBundle(const std::filesystem::path& onnxPath);

// ── External-data tensor index ────────────────────────────────────────────────
// Parsed once per .onnx file; used by the AddExternalInitializers LoRA path.

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

// Builds a suffix lookup table over an OnnxExternalIndex.
// Enables O(1) LoRA key matching by suffix. The returned map holds iterators
// into `index`; `index` must outlive it.
OnnxExternalSuffixIndex buildExternalSuffixIndex(const OnnxExternalIndex& index);

// ── LoRA layer types ──────────────────────────────────────────────────────────

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

} // namespace sd
