#pragma once
#include "SdOnnxPatcher.hpp"
#include "../../config/AppConfig.hpp"
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <onnxruntime_cxx_api.h>

namespace sd {

// ── LoraOverrides ─────────────────────────────────────────────────────────────
// Owns the merged-weight tensors produced by LoraInjector::applyLoras().
// The caller must keep this object alive for the entire duration of
// Ort::Session construction; it can be destroyed immediately afterwards.

struct LoraOverrides {
    std::vector<std::string>           names;     // ONNX initializer names
    std::vector<Ort::Value>            values;    // non-owning views into the buffers below
    std::vector<std::vector<uint16_t>> fp16Bufs;  // backing storage for fp16 tensors
    std::vector<std::vector<float>>    fp32Bufs;  // backing storage for fp32 tensors
    Ort::MemoryInfo                    memInfo;

    LoraOverrides();
    bool empty() const noexcept { return names.empty(); }
};

// ── LoraInjector ─────────────────────────────────────────────────────────────
// Per-component LoRA injector.  Create one instance per ONNX model file.
//
// Typical usage per session load:
//   1.  injector.loadModelMetadata(onnxPath)          — parse ONNX once; idempotent
//   2.  overrides = injector.applyLoras(opts, weightsPath, loras)
//   3.  Ort::Session(env, onnxPath, opts)             — session construction
//   4.  overrides goes out of scope                    — backing memory freed
//
// Base weights are read from a companion safetensors file produced by the
// export script (e.g. unet_weights.safetensors next to unet.onnx).  If the
// file is absent, LoRA injection is silently skipped and opts is not modified.
//
// Thread safety:
//   Concurrent calls to applyLoras() on the same instance are serialised by an
//   internal mutex.  Cache misses block other callers while the matmul is in
//   progress; cache hits are fast.  loadModelMetadata() is NOT thread-safe and
//   must be called before any concurrent use.
//
// Error handling:
//   Per-tensor shape/dtype/name mismatches are logged and skipped.  The method
//   never throws due to LoRA content; it only throws on internal logic errors.

class LoraInjector {
public:
    // Parse ONNX graph for initializer metadata (names, shapes, dtypes).
    // Idempotent for the same onnxPath.  Throws on I/O or parse errors.
    void loadModelMetadata(const std::string& onnxPath);

    // Compute merged weights (W_base + Σ scale_i * delta_i, applied in input
    // order) and inject into opts via AddExternalInitializers.
    //
    // weightsPath : companion <stem>_weights.safetensors file.
    // loras       : applied in order for deterministic multi-LoRA results.
    //
    // Returns LoraOverrides that must outlive Ort::Session construction.
    // Returns an empty LoraOverrides if nothing matched — opts is not modified.
    // Thread-safe (serialised by internal mutex).
    LoraOverrides applyLoras(Ort::SessionOptions&          opts,
                             const std::string&             weightsPath,
                             const std::vector<LoraEntry>& loras);

private:
    // One cached tensor: computed merged weight (fp32 or fp16 raw data).
    struct CachedTensor {
        std::string           onnxName;   // exact ONNX initializer name (with dots)
        std::vector<int64_t>  shape;
        int32_t               dtype;      // 1 = float32, 10 = float16
        std::vector<uint16_t> fp16Data;   // non-empty for dtype == 10
        std::vector<float>    fp32Data;   // non-empty for dtype == 1
    };

    // Set once by loadModelMetadata(); read-only afterwards.
    std::string             onnxPath_;
    OnnxExternalIndex       extIndex_;
    OnnxExternalSuffixIndex extSuffixIndex_;

    // Base weights loaded from the companion safetensors file.
    // Loaded lazily on the first applyLoras() call; reused on subsequent calls
    // as long as weightsPath doesn't change.
    std::string                                          loadedWeightsPath_;
    std::unordered_map<std::string, std::vector<float>> baseWeights_;

    // Merged-tensor cache: avoids recomputing matmuls when the same LoRA
    // configuration is requested again (e.g. after a session eviction).
    std::unordered_map<size_t, std::vector<CachedTensor>> mergeCache_;
    std::mutex cacheMutex_;

    // Load base weights from weightsPath into baseWeights_.
    // Returns false if the file is absent or fails to parse.
    bool ensureBaseWeights(const std::string& weightsPath);

    // Compute merged tensors for the given LoRA set.
    // Reads baseWeights_ and extSuffixIndex_; must be called with cacheMutex_ held.
    std::vector<CachedTensor> computeMerge(const std::vector<LoraEntry>& loras);

    // Build a LoraOverrides (copy from cache into owned buffers + Ort::Value views).
    // Must be called with cacheMutex_ held.
    void buildOverrides(const std::vector<CachedTensor>& tensors,
                        LoraOverrides&                    out) const;

    // Stable cache key: FNV-1a over the canonical (onnxPath, weightsPath, loras) string.
    static size_t makeCacheKey(const std::string&             onnxPath,
                                const std::string&             weightsPath,
                                const std::vector<LoraEntry>& loras);
};

} // namespace sd
