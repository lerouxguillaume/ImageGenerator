#pragma once
#include "SdTypes.hpp"
#include "../../config/AppConfig.hpp"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace sd {

// ── Cache key ─────────────────────────────────────────────────────────────────
// Identifies one fully-loaded model: directory + model schema + LoRA selection.
//
// Construction guarantees:
//   • modelDir  is weakly_canonical (resolves ./ and symlinks, no trailing slash)
//   • loras[]   is sorted by path so {A,B} and {B,A} map to the same key
//   • scales    are stored as int(lround(scale * 1000)) — fixed-point, rounded,
//               so representation noise ≤ 0.0005 never causes a spurious miss
//
// Both operator== and ModelCacheKeyHash operate on these canonical fields.
// They are always consistent because the key is normalised at construction time.

struct ModelCacheKey {
    std::string            modelDir;
    ModelConfig            cfg;    // type + resolution from model.json
    std::vector<LoraEntry> loras;  // sorted by path; scales rounded to 0.001

    // Factory: normalises modelDir and loras before storing them.
    static ModelCacheKey make(const ModelConfig&            cfg,
                              const std::string&            modelDir,
                              const std::vector<LoraEntry>& loras);

    bool operator==(const ModelCacheKey& o) const noexcept;
};

struct ModelCacheKeyHash {
    // Uses XXH64 over a single contiguous string buffer so the hash quality
    // does not depend on the stdlib's std::hash<std::string> implementation.
    size_t operator()(const ModelCacheKey& k) const noexcept;
};

// ── ModelManager ──────────────────────────────────────────────────────────────
// Caches every loaded ModelInstance keyed by ModelCacheKey.
// Multiple distinct configurations can coexist in the cache simultaneously.
//
// On get():
//   cache hit  → reuse the stored ModelInstance; reset run_opts.
//   cache miss → call loadModels(), store the result, reset run_opts.
//
// The returned GenerationContext& is valid until the ModelManager is destroyed.
// It must not be accessed after the ModelManager goes out of scope.

class ModelManager {
public:
    // Returns a reference to the context for the requested configuration.
    // Resets run_opts on every call so a previous SetTerminate() never carries over.
    GenerationContext& get(const ModelConfig&            cfg,
                           const std::string&            modelDir,
                           const std::vector<LoraEntry>& loras);

private:
    std::unordered_map<ModelCacheKey, ModelInstance, ModelCacheKeyHash> cache_;
};

} // namespace sd
