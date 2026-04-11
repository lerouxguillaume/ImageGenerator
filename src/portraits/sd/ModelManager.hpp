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
// Float stability (Task 4): LoRA scales are compared and hashed as
//   int(scale * 1000)
// This avoids false cache misses from floating-point representation noise
// while still distinguishing values that differ by ≥ 0.001.

struct ModelCacheKey {
    std::string            modelDir;
    ModelConfig            cfg;      // type + resolution from model.json
    std::vector<LoraEntry> loras;    // path + scale for each active adapter

    bool operator==(const ModelCacheKey& o) const noexcept;
};

struct ModelCacheKeyHash {
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
