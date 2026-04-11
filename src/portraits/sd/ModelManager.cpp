#include "ModelManager.hpp"
#include "SdLoader.hpp"
#include "../../managers/Logger.hpp"
#include <functional>

namespace sd {

// ── ModelCacheKey equality ────────────────────────────────────────────────────
// LoRA scales are quantized to int(scale * 1000) before comparison so that
// floating-point representation noise cannot produce spurious cache misses.

bool ModelCacheKey::operator==(const ModelCacheKey& o) const noexcept {
    if (modelDir  != o.modelDir)  return false;
    if (cfg.type  != o.cfg.type)  return false;
    if (loras.size() != o.loras.size()) return false;
    for (size_t i = 0; i < loras.size(); ++i) {
        if (loras[i].path != o.loras[i].path) return false;
        if (int(loras[i].scale * 1000.0f) != int(o.loras[i].scale * 1000.0f)) return false;
    }
    return true;
}

// ── ModelCacheKey hash ────────────────────────────────────────────────────────
// Combines modelDir, cfg.type, and per-LoRA (path, quantized scale) using
// the standard boost-style hash_combine pattern.

size_t ModelCacheKeyHash::operator()(const ModelCacheKey& k) const noexcept {
    // hash_combine: h ^= hash(v) + 0x9e3779b9 + (h << 6) + (h >> 2)
    auto combine = [](size_t h, size_t v) noexcept -> size_t {
        return h ^ (v + 0x9e3779b9u + (h << 6) + (h >> 2));
    };

    size_t h = std::hash<std::string>{}(k.modelDir);
    h = combine(h, std::hash<int>{}(static_cast<int>(k.cfg.type)));
    for (const auto& lo : k.loras) {
        h = combine(h, std::hash<std::string>{}(lo.path));
        h = combine(h, std::hash<int>{}(int(lo.scale * 1000.0f)));
    }
    return h;
}

// ── ModelManager::get ─────────────────────────────────────────────────────────

GenerationContext& ModelManager::get(const ModelConfig&            cfg,
                                      const std::string&            modelDir,
                                      const std::vector<LoraEntry>& loras) {
    const ModelCacheKey key{modelDir, cfg, loras};

    auto it = cache_.find(key);
    if (it == cache_.end()) {
        Logger::info("ModelManager: cache miss — loading models.");
        ModelInstance inst = loadModels(cfg, modelDir, loras);
        inst.config = cfg;
        auto [inserted, ok] = cache_.emplace(key, std::move(inst));
        (void)ok;
        it = inserted;
    } else {
        Logger::info("ModelManager: cache hit — reusing loaded sessions.");
    }

    // Always reset run_opts so a SetTerminate() from a previous cancelled run
    // does not affect the new run.
    it->second.ctx.run_opts = Ort::RunOptions{};
    return it->second.ctx;
}

} // namespace sd
