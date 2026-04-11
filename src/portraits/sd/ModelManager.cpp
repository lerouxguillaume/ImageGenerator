#include "ModelManager.hpp"
#include "SdLoader.hpp"
#include "../../managers/Logger.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <string>

// Header-only xxHash — vendored so the Windows cross-compile (MinGW) doesn't
// need a system package.  XXH_IMPLEMENTATION compiles the routines inline
// into this single translation unit.
#define XXH_STATIC_LINKING_ONLY
#define XXH_IMPLEMENTATION
#include "../../../third_party/xxhash/xxhash.h"

namespace sd {

// ── Helpers ───────────────────────────────────────────────────────────────────

// Round a LoRA scale to the nearest 0.001 and return it as a fixed-point int.
// Using lround (not a plain cast) ensures 0.9995f → 1000, not 999.
static int scaleKey(float s) noexcept {
    return static_cast<int>(std::lround(s * 1000.0f));
}

// Normalise a filesystem path to a canonical string:
//   • resolves "." and ".." components
//   • removes trailing separators
//   • does NOT require the path to exist (weakly_canonical)
static std::string canonicalPath(const std::string& p) {
    return std::filesystem::weakly_canonical(p).string();
}

// ── ModelCacheKey::make ───────────────────────────────────────────────────────
// Single construction point that enforces all normalisation invariants so that
// operator== and the hash function never need to re-derive them.

ModelCacheKey ModelCacheKey::make(const ModelConfig&            cfg,
                                   const std::string&            modelDir,
                                   const std::vector<LoraEntry>& loras) {
    ModelCacheKey k;
    k.cfg      = cfg;
    k.modelDir = canonicalPath(modelDir);

    k.loras.reserve(loras.size());
    for (const auto& lo : loras)
        k.loras.push_back({ canonicalPath(lo.path), lo.scale });

    // Sort by canonical path so {A,B} and {B,A} produce the same key.
    // LoRA application is commutative in terms of which weights end up merged.
    std::sort(k.loras.begin(), k.loras.end(),
              [](const LoraEntry& a, const LoraEntry& b) { return a.path < b.path; });

    return k;
}

// ── ModelCacheKey::operator== ─────────────────────────────────────────────────
// Operates on already-normalised fields (canonical paths, sorted loras).
// Scales are compared via scaleKey() so representation noise ≤ 0.0005 is ignored.

bool ModelCacheKey::operator==(const ModelCacheKey& o) const noexcept {
    if (cfg.type  != o.cfg.type)  return false;
    if (modelDir  != o.modelDir)  return false;
    if (loras.size() != o.loras.size()) return false;
    for (size_t i = 0; i < loras.size(); ++i) {
        if (loras[i].path != o.loras[i].path) return false;
        if (scaleKey(loras[i].scale) != scaleKey(o.loras[i].scale)) return false;
    }
    return true;
}

// ── ModelCacheKeyHash ─────────────────────────────────────────────────────────
// Feeds a single canonical string buffer into XXH64 so the hash is:
//   • order-independent  (loras are sorted)
//   • path-independent   (canonical strings)
//   • scale-stable       (fixed-point ints written as decimal)
//
// The buffer layout is:  "<modelDir>\0<type_int>\0<path>\0<scale_int>\0..."
// Field separators ('\0') prevent adjacent fields from accidentally merging
// (e.g. "ab" + "c" vs "a" + "bc").

size_t ModelCacheKeyHash::operator()(const ModelCacheKey& k) const noexcept {
    std::string buf;
    buf.reserve(k.modelDir.size() + 16 + k.loras.size() * 64);

    buf += k.modelDir;
    buf += '\0';
    buf += std::to_string(static_cast<int>(k.cfg.type));
    buf += '\0';

    for (const auto& lo : k.loras) {
        buf += lo.path;
        buf += '\0';
        buf += std::to_string(scaleKey(lo.scale));
        buf += '\0';
    }

    return static_cast<size_t>(XXH64(buf.data(), buf.size(), 0));
}

// ── ModelManager::get ─────────────────────────────────────────────────────────

GenerationContext& ModelManager::get(const ModelConfig&            cfg,
                                      const std::string&            modelDir,
                                      const std::vector<LoraEntry>& loras) {
    const ModelCacheKey key = ModelCacheKey::make(cfg, modelDir, loras);

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
