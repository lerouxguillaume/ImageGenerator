// LoraInjector.cpp — production-safe LoRA injection via AddExternalInitializers.
//
// Base weights are read from a companion <stem>_weights.safetensors file
// written by the export script.  The C++ code never reads .onnx.data: ORT
// remains the sole component responsible for loading external tensor data.
//
// Cache design:
//   On a cache miss:  load base weights (once) + run matmuls + store CachedTensors.
//   On a cache hit:   copy CachedTensors → LoraOverrides (cheap memcpy) and return.
//
//   The backing buffers live inside LoraOverrides, which the caller keeps alive
//   until Ort::Session construction completes (ORT holds pointers into them).
#include "LoraInjector.hpp"
#include "SdLoraMatch.hpp"
#include "SdSafetensors.hpp"
#include "../../managers/Logger.hpp"
#include <cmath>
#include <filesystem>
#include <stdexcept>

namespace sd {

// ── LoraOverrides ─────────────────────────────────────────────────────────────

LoraOverrides::LoraOverrides()
    : memInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{}

// ── FNV-1a hash (self-contained, no third-party dependency) ──────────────────

static size_t fnv1a(const std::string& s, size_t seed = 14695981039346656037ULL) {
    size_t h = seed;
    for (unsigned char c : s) {
        h ^= c;
        h *= 1099511628211ULL;
    }
    return h;
}

// ── LoraInjector implementation ───────────────────────────────────────────────

void LoraInjector::loadModelMetadata(const std::string& onnxPath) {
    if (onnxPath_ == onnxPath) return;  // idempotent

    namespace fs = std::filesystem;
    Logger::info("LoraInjector: loading metadata from "
                 + fs::path(onnxPath).filename().string());

    const OnnxModelBundle bundle = resolveBundle(onnxPath);
    extIndex_       = parseExternalIndex(bundle);
    extSuffixIndex_ = buildExternalSuffixIndex(extIndex_);
    onnxPath_       = onnxPath;

    Logger::info("LoraInjector: " + std::to_string(extIndex_.size())
                 + " external initializer(s) indexed");
}

bool LoraInjector::ensureBaseWeights(const std::string& weightsPath) {
    if (loadedWeightsPath_ == weightsPath) return true;

    namespace fs = std::filesystem;
    if (!fs::exists(weightsPath)) {
        Logger::info("LoraInjector: companion weights file not found: "
                     + weightsPath
                     + "\n  Re-export with the project scripts to generate it."
                       "  LoRA injection disabled for this component.");
        return false;
    }

    Logger::info("LoraInjector: loading base weights from "
                 + fs::path(weightsPath).filename().string());

    SafetensorsMap raw;
    try {
        raw = loadSafetensors(weightsPath);
    } catch (const std::exception& e) {
        Logger::info("LoraInjector: failed to load companion weights: "
                     + std::string(e.what()));
        return false;
    }

    baseWeights_.clear();
    baseWeights_.reserve(raw.size());
    for (auto& [name, st] : raw)
        baseWeights_.emplace(name, std::move(st.data));  // st.data is already fp32

    loadedWeightsPath_ = weightsPath;
    Logger::info("LoraInjector: loaded " + std::to_string(baseWeights_.size())
                 + " base weight tensor(s)");
    return true;
}

size_t LoraInjector::makeCacheKey(const std::string&             onnxPath,
                                    const std::string&             weightsPath,
                                    const std::vector<LoraEntry>& loras) {
    // Canonical string: onnxPath\0weightsPath\0(path\0roundedScale\0)*
    std::string buf;
    buf.reserve(onnxPath.size() + weightsPath.size() + loras.size() * 80 + 2);
    buf += onnxPath;     buf += '\0';
    buf += weightsPath;  buf += '\0';
    for (const auto& lo : loras) {
        buf += lo.path;  buf += '\0';
        // Round scale to the same precision as ModelCacheKey (0.001 granularity)
        buf += std::to_string(static_cast<int>(std::lround(lo.scale * 1000)));
        buf += '\0';
    }
    return fnv1a(buf);
}

std::vector<LoraInjector::CachedTensor>
LoraInjector::computeMerge(const std::vector<LoraEntry>& loras) {
    // Per matched ONNX name: float32 accumulator (base + all LoRA deltas).
    struct MergedEntry {
        const ExternalTensorMeta* meta  = nullptr;
        std::vector<float>        w;       // float32 accumulated weights
        bool                      dirty = false;
    };
    std::unordered_map<std::string, MergedEntry> merged;

    int totalPatched = 0;
    int totalMissed  = 0;

    for (const auto& lo : loras) {
        Logger::info("  LoraInjector: applying "
                     + std::filesystem::path(lo.path).filename().string()
                     + "  scale=" + std::to_string(lo.scale));

        SafetensorsMap rawLora;
        try {
            rawLora = loadSafetensors(lo.path);
        } catch (const std::exception& e) {
            Logger::info("  LoraInjector: load failed — " + std::string(e.what())
                         + " — skipping this adapter");
            continue;
        }

        const ParsedLora parsed = parseLoraLayers(rawLora);
        int patched = 0, missed = 0;

        for (const auto& [loraBase, layer] : parsed.layers) {
            if (!layer.down || !layer.up) continue;

            // ── 1. Match LoRA key → ONNX initializer via suffix index ──────────
            const ExternalTensorMeta* meta =
                matchExternalLoraKey(extSuffixIndex_, loraBase);
            if (!meta) { ++missed; continue; }
            if (meta->dtype != 1 && meta->dtype != 10) continue;  // unsupported dtype

            // ── 2. Validate LoRA tensor shapes ────────────────────────────────
            const int64_t rank    = layer.down->shape[0];
            int64_t       in_feat = 1;
            for (size_t d = 1; d < layer.down->shape.size(); ++d)
                in_feat *= layer.down->shape[d];
            const int64_t out_feat = layer.up->shape[0];

            int64_t baseElems = 1;
            for (auto d : meta->shape) baseElems *= d;

            if (baseElems != out_feat * in_feat) {
                Logger::info("  LoraInjector: shape mismatch for '" + loraBase
                             + "' (LoRA=" + std::to_string(out_feat) + "×"
                             + std::to_string(in_feat) + ", base="
                             + std::to_string(baseElems) + ") — skipping");
                continue;
            }

            // ── 3. Exact base weight lookup by ONNX initializer name ──────────
            auto baseIt = baseWeights_.find(meta->onnxName);
            if (baseIt == baseWeights_.end()) {
                Logger::info("  LoraInjector: base weight not found: '"
                             + meta->onnxName + "' — skipping");
                ++missed;
                continue;
            }
            if (static_cast<int64_t>(baseIt->second.size()) != baseElems) {
                Logger::info("  LoraInjector: base weight size mismatch for '"
                             + meta->onnxName + "' (expected "
                             + std::to_string(baseElems) + ", got "
                             + std::to_string(baseIt->second.size())
                             + ") — skipping");
                ++missed;
                continue;
            }

            // ── 4. Accumulate delta: W += effectiveScale * (up @ down) ────────
            const float alpha =
                (layer.alpha > 0.0f) ? layer.alpha : static_cast<float>(rank);
            const float effectiveScale = lo.scale * (alpha / static_cast<float>(rank));

            auto& entry = merged[meta->onnxName];
            if (!entry.meta) {
                entry.meta = meta;
                entry.w    = baseIt->second;   // copy base weight into accumulator
            }

            const std::vector<float> delta =
                computeLoraDelta(*layer.up, *layer.down, effectiveScale);
            for (size_t k = 0; k < static_cast<size_t>(out_feat * in_feat); ++k)
                entry.w[k] += delta[k];
            entry.dirty = true;
            ++patched;
        }

        if (missed > 0)
            Logger::info("  LoraInjector: " + std::to_string(missed)
                         + " unmatched layer(s)");
        Logger::info("  LoraInjector: " + std::to_string(patched)
                     + " patched from "
                     + std::filesystem::path(lo.path).filename().string());
        totalPatched += patched;
        totalMissed  += missed;
    }

    Logger::info("LoraInjector: total " + std::to_string(totalPatched)
                 + " patch(es), " + std::to_string(totalMissed) + " miss(es)");

    // ── Validate and convert dirty entries to CachedTensors ───────────────────
    std::vector<CachedTensor> result;
    result.reserve(merged.size());

    for (auto& [name, entry] : merged) {
        if (!entry.dirty) continue;

        // NaN / Inf guard: a corrupted LoRA weight would poison inference.
        bool bad = false;
        for (float v : entry.w) {
            if (!std::isfinite(v)) { bad = true; break; }
        }
        if (bad) {
            Logger::info("  LoraInjector: NaN/Inf in merged tensor '" + name
                         + "' — skipping (corrupt LoRA?)");
            continue;
        }

        CachedTensor ct;
        ct.onnxName = entry.meta->onnxName;
        ct.shape    = entry.meta->shape;
        ct.dtype    = entry.meta->dtype;

        const int64_t nElems = static_cast<int64_t>(entry.w.size());

        if (ct.dtype == 1) {
            // float32: move buffer directly
            ct.fp32Data = std::move(entry.w);
        } else {
            // float16: convert from accumulated float32
            ct.fp16Data.resize(static_cast<size_t>(nElems));
            for (int64_t k = 0; k < nElems; ++k)
                ct.fp16Data[static_cast<size_t>(k)] =
                    floatToFp16(entry.w[static_cast<size_t>(k)]);
        }
        result.push_back(std::move(ct));
    }

    Logger::info("LoraInjector: " + std::to_string(result.size())
                 + " tensor override(s) ready");
    return result;
}

void LoraInjector::buildOverrides(const std::vector<CachedTensor>& tensors,
                                   LoraOverrides&                    out) const {
    out.names.clear();
    out.values.clear();
    out.fp16Bufs.clear();
    out.fp32Bufs.clear();

    out.names.reserve(tensors.size());
    out.values.reserve(tensors.size());

    for (const auto& ct : tensors) {
        // Final validation: name must still be present in the model index.
        // (Protects against stale cache entries if the model file is replaced.)
        std::string normName = ct.onnxName;
        for (char& c : normName) if (c == '.' || c == '/') c = '_';

        if (extIndex_.find(normName) == extIndex_.end()) {
            Logger::info("  LoraInjector: stale cache entry '" + ct.onnxName
                         + "' not in model index — skipping");
            continue;
        }

        const int64_t nElems = static_cast<int64_t>(
            ct.dtype == 1 ? ct.fp32Data.size() : ct.fp16Data.size());
        std::vector<int64_t> shape = ct.shape;

        out.names.push_back(ct.onnxName);

        if (ct.dtype == 1) {
            out.fp32Bufs.push_back(ct.fp32Data);  // copy into backing buffer
            auto& buf = out.fp32Bufs.back();
            out.values.push_back(Ort::Value::CreateTensor<float>(
                out.memInfo, buf.data(), static_cast<size_t>(nElems),
                shape.data(), shape.size()));
        } else {
            out.fp16Bufs.push_back(ct.fp16Data);  // copy into backing buffer
            auto& buf = out.fp16Bufs.back();
            out.values.push_back(Ort::Value::CreateTensor(
                out.memInfo,
                reinterpret_cast<Ort::Float16_t*>(buf.data()),
                static_cast<size_t>(nElems),
                shape.data(), shape.size()));
        }
    }
}

LoraOverrides LoraInjector::applyLoras(Ort::SessionOptions&          opts,
                                        const std::string&             weightsPath,
                                        const std::vector<LoraEntry>& loras) {
    if (onnxPath_.empty())
        throw std::logic_error(
            "LoraInjector::applyLoras called before loadModelMetadata");

    LoraOverrides result;

    if (extIndex_.empty()) {
        Logger::info("LoraInjector: no external initializers — skipping injection");
        return result;
    }

    const size_t key = makeCacheKey(onnxPath_, weightsPath, loras);

    std::lock_guard<std::mutex> lk(cacheMutex_);

    auto it = mergeCache_.find(key);
    if (it == mergeCache_.end()) {
        // Cache miss: load base weights (fast, done under lock) then compute
        // the matmuls (slow, still under lock — serialises concurrent callers).
        if (!ensureBaseWeights(weightsPath)) return result;

        auto tensors = computeMerge(loras);
        if (tensors.empty()) {
            // Store the empty result so subsequent calls skip computation.
            it = mergeCache_.emplace(key, std::vector<CachedTensor>{}).first;
            return result;
        }
        Logger::info("LoraInjector: cache miss — storing "
                     + std::to_string(tensors.size()) + " tensor(s)");
        it = mergeCache_.emplace(key, std::move(tensors)).first;
    } else {
        Logger::info("LoraInjector: cache hit ("
                     + std::to_string(it->second.size()) + " tensor(s))");
    }

    buildOverrides(it->second, result);
    if (result.empty()) return result;

    opts.AddExternalInitializers(result.names, result.values);
    Logger::info("LoraInjector: injected " + std::to_string(result.names.size())
                 + " override(s) into session options");
    return result;
}

} // namespace sd
