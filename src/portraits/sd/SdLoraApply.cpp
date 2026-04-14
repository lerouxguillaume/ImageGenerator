// SdLoraApply.cpp: builds per-tensor LoRA overrides for AddExternalInitializers.
//
// Flow:
//   1. parseExternalIndex() gave us the external tensor metadata (name/shape/dtype/offset/length).
//   2. We group the safetensors LoRA keys into (down, up, alpha) triplets via parseLoraLayers().
//   3. For each LoRA layer, matchExternalLoraKey() finds the matching ONNX initializer.
//   4. We read the base weight bytes from .onnx.data, upcast to float32, apply the delta,
//      downcast back to the model dtype, and store the result.
//   5. The caller creates Ort::Value non-owning views over those buffers and passes them
//      to SessionOptions::AddExternalInitializers before creating the session.
#include "SdLoraApply.hpp"
#include "SdLoraMatch.hpp"
#include "SdSafetensors.hpp"
#include "../../managers/Logger.hpp"
#include <fstream>
#include <stdexcept>

namespace sd {

LegacyLoraOverrides::LegacyLoraOverrides()
    : memInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{}

LegacyLoraOverrides buildLoraOverrides(const OnnxModelBundle&          bundle,
                                 const OnnxExternalIndex&        extIndex,
                                 const OnnxExternalSuffixIndex&  extSuffixIndex,
                                 const std::vector<LoraEntry>&   loras) {
    LegacyLoraOverrides result;

    if (extIndex.empty()) {
        Logger::info("  buildLoraOverrides: no external tensors in "
                     + bundle.onnxPath.filename().string() + " — skipping");
        return result;
    }

    if (!bundle.hasExternalData()) {
        Logger::info("  buildLoraOverrides: no .onnx.data for "
                     + bundle.onnxPath.filename().string() + " — skipping");
        return result;
    }

    const std::string dataPath = bundle.dataPath.string();
    std::ifstream dataFile(dataPath, std::ios::binary);
    if (!dataFile)
        throw std::runtime_error("buildLoraOverrides: cannot open data file: " + dataPath);

    // For each external tensor that gets at least one LoRA patch, we accumulate a
    // merged float32 buffer.  We defer creating the Ort::Value until all LoRAs for
    // that tensor have been applied, then convert once to the target dtype.
    //
    // merged[normName] = { float32 buffer, ExternalTensorMeta* }
    struct MergedEntry {
        const ExternalTensorMeta* meta    = nullptr;
        std::vector<float>        weights; // float32 accumulated base + deltas
        bool                      dirty   = false;
    };
    std::map<std::string, MergedEntry> merged;

    auto getOrLoadBase = [&](const std::string& normName,
                              const ExternalTensorMeta& meta) -> MergedEntry& {
        auto it = merged.find(normName);
        if (it != merged.end()) return it->second;

        MergedEntry entry;
        entry.meta = &meta;

        // Read raw bytes from .onnx.data
        dataFile.seekg(static_cast<std::streamoff>(meta.dataOffset));
        const size_t byteLen = static_cast<size_t>(meta.dataLength);
        std::vector<uint8_t> raw(byteLen);
        dataFile.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(byteLen));
        if (!dataFile)
            throw std::runtime_error(
                "buildLoraOverrides: short read from " + dataPath
                + " (offset=" + std::to_string(meta.dataOffset)
                + ", len="    + std::to_string(byteLen) + ")");

        int64_t nElems = 1;
        for (auto d : meta.shape) nElems *= d;
        entry.weights.resize(static_cast<size_t>(nElems));

        if (meta.dtype == 1) {
            // float32
            std::memcpy(entry.weights.data(), raw.data(), byteLen);
        } else if (meta.dtype == 10) {
            // float16 → float32
            const auto* src = reinterpret_cast<const uint16_t*>(raw.data());
            for (int64_t k = 0; k < nElems; ++k)
                entry.weights[static_cast<size_t>(k)] = fp16ToFloat(src[k]);
        } else {
            // Unsupported dtype — leave zeros; LoRA delta will still be applied.
            Logger::info("  buildLoraOverrides: unsupported dtype=" + std::to_string(meta.dtype)
                         + " for " + normName + " — base left as zeros");
        }

        return merged.emplace(normName, std::move(entry)).first->second;
    };

    int totalPatches = 0;
    int totalMiss    = 0;

    for (const auto& lo : loras) {
        Logger::info("  Applying LoRA (AddExternalInitializers): " + lo.path
                     + "  scale=" + std::to_string(lo.scale));

        SafetensorsMap rawLoraMap;
        try { rawLoraMap = loadSafetensors(lo.path); }
        catch (const std::exception& e) {
            Logger::info("  LoRA '" + lo.path + "' load failed: " + e.what());
            continue;
        }

        Logger::info("  LoRA tensors: " + std::to_string(rawLoraMap.size()));

        const ParsedLora parsed = parseLoraLayers(rawLoraMap);
        int patchCount = 0;
        int missCount  = 0;

        for (const auto& [loraBase, layer] : parsed.layers) {
            if (!layer.down || !layer.up) continue;

            const ExternalTensorMeta* meta = matchExternalLoraKey(extSuffixIndex, loraBase);
            if (!meta) { ++missCount; continue; }
            if (meta->dtype != 1 && meta->dtype != 10) continue;

            // Validate shape compatibility
            const int64_t rank    = layer.down->shape[0];
            int64_t       in_feat = 1;
            for (size_t d = 1; d < layer.down->shape.size(); ++d) in_feat *= layer.down->shape[d];
            const int64_t out_feat = layer.up->shape[0];

            int64_t baseElems = 1;
            for (auto d : meta->shape) baseElems *= d;
            if (baseElems != out_feat * in_feat) {
                Logger::info("  LoRA shape mismatch: " + loraBase + " — skipping");
                continue;
            }

            const float alpha          = (layer.alpha > 0.0f) ? layer.alpha : static_cast<float>(rank);
            const float effectiveScale = lo.scale * (alpha / static_cast<float>(rank));

            // Find normName in extIndex to get the key
            // meta->onnxName → need normName for merged map key
            std::string normName = meta->onnxName;
            for (char& c : normName) if (c == '.' || c == '/') c = '_';

            MergedEntry& entry = getOrLoadBase(normName, *meta);

            const std::vector<float> delta = computeLoraDelta(*layer.up, *layer.down, effectiveScale);
            for (size_t k = 0; k < static_cast<size_t>(out_feat * in_feat); ++k)
                entry.weights[k] += delta[k];
            entry.dirty = true;
            ++patchCount;
        }

        if (missCount > 0)
            Logger::info("  LoRA unmatched layers: " + std::to_string(missCount));
        Logger::info("  Patched: " + std::to_string(patchCount) + " layer(s)");
        totalPatches += patchCount;
        totalMiss    += missCount;
    }

    Logger::info("buildLoraOverrides: " + bundle.onnxPath.filename().string()
                 + " — " + std::to_string(totalPatches) + " patch(es), "
                 + std::to_string(totalMiss) + " miss(es)");

    // Build Ort::Value views for dirty (patched) entries only.
    // Allocate backing storage in the LoraOverrides; create non-owning views over it.
    for (auto& [normName, entry] : merged) {
        if (!entry.dirty) continue;

        std::vector<int64_t> shape = entry.meta->shape;
        const int64_t nElems = static_cast<int64_t>(entry.weights.size());

        result.names.push_back(entry.meta->onnxName);

        if (entry.meta->dtype == 1) {
            // float32 — move the buffer directly
            result.fp32Bufs.emplace_back(std::move(entry.weights));
            auto& buf = result.fp32Bufs.back();
            result.values.push_back(Ort::Value::CreateTensor<float>(
                result.memInfo, buf.data(), static_cast<size_t>(nElems),
                shape.data(), shape.size()));
        } else {
            // float16 — convert back from float32
            result.fp16Bufs.emplace_back(static_cast<size_t>(nElems));
            auto& buf = result.fp16Bufs.back();
            for (int64_t k = 0; k < nElems; ++k)
                buf[static_cast<size_t>(k)] = floatToFp16(entry.weights[static_cast<size_t>(k)]);
            result.values.push_back(Ort::Value::CreateTensor(
                result.memInfo,
                reinterpret_cast<Ort::Float16_t*>(buf.data()),
                static_cast<size_t>(nElems),
                shape.data(), shape.size()));
        }
    }

    Logger::info("buildLoraOverrides: " + std::to_string(result.names.size())
                 + " tensor override(s) prepared");
    return result;
}

} // namespace sd
