// applyLoraToBytes: copies an ONNX buffer and patches all matched LoRA layers.
#include "SdOnnxPatcher.hpp"
#include "SdLoraMatch.hpp"
#include "../../managers/Logger.hpp"

namespace sd {

PatchResult applyLoraToBytes(std::shared_ptr<const std::vector<uint8_t>> onnxBytes,
                              const OnnxSuffixIndex&                      suffixIndex,
                              const SafetensorsMap&                       lora,
                              float                                       userScale) {
    // Copy the immutable input into a new writable buffer.
    auto out = std::make_shared<std::vector<uint8_t>>(*onnxBytes);

    const ParsedLora parsed = parseLoraLayers(lora);
    const auto& layers      = parsed.layers;

    // Count complete pairs (both down + up present).
    int completePairs = 0;
    for (const auto& [base, layer] : layers)
        if (layer.down && layer.up) ++completePairs;

    Logger::info("LoRA apply: " + std::to_string(completePairs) + " complete layer pair(s) in file"
                 + "  (scale=" + std::to_string(userScale) + ")");

    int patchCount    = 0;
    int missCount     = 0;
    int tePatchCount  = 0;
    int unetPatchCount = 0;

    for (const auto& [loraBase, layer] : layers) {
        if (!layer.down || !layer.up) {
            Logger::info("  LoRA incomplete pair (missing down or up): " + loraBase);
            continue;
        }

        const TensorIndex* ti = matchLoraKey(suffixIndex, loraBase);
        if (!ti) {
            if (missCount < 5)
                Logger::info("  LoRA no match: " + loraBase + " (_weight / _bias)");
            ++missCount;
            continue;
        }
        if (ti->dtype != 1 && ti->dtype != 10) continue;  // only fp32 / fp16

        // Validate that the delta shape matches the base weight element count.
        const int64_t rank     = layer.down->shape[0];
        int64_t       in_feat  = 1;
        for (size_t d = 1; d < layer.down->shape.size(); ++d) in_feat *= layer.down->shape[d];
        const int64_t out_feat = layer.up->shape[0];

        int64_t baseElems = 1;
        for (auto d : ti->shape) baseElems *= d;
        if (baseElems != out_feat * in_feat) {
            Logger::info("LoRA shape mismatch: " + loraBase + " — skipping.");
            continue;
        }

        // effective_scale = userScale * alpha / rank  (alpha defaults to rank if absent)
        const float alpha          = (layer.alpha > 0.0f) ? layer.alpha : static_cast<float>(rank);
        const float effectiveScale = userScale * (alpha / static_cast<float>(rank));

        const std::vector<float> delta = computeLoraDelta(*layer.up, *layer.down, effectiveScale);

        // Patch raw_data in the output copy (same byte count — only values change, not size).
        uint8_t* raw = out->data() + ti->rawDataOffset;

        if (ti->dtype == 1) {
            // float32 base
            auto* base = reinterpret_cast<float*>(raw);
            for (int64_t k = 0; k < out_feat * in_feat; ++k)
                base[k] += delta[static_cast<size_t>(k)];
        } else {
            // float16 base: upcast → add → downcast
            auto* base = reinterpret_cast<uint16_t*>(raw);
            for (int64_t k = 0; k < out_feat * in_feat; ++k)
                base[k] = floatToFp16(fp16ToFloat(base[k]) + delta[static_cast<size_t>(k)]);
        }

        ++patchCount;
        if (loraBase.rfind("text_model_", 0) == 0) ++tePatchCount;
        else                                        ++unetPatchCount;
    }

    if (missCount > 5)
        Logger::info("  LoRA no match: ..." + std::to_string(missCount - 5) + " more (total unmatched: "
                     + std::to_string(missCount) + ")");

    Logger::info("LoRA apply summary: " + std::to_string(patchCount) + "/" +
                 std::to_string(completePairs) + " layers patched"
                 + "  (TE=" + std::to_string(tePatchCount)
                 + " UNet=" + std::to_string(unetPatchCount) + ")");
    return {out, patchCount};
}

} // namespace sd
