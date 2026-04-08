#include "SdOnnxPatcher.hpp"
#include "../../managers/Logger.hpp"
#include <stdexcept>
#include <string>

namespace sd {

// ── Protobuf wire-format primitives ──────────────────────────────────────────
// ONNX is a standard protobuf binary. Only the subset needed to locate
// TensorProto initializers and read their raw_data is implemented here.
//
// Wire types used:
//   0 = varint        (field_num << 3 | 0)
//   2 = LEN-delimited (field_num << 3 | 2)  — used for strings, bytes, sub-messages
//
// Relevant field numbers:
//   ModelProto  field 7  → GraphProto
//   GraphProto  field 6  → TensorProto (repeated)
//   TensorProto field 1  → dims        (repeated varint)
//               field 2  → data_type   (varint: 1=float32, 10=float16)
//               field 4  → data_location (varint: 2=EXTERNAL)
//               field 8  → name        (string)
//               field 9  → raw_data    (bytes)

static uint64_t readVarint(const uint8_t* data, size_t& pos, size_t end) {
    uint64_t result = 0;
    int      shift  = 0;
    while (pos < end) {
        const uint8_t b = data[pos++];
        result |= static_cast<uint64_t>(b & 0x7Fu) << shift;
        if (!(b & 0x80u)) return result;
        shift += 7;
        if (shift >= 64) throw std::runtime_error("ONNX parse: varint overflow");
    }
    throw std::runtime_error("ONNX parse: truncated varint");
}

static void skipField(const uint8_t* data, size_t& pos, size_t end, int wireType) {
    switch (wireType) {
        case 0: readVarint(data, pos, end); break;
        case 1: pos += 8; break;
        case 2: { const uint64_t len = readVarint(data, pos, end); pos += len; break; }
        case 5: pos += 4; break;
        default: throw std::runtime_error("ONNX parse: unknown wire type " + std::to_string(wireType));
    }
}

// ── parseTensorIndex ──────────────────────────────────────────────────────────

OnnxTensorIndex parseTensorIndex(const std::vector<uint8_t>& onnxBytes) {
    const uint8_t* data  = onnxBytes.data();
    const size_t   total = onnxBytes.size();

    OnnxTensorIndex result;
    int extCount = 0;

    // Scan ModelProto for field 7 (graph).
    size_t pos = 0;
    while (pos < total) {
        const uint64_t tag      = readVarint(data, pos, total);
        const int      fieldNum = static_cast<int>(tag >> 3);
        const int      wireType = static_cast<int>(tag & 7u);

        if (fieldNum != 7 || wireType != 2) { skipField(data, pos, total, wireType); continue; }

        const uint64_t graphLen = readVarint(data, pos, total);
        const size_t   graphEnd = pos + static_cast<size_t>(graphLen);

        // Scan GraphProto for field 6 (initializer, repeated).
        while (pos < graphEnd) {
            const uint64_t gtag      = readVarint(data, pos, graphEnd);
            const int      gFieldNum = static_cast<int>(gtag >> 3);
            const int      gWireType = static_cast<int>(gtag & 7u);

            if (gFieldNum != 6 || gWireType != 2) { skipField(data, pos, graphEnd, gWireType); continue; }

            const uint64_t tpLen = readVarint(data, pos, graphEnd);
            const size_t   tpEnd = pos + static_cast<size_t>(tpLen);

            // Parse TensorProto fields.
            std::string          name;
            std::vector<int64_t> shape;
            int32_t              dtype      = 0;
            size_t               rdOffset   = 0;
            size_t               rdLength   = 0;
            bool                 hasRawData = false;
            bool                 hasExtData = false;

            while (pos < tpEnd) {
                const uint64_t ttag      = readVarint(data, pos, tpEnd);
                const int      tFieldNum = static_cast<int>(ttag >> 3);
                const int      tWireType = static_cast<int>(ttag & 7u);

                if (tFieldNum == 1 && tWireType == 0) {
                    // dims — repeated varint, one tag+value pair per dimension.
                    shape.push_back(static_cast<int64_t>(readVarint(data, pos, tpEnd)));
                } else if (tFieldNum == 1 && tWireType == 2) {
                    // dims — packed encoding (proto3 may use this).
                    const uint64_t packLen = readVarint(data, pos, tpEnd);
                    const size_t   packEnd = pos + static_cast<size_t>(packLen);
                    while (pos < packEnd)
                        shape.push_back(static_cast<int64_t>(readVarint(data, pos, packEnd)));
                } else if (tFieldNum == 2 && tWireType == 0) {
                    dtype = static_cast<int32_t>(readVarint(data, pos, tpEnd));
                } else if (tFieldNum == 4 && tWireType == 0) {
                    if (readVarint(data, pos, tpEnd) == 2u) hasExtData = true;  // EXTERNAL
                } else if (tFieldNum == 8 && tWireType == 2) {
                    const uint64_t nLen = readVarint(data, pos, tpEnd);
                    name = std::string(reinterpret_cast<const char*>(data + pos), nLen);
                    pos += static_cast<size_t>(nLen);
                } else if (tFieldNum == 9 && tWireType == 2) {
                    const uint64_t rdLen = readVarint(data, pos, tpEnd);
                    rdOffset   = pos;
                    rdLength   = static_cast<size_t>(rdLen);
                    hasRawData = true;
                    pos += rdLength;
                } else {
                    skipField(data, pos, tpEnd, tWireType);
                }
            }
            pos = tpEnd;

            if (hasExtData) {
                ++extCount;
            } else if (hasRawData && !name.empty()) {
                // Normalise name: dots → underscores (matches kohya LoRA key format).
                std::string norm = name;
                for (char& c : norm) if (c == '.') c = '_';
                result[norm] = { rdOffset, rdLength, shape, dtype };
            }
        }
        pos = graphEnd;
    }

    if (extCount > 0)
        Logger::info("ONNX patcher: " + std::to_string(extCount) +
                     " initialiser(s) use external data — LoRA skipped for those layers.");
    Logger::info("ONNX patcher: indexed " + std::to_string(result.size()) + " initialisers.");

    // Log a few sample names so the user can verify the dot→underscore normalisation
    // matches the kohya LoRA key format if 0 layers end up being patched.
    int sample = 0;
    for (const auto& [norm, _] : result) {
        Logger::info("  ONNX index sample: " + norm);
        if (++sample >= 3) break;
    }
    return result;
}

// ── applyLoraToBytes ──────────────────────────────────────────────────────────

static bool endsWith(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

int applyLoraToBytes(std::vector<uint8_t>&  onnxBytes,
                     const OnnxTensorIndex& index,
                     const SafetensorsMap&  lora,
                     float                  userScale) {
    // Group safetensors keys into (down, up, alpha) triplets by base layer name.
    // Supports both lora_unet_ (UNet) and lora_te_ (text encoder) prefixes.
    struct Layer {
        const SafeTensor* down  = nullptr;
        const SafeTensor* up    = nullptr;
        float             alpha = 0.0f;  // 0 → default to rank
    };
    std::map<std::string, Layer> layers;

    for (const auto& [key, tensor] : lora) {
        std::string body;
        if      (key.rfind("lora_unet_", 0) == 0) body = key.substr(10);
        else if (key.rfind("lora_te_",   0) == 0) body = key.substr(8);
        else continue;

        if (endsWith(body, ".lora_down.weight"))
            layers[body.substr(0, body.size() - 17)].down = &tensor;
        else if (endsWith(body, ".lora_up.weight"))
            layers[body.substr(0, body.size() - 15)].up = &tensor;
        else if (endsWith(body, ".alpha") && !tensor.data.empty())
            layers[body.substr(0, body.size() - 6)].alpha = tensor.data[0];
    }

    // Count complete pairs (both down + up present).
    int completePairs = 0;
    for (const auto& [base, layer] : layers)
        if (layer.down && layer.up) ++completePairs;

    Logger::info("LoRA apply: " + std::to_string(completePairs) + " complete layer pair(s) in file"
                 + "  (scale=" + std::to_string(userScale) + ")");

    int patchCount  = 0;
    int missCount   = 0;

    for (const auto& [loraBase, layer] : layers) {
        if (!layer.down || !layer.up) {
            Logger::info("  LoRA incomplete pair (missing down or up): " + loraBase);
            continue;
        }

        // Lookup in ONNX index: LoRA base name + "_weight" suffix.
        const auto it = index.find(loraBase + "_weight");
        if (it == index.end()) {
            // Log the first few misses so the user can diagnose name-mapping issues.
            if (missCount < 5)
                Logger::info("  LoRA no match: " + loraBase + "_weight");
            ++missCount;
            continue;
        }

        const TensorIndex& ti = it->second;
        if (ti.dtype != 1 && ti.dtype != 10) continue;  // only fp32 / fp16

        // in_feat = product of all dims after the rank dim.
        const int64_t rank     = layer.down->shape[0];
        int64_t       in_feat  = 1;
        for (size_t d = 1; d < layer.down->shape.size(); ++d) in_feat *= layer.down->shape[d];
        const int64_t out_feat = layer.up->shape[0];

        // Validate that the delta shape matches the base weight element count.
        int64_t baseElems = 1;
        for (auto d : ti.shape) baseElems *= d;
        if (baseElems != out_feat * in_feat) {
            Logger::info("LoRA shape mismatch: " + loraBase + " — skipping.");
            continue;
        }

        // effective_scale = userScale * alpha / rank  (alpha defaults to rank if absent)
        const float alpha          = (layer.alpha > 0.0f) ? layer.alpha : static_cast<float>(rank);
        const float effectiveScale = userScale * (alpha / static_cast<float>(rank));

        // Compute delta = effectiveScale * (lora_up @ lora_down) in fp32.
        // lora_up:   [out_feat, rank]
        // lora_down: [rank, in_feat]
        std::vector<float> delta(static_cast<size_t>(out_feat * in_feat), 0.0f);
        const float* upData   = layer.up->data.data();
        const float* downData = layer.down->data.data();

        for (int64_t o = 0; o < out_feat; ++o)
            for (int64_t r = 0; r < rank; ++r) {
                const float u = upData[o * rank + r];
                if (u == 0.0f) continue;
                for (int64_t i = 0; i < in_feat; ++i)
                    delta[static_cast<size_t>(o * in_feat + i)] += u * downData[r * in_feat + i];
            }
        for (float& v : delta) v *= effectiveScale;

        // Patch raw_data in-place (same byte count — only values change, not size).
        uint8_t* raw = onnxBytes.data() + ti.rawDataOffset;

        if (ti.dtype == 1) {
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
        Logger::info("  LoRA patched: " + loraBase +
                     " [" + std::to_string(out_feat) + "x" + std::to_string(in_feat) + "]"
                     + "  rank=" + std::to_string(rank)
                     + "  effective_scale=" + std::to_string(effectiveScale));
    }

    if (missCount > 5)
        Logger::info("  LoRA no match: ..." + std::to_string(missCount - 5) + " more (total unmatched: "
                     + std::to_string(missCount) + ")");

    Logger::info("LoRA apply summary: " + std::to_string(patchCount) + "/" +
                 std::to_string(completePairs) + " layers patched.");
    return patchCount;
}

} // namespace sd
