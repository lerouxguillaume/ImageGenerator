#include "SdOnnxPatcher.hpp"
#include "SdLoraMatch.hpp"
#include "../../managers/Logger.hpp"
#include <stdexcept>
#include <string>
#include <unordered_map>

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
//   GraphProto  field 5  → TensorProto initializer (repeated)  ← PyTorch/ONNX runtime uses 5
//               field 6  → TensorProto initializer (repeated)  ← some ONNX versions use 6
//   TensorProto field 1  → dims        (repeated varint)
//               field 2  → data_type   (varint: 1=float32, 10=float16)
//               field 8  → name        (string)
//               field 9  → raw_data    (bytes)

static bool endsWith(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

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
    int extCount   = 0;
    int graphCount = 0;
    int initCount  = 0;  // all initializers seen (inline + external)
    int f1seen     = 0;  // NodeProto entries (field 1) encountered
    int f6seen     = 0;  // TensorProto initializer entries (field 6) encountered

    // Scan ModelProto for field 7 (graph).
    size_t pos = 0;
    while (pos < total) {
        const uint64_t tag      = readVarint(data, pos, total);
        const int      fieldNum = static_cast<int>(tag >> 3);
        const int      wireType = static_cast<int>(tag & 7u);

        if (fieldNum != 7 || wireType != 2) { skipField(data, pos, total, wireType); continue; }
        ++graphCount;

        const uint64_t graphLen = readVarint(data, pos, total);
        const size_t   graphEnd = pos + static_cast<size_t>(graphLen);
        Logger::info("GraphProto: graphLen=" + std::to_string(graphLen)
                     + "  graphEnd=" + std::to_string(graphEnd)
                     + "  total=" + std::to_string(total));

        // Parse one TensorProto sub-message (initializer or Constant value).
        // Reads from [pos, tpEnd), populates the result map, updates counters.
        // 'nameHint': for Constant nodes the name comes from NodeProto.output[0]
        // rather than TensorProto.name (which is typically empty there).
        auto parseTensorProto = [&](size_t& p, size_t tpEnd, const std::string& nameHint) {
            std::string          name = nameHint;
            std::vector<int64_t> shape;
            int32_t              dtype      = 0;
            size_t               rdOffset   = 0;
            size_t               rdLength   = 0;
            bool                 hasRawData = false;
            bool                 hasExtData = false;

            while (p < tpEnd) {
                const uint64_t ttag      = readVarint(data, p, tpEnd);
                const int      tFieldNum = static_cast<int>(ttag >> 3);
                const int      tWireType = static_cast<int>(ttag & 7u);

                if (tFieldNum == 1 && tWireType == 0) {
                    shape.push_back(static_cast<int64_t>(readVarint(data, p, tpEnd)));
                } else if (tFieldNum == 1 && tWireType == 2) {
                    // packed dims
                    const uint64_t packLen = readVarint(data, p, tpEnd);
                    const size_t   packEnd = p + static_cast<size_t>(packLen);
                    while (p < packEnd)
                        shape.push_back(static_cast<int64_t>(readVarint(data, p, packEnd)));
                } else if (tFieldNum == 2 && tWireType == 0) {
                    dtype = static_cast<int32_t>(readVarint(data, p, tpEnd));
                } else if (tFieldNum == 4 && tWireType == 0) {
                    if (readVarint(data, p, tpEnd) == 2u) hasExtData = true;  // EXTERNAL
                } else if (tFieldNum == 8 && tWireType == 2) {
                    // TensorProto.name — prefer the nameHint if already set
                    const uint64_t nLen = readVarint(data, p, tpEnd);
                    if (name.empty())
                        name = std::string(reinterpret_cast<const char*>(data + p), nLen);
                    p += static_cast<size_t>(nLen);
                } else if (tFieldNum == 9 && tWireType == 2) {
                    const uint64_t rdLen = readVarint(data, p, tpEnd);
                    rdOffset   = p;
                    rdLength   = static_cast<size_t>(rdLen);
                    hasRawData = true;
                    p += rdLength;
                } else {
                    skipField(data, p, tpEnd, tWireType);
                }
            }
            p = tpEnd;

            ++initCount;
            if (hasExtData) {
                ++extCount;
            } else if (hasRawData && !name.empty()) {
                std::string norm = name;
                // Normalise both '.' and '/' to '_' so that:
                //   - dotted PyTorch names  "text_encoder.text_model.encoder..."
                //   - ONNX path names       "/text_encoder/text_model/encoder..."
                // both become underscore-separated and match kohya LoRA keys.
                for (char& c : norm) if (c == '.' || c == '/') c = '_';
                result[norm] = { rdOffset, rdLength, shape, dtype };
            }
        };

        // Diagnostic: count ALL field numbers in the GraphProto.
        // skipField jumps over bulk data, so this is fast (~1200 fields total).
        {
            size_t dpos = pos;
            std::map<int,int> fieldCounts;
            while (dpos < graphEnd) {
                const uint64_t dtag = readVarint(data, dpos, graphEnd);
                const int df = static_cast<int>(dtag >> 3);
                const int dw = static_cast<int>(dtag & 7u);
                fieldCounts[df]++;
                skipField(data, dpos, graphEnd, dw);
            }
            std::string fc;
            for (const auto& [f, c] : fieldCounts)
                fc += "f" + std::to_string(f) + "x" + std::to_string(c) + " ";
            Logger::info("ONNX graph fields (ALL): " + fc);
        }

        // Scan GraphProto for:
        //   field 6 (initializer, repeated TensorProto) — standard initializers
        //   field 1 (node,        repeated NodeProto)   — Constant ops (some exporters
        //       skip initializers and embed all weights as Constant nodes instead)
        while (pos < graphEnd) {
            const uint64_t gtag      = readVarint(data, pos, graphEnd);
            const int      gFieldNum = static_cast<int>(gtag >> 3);
            const int      gWireType = static_cast<int>(gtag & 7u);

            if (gWireType != 2) { skipField(data, pos, graphEnd, gWireType); continue; }

            const uint64_t blockLen = readVarint(data, pos, graphEnd);
            const size_t   blockEnd = pos + static_cast<size_t>(blockLen);

            if (gFieldNum == 5 || gFieldNum == 6) {
                // ── Standard initializer ──────────────────────────────────────
                // PyTorch-exported ONNX models use field 5; some other tools use field 6.
                ++f6seen;
                size_t tpPos = pos;
                parseTensorProto(tpPos, blockEnd, "");
                pos = tpPos;

            } else if (gFieldNum == 1) {
                ++f1seen;
                // ── NodeProto ─────────────────────────────────────────────────
                // Parse just enough to detect op_type == "Constant".
                // NodeProto fields:
                //   1 = input (repeated string)   2 = output (repeated string)
                //   3 = name (string)             4 = op_type (string)
                //   5 = attribute (repeated AttributeProto)
                std::string opType;
                std::string outputName;  // first output = name of the produced value

                size_t nodeScan = pos;
                while (nodeScan < blockEnd) {
                    const uint64_t ntag      = readVarint(data, nodeScan, blockEnd);
                    const int      nFieldNum = static_cast<int>(ntag >> 3);
                    const int      nWireType = static_cast<int>(ntag & 7u);
                    if (nWireType != 2) { skipField(data, nodeScan, blockEnd, nWireType); continue; }

                    const uint64_t nfLen  = readVarint(data, nodeScan, blockEnd);
                    const size_t   nfEnd  = nodeScan + static_cast<size_t>(nfLen);
                    if (nFieldNum == 2 && outputName.empty()) {
                        outputName = std::string(
                            reinterpret_cast<const char*>(data + nodeScan), nfLen);
                    } else if (nFieldNum == 4) {
                        opType = std::string(
                            reinterpret_cast<const char*>(data + nodeScan), nfLen);
                    }
                    nodeScan = nfEnd;
                }

                if (opType == "Constant") {
                    // Second pass over the NodeProto to find the "value" attribute.
                    // AttributeProto fields:
                    //   1 = name (string)   4 = type (varint)   5 = t (TensorProto)
                    size_t nodePos = pos;
                    while (nodePos < blockEnd) {
                        const uint64_t ntag2 = readVarint(data, nodePos, blockEnd);
                        const int      nf2   = static_cast<int>(ntag2 >> 3);
                        const int      nw2   = static_cast<int>(ntag2 & 7u);
                        if (nw2 != 2) { skipField(data, nodePos, blockEnd, nw2); continue; }

                        const uint64_t afLen = readVarint(data, nodePos, blockEnd);
                        const size_t   afEnd = nodePos + static_cast<size_t>(afLen);

                        if (nf2 == 5) {
                            // AttributeProto — scan for name=="value" and field 5 (t)
                            std::string attrName;
                            size_t attrScan = nodePos;
                            while (attrScan < afEnd) {
                                const uint64_t atag = readVarint(data, attrScan, afEnd);
                                const int      af   = static_cast<int>(atag >> 3);
                                const int      aw   = static_cast<int>(atag & 7u);
                                if (aw == 2) {
                                    const uint64_t avLen = readVarint(data, attrScan, afEnd);
                                    if (af == 1) {  // AttributeProto.name
                                        attrName = std::string(
                                            reinterpret_cast<const char*>(data + attrScan), avLen);
                                    }
                                    attrScan += static_cast<size_t>(avLen);
                                } else {
                                    skipField(data, attrScan, afEnd, aw);
                                }
                            }
                            if (attrName == "value") {
                                // Re-scan the same AttributeProto for field 5 (t = TensorProto).
                                attrScan = nodePos;
                                while (attrScan < afEnd) {
                                    const uint64_t atag = readVarint(data, attrScan, afEnd);
                                    const int      af   = static_cast<int>(atag >> 3);
                                    const int      aw   = static_cast<int>(atag & 7u);
                                    if (aw == 2) {
                                        const uint64_t avLen = readVarint(data, attrScan, afEnd);
                                        const size_t   avEnd = attrScan + static_cast<size_t>(avLen);
                                        if (af == 5) {  // AttributeProto.t (TensorProto)
                                            size_t tpPos = attrScan;
                                            parseTensorProto(tpPos, avEnd, outputName);
                                            attrScan = tpPos;
                                        } else {
                                            attrScan = avEnd;
                                        }
                                    } else {
                                        skipField(data, attrScan, afEnd, aw);
                                    }
                                }
                            }
                            nodePos = afEnd;
                        } else {
                            nodePos = afEnd;
                        }
                    }
                    pos = blockEnd;
                } else {
                    pos = blockEnd;
                }
            } else {
                pos = blockEnd;
            }
        }
        if (pos != graphEnd)
            Logger::info("  WARNING: inner loop ended at pos=" + std::to_string(pos)
                         + " expected graphEnd=" + std::to_string(graphEnd)
                         + " (diff=" + std::to_string(static_cast<int64_t>(graphEnd) - static_cast<int64_t>(pos)) + ")");
        pos = graphEnd;
    }

    Logger::info("ONNX patcher: field1_nodes=" + std::to_string(f1seen)
                 + "  field6_initializers=" + std::to_string(f6seen));
    Logger::info("ONNX patcher: graphs=" + std::to_string(graphCount)
                 + "  tensors_found=" + std::to_string(initCount)
                 + "  inline=" + std::to_string(result.size())
                 + "  external=" + std::to_string(extCount)
                 + "  (includes Constant nodes)");
    if (graphCount == 0)
        Logger::info("  WARNING: no GraphProto (field 7) found in ModelProto — "
                     "file may be corrupted or use an unsupported ONNX variant");
    if (extCount > 0)
        Logger::info("  " + std::to_string(extCount) +
                     " initialiser(s) use external data — LoRA skipped for those layers");
    // Legacy single-line for backward log parsing.
    Logger::info("ONNX patcher: indexed " + std::to_string(result.size()) + " initialisers.");

    // Log sample names so the user can verify normalisation matches kohya LoRA keys.
    // Show the first 5 and then 5 names that contain "weight" (more diagnostic value).
    int sample = 0;
    for (const auto& [norm, _] : result) {
        Logger::info("  ONNX index sample: " + norm);
        if (++sample >= 5) break;
    }
    int wsample = 0;
    for (const auto& [norm, _] : result) {
        if (norm.find("weight") != std::string::npos) {
            Logger::info("  ONNX weight sample: " + norm);
            if (++wsample >= 5) break;
        }
    }
    return result;
}

// ── buildSuffixIndex ─────────────────────────────────────────────────────────

OnnxSuffixIndex buildSuffixIndex(const OnnxTensorIndex& index) {
    OnnxSuffixIndex result;
    result.reserve(index.size() * 6); // realistic average

    for (auto it = index.cbegin(); it != index.cend(); ++it) {
        const std::string& name = it->first;

        size_t pos = 0;
        while (pos < name.size()) {
            std::string suffix = name.substr(pos);

            // Count segments (avoid useless suffixes like "weight")
            int underscoreCount = 0;
            for (char c : suffix) if (c == '_') ++underscoreCount;

            if (underscoreCount >= 1) { // keep only meaningful suffixes
                result[suffix].push_back({
                    it,
                    suffix.size()
                });
            }

            const size_t next = name.find('_', pos);
            if (next == std::string::npos)
                break;

            pos = next + 1;
        }
    }

    return result;
}

// ── computeLoraDelta ─────────────────────────────────────────────────────────

std::vector<float> computeLoraDelta(const SafeTensor& up,
                                    const SafeTensor& down,
                                    float             effectiveScale) {
    const int64_t rank    = down.shape[0];
    int64_t       in_feat = 1;
    for (size_t d = 1; d < down.shape.size(); ++d) in_feat *= down.shape[d];
    const int64_t out_feat = up.shape[0];

    std::vector<float> delta(static_cast<size_t>(out_feat * in_feat), 0.0f);
    const float* upData   = up.data.data();
    const float* downData = down.data.data();

    for (int64_t o = 0; o < out_feat; ++o)
        for (int64_t r = 0; r < rank; ++r) {
            const float u = upData[o * rank + r];
            if (u == 0.0f) continue;
            for (int64_t i = 0; i < in_feat; ++i)
                delta[static_cast<size_t>(o * in_feat + i)] += u * downData[r * in_feat + i];
        }
    for (float& v : delta) v *= effectiveScale;
    return delta;
}

// ── parseLoraLayers ───────────────────────────────────────────────────────────

ParsedLora parseLoraLayers(const SafetensorsMap& lora) {
    ParsedLora result;

    for (const auto& [key, tensor] : lora) {
        std::string body;
        // Strip known kohya LoRA prefixes.
        // lora_unet_  → UNet layers
        // lora_te_    → text encoder 1 (SD 1.5 and SDXL encoder-1 / CLIP-L)
        // lora_te2_   → text encoder 2 (SDXL OpenCLIP-G)
        if      (key.rfind("lora_unet_", 0) == 0) body = key.substr(10);
        else if (key.rfind("lora_te2_",  0) == 0) body = key.substr(9);
        else if (key.rfind("lora_te_",   0) == 0) body = key.substr(8);
        else continue;

        if (endsWith(body, ".lora_down.weight"))
            result.layers[body.substr(0, body.size() - 17)].down = &tensor;
        else if (endsWith(body, ".lora_up.weight"))
            result.layers[body.substr(0, body.size() - 15)].up = &tensor;
        else if (endsWith(body, ".alpha") && !tensor.data.empty())
            result.layers[body.substr(0, body.size() - 6)].alpha = tensor.data[0];
    }

    return result;
}

// ── applyLoraToBytes ──────────────────────────────────────────────────────────

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
