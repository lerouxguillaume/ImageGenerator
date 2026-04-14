// OnnxParser.cpp: ONNX protobuf binary reader.
//
// Public interface (declared in SdOnnxPatcher.hpp):
//   resolveBundle()      — filesystem-only bundle detection
//   parseExternalIndex() — metadata index of all external-data initializers
//
// The ONNX C++ protobuf library (onnx.pb.h / libonnx) is NOT linked in this
// project: the ORT pre-built package does not ship it.  The binary parsing
// below is the sole implementation path; wire-format details do not leak to callers.
#include "SdOnnxPatcher.hpp"
#include "../../managers/Logger.hpp"
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace sd {

// ── Bundle resolution ─────────────────────────────────────────────────────────

OnnxModelBundle resolveBundle(const std::filesystem::path& onnxPath) {
    if (!std::filesystem::exists(onnxPath))
        throw std::runtime_error("resolveBundle: model file not found: " + onnxPath.string());

    OnnxModelBundle bundle;
    bundle.onnxPath = onnxPath;

    const auto dataPath = std::filesystem::path(onnxPath.string() + ".data");
    if (std::filesystem::exists(dataPath))
        bundle.dataPath = dataPath;

    Logger::info("  [bundle] " + onnxPath.filename().string()
                 + (bundle.hasExternalData() ? " + .data (external)" : " (inline)"));
    return bundle;
}

// ── Protobuf wire-format primitives ──────────────────────────────────────────
// Wire types used:
//   0 = varint        (field_num << 3 | 0)
//   2 = LEN-delimited (field_num << 3 | 2)
//
// Relevant field numbers for parseExternalIndex:
//   ModelProto  field 7  → GraphProto
//   GraphProto  field 5  → TensorProto initializer (PyTorch/ORT exports use 5)
//               field 6  → TensorProto initializer (some ONNX tools use 6)
//   TensorProto field 1  → dims        (repeated varint)
//               field 2  → data_type   (varint: 1=float32, 10=float16)
//               field 8  → name        (string)
//               field 13 → external_data (repeated StringStringEntryProto: location/offset/length)
//               field 14 → data_location (varint: DEFAULT=0, EXTERNAL=1)

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

// ── parseExternalIndex ────────────────────────────────────────────────────────
// Parses the .onnx file (graph structure only) and returns one ExternalTensorMeta
// entry per initializer whose data_location == EXTERNAL (field 14, value 1).

OnnxExternalIndex parseExternalIndex(const OnnxModelBundle& bundle) {
    const std::string onnxPathStr = bundle.onnxPath.string();
    std::ifstream f(onnxPathStr, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("parseExternalIndex: cannot open " + onnxPathStr);
    const auto sz = static_cast<size_t>(f.tellg());
    f.seekg(0);
    std::vector<uint8_t> onnxBytes(sz);
    f.read(reinterpret_cast<char*>(onnxBytes.data()), static_cast<std::streamsize>(sz));

    const uint8_t* data  = onnxBytes.data();
    const size_t   total = onnxBytes.size();

    OnnxExternalIndex result;

    // ── Locate GraphProto (ModelProto field 7) ────────────────────────────────
    size_t graphBodyStart = SIZE_MAX;
    size_t graphBodyEnd   = SIZE_MAX;
    {
        size_t pos = 0;
        while (pos < total) {
            const uint64_t tag = readVarint(data, pos, total);
            const int fn = static_cast<int>(tag >> 3);
            const int wt = static_cast<int>(tag & 7u);
            if (wt == 2) {
                const uint64_t blen = readVarint(data, pos, total);
                const size_t   bend = pos + static_cast<size_t>(blen);
                if (fn == 7) { graphBodyStart = pos; graphBodyEnd = bend; break; }
                pos = bend;
            } else { skipField(data, pos, total, wt); }
        }
    }
    if (graphBodyStart == SIZE_MAX)
        throw std::runtime_error("parseExternalIndex: no GraphProto in " + onnxPathStr);

    // ── Scan GraphProto for initializer fields (5 and 6) ─────────────────────
    size_t pos = graphBodyStart;
    while (pos < graphBodyEnd) {
        const uint64_t gtag = readVarint(data, pos, graphBodyEnd);
        const int gfn = static_cast<int>(gtag >> 3);
        const int gwt = static_cast<int>(gtag & 7u);
        if (gwt != 2) { skipField(data, pos, graphBodyEnd, gwt); continue; }

        const uint64_t blockLen = readVarint(data, pos, graphBodyEnd);
        const size_t   bodyEnd  = pos + static_cast<size_t>(blockLen);

        if (gfn == 5 || gfn == 6) {
            // ── Parse TensorProto ─────────────────────────────────────────────
            std::string          name;
            std::vector<int64_t> shape;
            int32_t              dtype    = 0;
            int64_t              offset   = 0;
            int64_t              length   = -1;
            std::string          location;
            bool                 isExt    = false;

            size_t p = pos;
            while (p < bodyEnd) {
                const uint64_t ttag = readVarint(data, p, bodyEnd);
                const int tf = static_cast<int>(ttag >> 3);
                const int tw = static_cast<int>(ttag & 7u);

                if (tf == 1 && tw == 0) {
                    shape.push_back(static_cast<int64_t>(readVarint(data, p, bodyEnd)));
                } else if (tf == 1 && tw == 2) {
                    // packed dims
                    const uint64_t plen = readVarint(data, p, bodyEnd);
                    const size_t   pend = p + static_cast<size_t>(plen);
                    while (p < pend)
                        shape.push_back(static_cast<int64_t>(readVarint(data, p, pend)));
                } else if (tf == 2 && tw == 0) {
                    dtype = static_cast<int32_t>(readVarint(data, p, bodyEnd));
                } else if (tf == 8 && tw == 2) {
                    const uint64_t nlen = readVarint(data, p, bodyEnd);
                    name = std::string(reinterpret_cast<const char*>(data + p), nlen);
                    p += static_cast<size_t>(nlen);
                } else if (tf == 13 && tw == 2) {
                    // external_data: StringStringEntryProto {key, value}
                    const uint64_t elen = readVarint(data, p, bodyEnd);
                    const size_t   eend = p + static_cast<size_t>(elen);
                    std::string key, val;
                    while (p < eend) {
                        const uint64_t etag = readVarint(data, p, eend);
                        const int ef = static_cast<int>(etag >> 3);
                        const int ew = static_cast<int>(etag & 7u);
                        if (ew == 2) {
                            const uint64_t slen = readVarint(data, p, eend);
                            std::string s(reinterpret_cast<const char*>(data + p), slen);
                            p += static_cast<size_t>(slen);
                            if (ef == 1) key = std::move(s);
                            else if (ef == 2) val = std::move(s);
                        } else { skipField(data, p, eend, ew); }
                    }
                    p = eend;
                    if      (key == "location") location = std::move(val);
                    else if (key == "offset")   { try { offset = std::stoll(val); } catch (...) {} }
                    else if (key == "length")   { try { length = std::stoll(val); } catch (...) {} }
                } else if (tf == 14 && tw == 0) {
                    // data_location: EXTERNAL = 1
                    if (readVarint(data, p, bodyEnd) == 1u) isExt = true;
                } else {
                    skipField(data, p, bodyEnd, tw);
                }
            }

            if (isExt && !name.empty() && !location.empty() && length > 0) {
                std::string norm = name;
                for (char& c : norm) if (c == '.' || c == '/') c = '_';
                result[norm] = { name, shape, dtype, offset, length };
            }
        }
        pos = bodyEnd;
    }

    Logger::info("parseExternalIndex: " + bundle.onnxPath.filename().string()
                 + " — " + std::to_string(result.size()) + " external initializer(s)");
    return result;
}

} // namespace sd
