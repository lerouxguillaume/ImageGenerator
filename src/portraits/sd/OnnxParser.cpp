// OnnxParser.cpp: ONNX protobuf binary reader.
//
// Public interface (declared in SdOnnxPatcher.hpp):
//   resolveBundle()     — filesystem-only bundle detection
//   materializeBundle() — external → inline ONNX byte buffer
//   parseTensorIndex()  — index of inline initializer positions for LoRA patching
//
// The ONNX C++ protobuf library (onnx.pb.h / libonnx) is NOT linked in this
// project: the ORT pre-built package does not ship it.  The binary parsing
// below is therefore the sole implementation path; it is fully contained in
// this translation unit and does not leak the wire-format details to callers.
#include "SdOnnxPatcher.hpp"
#include "../../managers/Logger.hpp"
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>

namespace sd {

// ── Bundle resolution ─────────────────────────────────────────────────────────

OnnxModelBundle resolveBundle(const std::filesystem::path& onnxPath) {
    if (!std::filesystem::exists(onnxPath))
        throw std::runtime_error("resolveBundle: model file not found: " + onnxPath.string());

    OnnxModelBundle bundle;
    bundle.onnxPath = onnxPath;

    // Companion data file convention: <model>.onnx.data
    const auto dataPath = std::filesystem::path(onnxPath.string() + ".data");
    if (std::filesystem::exists(dataPath))
        bundle.dataPath = dataPath;

    Logger::info("  [bundle] " + onnxPath.filename().string()
                 + (bundle.hasExternalData() ? " + .data (external)" : " (inline)"));
    return bundle;
}

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

// ── materializeBundle — write helpers ────────────────────────────────────────
// The parser above only reads.  These helpers write protobuf fields into the
// new inline TensorProto entries that replace the external-data references.
//
// Wire types used when writing:
//   0 = varint   (field << 3 | 0)
//   2 = LEN      (field << 3 | 2)  — strings, bytes, sub-messages

static void appendVarint(std::vector<uint8_t>& out, uint64_t v) {
    while (v >= 0x80u) {
        out.push_back(static_cast<uint8_t>((v & 0x7Fu) | 0x80u));
        v >>= 7;
    }
    out.push_back(static_cast<uint8_t>(v));
}

static void appendVarintField(std::vector<uint8_t>& out, uint32_t fieldNum, uint64_t value) {
    appendVarint(out, (static_cast<uint64_t>(fieldNum) << 3) | 0u);
    appendVarint(out, value);
}

static void appendLenField(std::vector<uint8_t>& out, uint32_t fieldNum,
                            const uint8_t* body, size_t len) {
    appendVarint(out, (static_cast<uint64_t>(fieldNum) << 3) | 2u);
    appendVarint(out, static_cast<uint64_t>(len));
    out.insert(out.end(), body, body + len);
}

// ── External data reference ───────────────────────────────────────────────────

struct ExtRef {
    std::string          name;
    std::vector<int64_t> dims;
    int32_t              dtype      = 0;
    std::string          location;   // relative path from ONNX file directory
    int64_t              dataOffset = 0;
    int64_t              dataLength = -1;
};

// Parse one TensorProto sub-message to extract its external data reference.
// Returns true iff this tensor has data_location == EXTERNAL (2) and a valid
// location + positive length — i.e. it needs to be inlined.
static bool parseExtRef(const uint8_t* data, size_t p, size_t end, ExtRef& ref) {
    ref = {};
    bool hasExtData = false;

    while (p < end) {
        const uint64_t tag = readVarint(data, p, end);
        const int fn = static_cast<int>(tag >> 3);
        const int wt = static_cast<int>(tag & 7u);

        if (fn == 1 && wt == 0) {
            // TensorProto.dims (repeated varint)
            ref.dims.push_back(static_cast<int64_t>(readVarint(data, p, end)));
        } else if (fn == 1 && wt == 2) {
            // packed dims
            const uint64_t plen = readVarint(data, p, end);
            const size_t pend = p + static_cast<size_t>(plen);
            while (p < pend)
                ref.dims.push_back(static_cast<int64_t>(readVarint(data, p, pend)));
        } else if (fn == 2 && wt == 0) {
            // TensorProto.data_type
            ref.dtype = static_cast<int32_t>(readVarint(data, p, end));
        } else if (fn == 4 && wt == 0) {
            // TensorProto.data_location  (DEFAULT=0, EXTERNAL=2)
            if (readVarint(data, p, end) == 2u) hasExtData = true;
        } else if (fn == 8 && wt == 2) {
            // TensorProto.name
            const uint64_t nlen = readVarint(data, p, end);
            ref.name = std::string(reinterpret_cast<const char*>(data + p), nlen);
            p += static_cast<size_t>(nlen);
        } else if (fn == 13 && wt == 2) {
            // TensorProto.external_data (repeated StringStringEntryProto)
            // Each occurrence carries one key-value pair.
            const uint64_t elen = readVarint(data, p, end);
            const size_t eend  = p + static_cast<size_t>(elen);
            std::string key, val;
            while (p < eend) {
                const uint64_t etag = readVarint(data, p, eend);
                const int ef = static_cast<int>(etag >> 3);
                const int ew = static_cast<int>(etag & 7u);
                if (ew == 2) {
                    const uint64_t slen = readVarint(data, p, eend);
                    std::string s(reinterpret_cast<const char*>(data + p), slen);
                    p += static_cast<size_t>(slen);
                    if      (ef == 1) key = std::move(s);
                    else if (ef == 2) val = std::move(s);
                } else {
                    skipField(data, p, eend, ew);
                }
            }
            p = eend;
            if      (key == "location") ref.location   = std::move(val);
            else if (key == "offset")   { try { ref.dataOffset = std::stoll(val); } catch (...) {} }
            else if (key == "length")   { try { ref.dataLength = std::stoll(val); } catch (...) {} }
        } else {
            skipField(data, p, end, wt);
        }
    }

    return hasExtData && !ref.location.empty() && ref.dataLength > 0;
}

// ── Build an inline TensorProto ───────────────────────────────────────────────
//
// Produces the protobuf body (no enclosing tag/len) for a TensorProto with
// raw_data embedded.  Field layout:
//   field 1 (varint, repeated) — each dim
//   field 2 (varint)           — data_type
//   field 8 (LEN)              — name
//   field 9 (LEN)              — raw_data
//
// Deliberately omits field 4 (data_location) and field 13 (external_data).

static std::vector<uint8_t> buildInlineTensorProto(const ExtRef& ref,
                                                    const uint8_t* rawData,
                                                    size_t rawLen) {
    std::vector<uint8_t> tp;
    tp.reserve(ref.dims.size() * 3 + 16 + ref.name.size() + rawLen);
    for (int64_t d : ref.dims)
        appendVarintField(tp, 1, static_cast<uint64_t>(d));
    appendVarintField(tp, 2, static_cast<uint64_t>(ref.dtype));
    appendLenField(tp, 8,
                   reinterpret_cast<const uint8_t*>(ref.name.data()), ref.name.size());
    appendLenField(tp, 9, rawData, rawLen);
    return tp;
}

// ── materializeBundle ─────────────────────────────────────────────────────────

std::vector<uint8_t> materializeBundle(const OnnxModelBundle& bundle) {
    const std::string name = bundle.onnxPath.filename().string();

    // Read the .onnx file (graph structure; tensor bytes are in .data for external models).
    const auto onnxBytes = [&] {
        const std::string p = bundle.onnxPath.string();
        std::ifstream f(p, std::ios::binary | std::ios::ate);
        if (!f) throw std::runtime_error("materializeBundle: cannot open " + p);
        const auto sz = static_cast<size_t>(f.tellg());
        f.seekg(0);
        std::vector<uint8_t> buf(sz);
        f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(sz));
        return buf;
    }();

    // Always scan for external-data references, regardless of bundle.hasExternalData().
    // bundle.hasExternalData() relies on a filesystem probe for <name>.onnx.data, but
    // the actual reference inside the binary uses the filename from the protobuf
    // external_data.location field, which may not match the probed path (different
    // name, CWD mismatch, legacy per-tensor sidecar layout, etc.).
    // Cost for a truly-inline model: one O(n) scan, then extSeen==0 → returns original bytes.
    const uint8_t* data  = onnxBytes.data();
    const size_t   total = onnxBytes.size();

    // ── Locate GraphProto (ModelProto field 7) ────────────────────────────────
    size_t graphTagStart  = SIZE_MAX;
    size_t graphBodyStart = SIZE_MAX;
    size_t graphBodyEnd   = SIZE_MAX;
    {
        size_t pos = 0;
        while (pos < total) {
            const size_t tagPos = pos;
            const uint64_t tag  = readVarint(data, pos, total);
            const int fn = static_cast<int>(tag >> 3);
            const int wt = static_cast<int>(tag & 7u);
            if (wt == 2) {
                const uint64_t blen = readVarint(data, pos, total);
                const size_t bodyEnd = pos + static_cast<size_t>(blen);
                if (fn == 7) {
                    graphTagStart  = tagPos;
                    graphBodyStart = pos;
                    graphBodyEnd   = bodyEnd;
                    break;
                }
                pos = bodyEnd;
            } else {
                skipField(data, pos, total, wt);
            }
        }
    }
    if (graphTagStart == SIZE_MAX)
        throw std::runtime_error(
            "materializeBundle: no GraphProto (field 7) in " + name);

    // ── Open the .onnx.data companion file (lazily, on first external tensor) ─
    // We keep a single ifstream; all our export scripts put every tensor in one
    // data file, so the location string should be constant across all tensors.
    std::string   openedDataPath;
    std::ifstream dataFile;

    auto readTensorBytes = [&](const ExtRef& ref) -> std::vector<uint8_t> {
        // Resolve ref.location relative to the model directory.
        // ref.location is stored as a plain filename in the ONNX binary (e.g.
        // "text_encoder.onnx.data"); prepend the model directory to get the full path.
        const std::string fpath =
            (bundle.onnxPath.parent_path() / ref.location).string();
        if (openedDataPath != fpath) {
            dataFile.close();
            dataFile.open(fpath, std::ios::binary);
            if (!dataFile)
                throw std::runtime_error(
                    "materializeBundle: cannot open data file\n"
                    "  model   : " + bundle.onnxPath.string() + "\n"
                    "  location: " + ref.location + "\n"
                    "  resolved: " + fpath);
            Logger::info("materializeBundle: opened data file: " + fpath);
            openedDataPath = fpath;
        }
        dataFile.seekg(static_cast<std::streamoff>(ref.dataOffset));
        const size_t len = static_cast<size_t>(ref.dataLength);
        std::vector<uint8_t> buf(len);
        dataFile.read(reinterpret_cast<char*>(buf.data()),
                      static_cast<std::streamsize>(len));
        if (!dataFile)
            throw std::runtime_error(
                "materializeBundle: short read from " + fpath +
                " (offset=" + std::to_string(ref.dataOffset) +
                ", length=" + std::to_string(len) + ")");
        return buf;
    };

    // ── Scan GraphProto, building a new body with external tensors inlined ────
    // Reserve a lower bound (graph structure size); will grow as tensor data is appended.
    std::vector<uint8_t> newGraphBody;
    newGraphBody.reserve(graphBodyEnd - graphBodyStart);

    size_t pos     = graphBodyStart;
    int    extSeen = 0;
    ExtRef extRef;

    while (pos < graphBodyEnd) {
        const size_t fieldTagPos = pos;
        const uint64_t gtag     = readVarint(data, pos, graphBodyEnd);
        const int gfn = static_cast<int>(gtag >> 3);
        const int gwt = static_cast<int>(gtag & 7u);

        if (gwt != 2) {
            // Varint / fixed-width field in GraphProto (defensive; shouldn't appear).
            skipField(data, pos, graphBodyEnd, gwt);
            newGraphBody.insert(newGraphBody.end(),
                                data + fieldTagPos, data + pos);
            continue;
        }

        const uint64_t blockLen = readVarint(data, pos, graphBodyEnd);
        const size_t   bodyStart = pos;
        const size_t   bodyEnd   = pos + static_cast<size_t>(blockLen);

        if ((gfn == 5 || gfn == 6) &&
            parseExtRef(data, bodyStart, bodyEnd, extRef))
        {
            // External tensor — read its bytes and rebuild as an inline TensorProto.
            ++extSeen;
            const std::vector<uint8_t> rawData = readTensorBytes(extRef);
            const std::vector<uint8_t> inlineTp =
                buildInlineTensorProto(extRef, rawData.data(), rawData.size());
            // Always write as field 5 (the canonical initializer field for
            // PyTorch-exported ONNX; parseTensorIndex and ORT both accept it).
            appendLenField(newGraphBody, 5, inlineTp.data(), inlineTp.size());
        } else {
            // Inline tensor or any other GraphProto field — copy verbatim.
            newGraphBody.insert(newGraphBody.end(),
                                data + fieldTagPos, data + bodyEnd);
        }

        pos = bodyEnd;
    }

    if (extSeen == 0) {
        // The model file was already fully inline despite having a .data companion.
        Logger::info("materializeBundle: " + name
                     + " — no external tensors found in graph; returning original bytes");
        return onnxBytes;
    }

    Logger::info("materializeBundle: " + name
                 + " — inlined " + std::to_string(extSeen) + " tensor(s)"
                 + "  graph_body=" + std::to_string(newGraphBody.size() >> 20) + " MB");

    // ── Reassemble ModelProto ─────────────────────────────────────────────────
    // Layout: [ModelProto prefix] [field7 tag+len] [new GraphProto body] [suffix]
    // The suffix (ModelProto fields after field 7) is typically empty for ONNX.
    std::vector<uint8_t> result;
    result.reserve(graphTagStart                    // prefix
                   + 16                             // new field7 tag + length varint (max 10 bytes each)
                   + newGraphBody.size()            // body
                   + (total - graphBodyEnd));       // suffix (usually 0)

    result.insert(result.end(), data, data + graphTagStart);
    appendVarint(result, (7u << 3) | 2u);                                  // field 7, wire 2
    appendVarint(result, static_cast<uint64_t>(newGraphBody.size()));
    result.insert(result.end(), newGraphBody.begin(), newGraphBody.end());
    result.insert(result.end(), data + graphBodyEnd, data + total);

    Logger::info("materializeBundle: " + name
                 + " total inline size = " + std::to_string(result.size() >> 20) + " MB");
    return result;
}

// ── parseExternalIndex ────────────────────────────────────────────────────────
// Parses the .onnx file (graph structure only) and returns one ExternalTensorMeta
// entry per initializer whose data_location == EXTERNAL.
//
// Relevant TensorProto fields (ONNX protobuf spec):
//   field 1  → dims        (repeated varint / packed)
//   field 2  → data_type   (varint)
//   field 8  → name        (string)
//   field 13 → external_data (repeated StringStringEntryProto: location/offset/length)
//   field 14 → data_location (varint: DEFAULT=0, EXTERNAL=1)
//
// NOTE: field 14 / value 1 is the correct encoding.  (Legacy code that checked
// field 4 / value 2 was wrong — those correspond to float_data, not data_location.)

OnnxExternalIndex parseExternalIndex(const OnnxModelBundle& bundle) {
    // Read the .onnx file (graph structure; data is in .onnx.data).
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

    // ── Scan GraphProto for initializer fields (5 and 6) ────────────────────
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
