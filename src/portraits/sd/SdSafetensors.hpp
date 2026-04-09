#pragma once
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace sd {

struct SafeTensor {
    std::vector<int64_t> shape;
    std::vector<float>   data;   // always float32 (F16 tensors are upcast on load)
};

using SafetensorsMap = std::map<std::string, SafeTensor>;

// ── Float16 ↔ Float32 ────────────────────────────────────────────────────────

static inline float fp16ToFloat(uint16_t h) {
    const uint32_t sign   = (h >> 15) & 1u;
    const uint32_t exp5   = (h >> 10) & 0x1Fu;
    const uint32_t mant10 = h & 0x3FFu;
    uint32_t bits;
    if (exp5 == 0u) {
        if (mant10 == 0u) {
            bits = sign << 31;                                // signed zero
        } else {
            uint32_t e = 0u, m = mant10;                    // subnormal → normalise
            while (!(m & 0x400u)) { m <<= 1; ++e; }
            bits = (sign << 31) | ((127u - 15u - e + 1u) << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (exp5 == 31u) {
        bits = (sign << 31) | (0xFFu << 23) | (mant10 << 13);  // Inf / NaN
    } else {
        bits = (sign << 31) | ((exp5 - 15u + 127u) << 23) | (mant10 << 13);
    }
    float result;
    std::memcpy(&result, &bits, 4);
    return result;
}

static inline uint16_t floatToFp16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    const uint32_t sign   = bits >> 31;
    const int32_t  exp32  = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127;
    const uint32_t mant32 = bits & 0x7FFFFFu;
    if (exp32 > 15)  return static_cast<uint16_t>((sign << 15) | 0x7C00u);  // clamp to Inf
    if (exp32 < -24) return static_cast<uint16_t>(sign << 15);              // underflow → zero
    if (exp32 < -14) {                                                        // subnormal fp16
        // Correct shift: mant10 = (1.mant32) >> (-exp32 - 1)
        // = (mant32 | 0x800000) >> (13 + (-14 - exp32))
        // The original used shift=1+(-14-exp32) which was 12 too small, causing
        // overflow in the lower 16 bits and producing fp16 NaN for ~[6e-8, 6e-5].
        const uint32_t m = (mant32 | 0x800000u) >> (13u + static_cast<uint32_t>(-14 - exp32));
        return static_cast<uint16_t>((sign << 15) | (m & 0x3FFu));
    }
    const uint32_t exp5   = static_cast<uint32_t>(exp32 + 15);
    const uint32_t mant10 = (mant32 + 0x1000u) >> 13;  // round-to-nearest
    return static_cast<uint16_t>((sign << 15) | (exp5 << 10) | (mant10 & 0x3FFu));
}

// ── BF16 → F32 ───────────────────────────────────────────────────────────────
// BF16 shares the same exponent layout as F32; conversion is a zero-extend:
// the 16-bit BF16 value occupies the upper 16 bits of the 32-bit float.

static inline float bf16ToFloat(uint16_t b) {
    const uint32_t bits = static_cast<uint32_t>(b) << 16;
    float result;
    std::memcpy(&result, &bits, 4);
    return result;
}

// ── Safetensors loader ────────────────────────────────────────────────────────
// Loads a .safetensors file and returns all F32/F16/BF16 tensors as float32.
// Integer tensors are silently skipped (not used in LoRA files).
// Throws std::runtime_error on I/O or format errors.

inline SafetensorsMap loadSafetensors(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open safetensors: " + path);

    uint64_t headerLen = 0;
    f.read(reinterpret_cast<char*>(&headerLen), 8);
    if (!f || headerLen == 0 || headerLen > 100'000'000u)
        throw std::runtime_error("Invalid safetensors header in: " + path);

    std::string headerJson(headerLen, '\0');
    f.read(headerJson.data(), static_cast<std::streamsize>(headerLen));
    const size_t dataOffset = 8u + static_cast<size_t>(headerLen);

    const auto meta = nlohmann::json::parse(headerJson);
    SafetensorsMap result;

    for (const auto& [name, info] : meta.items()) {
        if (name == "__metadata__") continue;

        const std::string dtypeStr = info["dtype"].get<std::string>();
        const bool isF32  = (dtypeStr == "F32");
        const bool isF16  = (dtypeStr == "F16");
        const bool isBF16 = (dtypeStr == "BF16");
        if (!isF32 && !isF16 && !isBF16) continue;  // int types — not in LoRA files

        const auto offsets   = info["data_offsets"].get<std::vector<uint64_t>>();
        const size_t begin   = dataOffset + static_cast<size_t>(offsets[0]);
        const size_t byteLen = static_cast<size_t>(offsets[1] - offsets[0]);
        const size_t elemSz  = isF32 ? 4u : 2u;  // F16 and BF16 are both 2 bytes
        const size_t numElems = byteLen / elemSz;

        std::vector<uint8_t> raw(byteLen);
        f.seekg(static_cast<std::streamoff>(begin));
        f.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(byteLen));

        SafeTensor t;
        t.shape = info["shape"].get<std::vector<int64_t>>();
        t.data.resize(numElems);

        if (isF32) {
            std::memcpy(t.data.data(), raw.data(), byteLen);
        } else if (isF16) {
            const auto* src = reinterpret_cast<const uint16_t*>(raw.data());
            for (size_t i = 0; i < numElems; ++i)
                t.data[i] = fp16ToFloat(src[i]);
        } else {  // BF16
            const auto* src = reinterpret_cast<const uint16_t*>(raw.data());
            for (size_t i = 0; i < numElems; ++i)
                t.data[i] = bf16ToFloat(src[i]);
        }
        result[name] = std::move(t);
    }
    return result;
}

} // namespace sd
