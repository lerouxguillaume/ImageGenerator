#include "SafetensorsInspector.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>

// ── SafetensorsInfo helpers ───────────────────────────────────────────────────

const char* SafetensorsInfo::modelTypeName() const noexcept {
    switch (modelType) {
        case ModelType::Checkpoint: return "Checkpoint";
        case ModelType::LoRA:       return "LoRA";
        case ModelType::VAE:        return "VAE";
        default:                    return "Unknown";
    }
}

const char* SafetensorsInfo::architectureName() const noexcept {
    switch (architecture) {
        case Architecture::SD15: return "SD 1.5";
        case Architecture::SDXL: return "SDXL";
        default:                 return "Unknown";
    }
}

const char* SafetensorsInfo::archArg() const noexcept {
    switch (architecture) {
        case Architecture::SD15: return "sd15";
        case Architecture::SDXL: return "sdxl";
        default:                 return "auto";
    }
}

// ── String-search heuristics ──────────────────────────────────────────────────

static size_t countSubstr(const std::string& hay, const char* needle) {
    size_t count = 0;
    size_t pos   = 0;
    const size_t len = strlen(needle);
    while ((pos = hay.find(needle, pos)) != std::string::npos) {
        ++count;
        pos += len;
    }
    return count;
}

static bool contains(const std::string& hay, const char* needle) {
    return hay.find(needle) != std::string::npos;
}

// ── Inspector ────────────────────────────────────────────────────────────────

SafetensorsInfo SafetensorsInspector::inspect(const std::filesystem::path& path) {
    SafetensorsInfo info;

    std::ifstream f(path, std::ios::binary);
    if (!f) {
        info.error = "Cannot open file: " + path.string();
        return info;
    }

    // Read 8-byte little-endian header size
    uint64_t headerSize = 0;
    f.read(reinterpret_cast<char*>(&headerSize), 8);
    if (!f || f.gcount() < 8) {
        info.error = "File too small to be a safetensors file";
        return info;
    }

    if (headerSize == 0 || headerSize > kMaxHeaderBytes) {
        info.error = "Implausible header size (" + std::to_string(headerSize) + " bytes)";
        return info;
    }

    std::string header(static_cast<size_t>(headerSize), '\0');
    f.read(header.data(), static_cast<std::streamsize>(headerSize));
    if (static_cast<uint64_t>(f.gcount()) < headerSize) {
        info.error = "File truncated before end of header";
        return info;
    }

    // ── Architecture ─────────────────────────────────────────────────────────
    if (contains(header, "conditioner.embedders.1."))
        info.architecture = SafetensorsInfo::Architecture::SDXL;
    else if (contains(header, "model.diffusion_model.") || contains(header, "cond_stage_model."))
        info.architecture = SafetensorsInfo::Architecture::SD15;

    // ── Model type ───────────────────────────────────────────────────────────
    const size_t loraUp   = countSubstr(header, "lora_up.weight");
    const size_t loraDown = countSubstr(header, "lora_down.weight");

    // Count total tensors by "data_offsets" entries
    info.tensorCount = static_cast<int>(countSubstr(header, "\"data_offsets\""));

    if (loraUp + loraDown >= 5) {
        info.modelType = SafetensorsInfo::ModelType::LoRA;
    } else if (info.architecture != SafetensorsInfo::Architecture::Unknown) {
        info.modelType = SafetensorsInfo::ModelType::Checkpoint;
    } else if (contains(header, "\"encoder.") && contains(header, "\"decoder.")) {
        // Standalone VAE: has encoder + decoder blocks but no diffusion model keys
        info.modelType = SafetensorsInfo::ModelType::VAE;
    }

    // ── Dominant dtype ───────────────────────────────────────────────────────
    const size_t nF16  = countSubstr(header, "\"F16\"");
    const size_t nF32  = countSubstr(header, "\"F32\"");
    const size_t nBF16 = countSubstr(header, "\"BF16\"");

    if (nF16 >= nF32 && nF16 >= nBF16)
        info.dtype = "F16";
    else if (nBF16 >= nF32)
        info.dtype = "BF16";
    else
        info.dtype = "F32";

    info.valid = true;
    return info;
}
