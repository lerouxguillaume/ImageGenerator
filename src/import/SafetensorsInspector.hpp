#pragma once
#include <filesystem>
#include <string>

struct SafetensorsInfo {
    enum class ModelType    { Checkpoint, LoRA, VAE, Unknown };
    enum class Architecture { SD15, SDXL, Unknown };

    ModelType    modelType    = ModelType::Unknown;
    Architecture architecture = Architecture::Unknown;
    std::string  dtype;        // dominant dtype: "F16", "F32", "BF16", …
    int          tensorCount  = 0;
    bool         valid        = false;
    std::string  error;

    // Convenience
    const char* modelTypeName()    const noexcept;
    const char* architectureName() const noexcept;
    const char* archArg()          const noexcept; // "sd15" | "sdxl" | "auto"
};

class SafetensorsInspector {
public:
    // Reads only the JSON header of the file (no tensor data loaded).
    // Always returns a result; check info.valid to distinguish success/failure.
    static SafetensorsInfo inspect(const std::filesystem::path& path);

private:
    static constexpr uint64_t kMaxHeaderBytes = 256u * 1024u * 1024u;
};
