#pragma once
#include <filesystem>
#include <string>
#include <vector>

struct ModelCapabilities {
    bool vaeEncoderAvailable = true; // false → img2img / reference not supported
    bool loraCompatible      = true; // false → LoRA injection not supported
};

struct ImportedModel {
    std::string           id;
    std::string           name;
    std::string           arch;         // "sd15" | "sdxl"
    std::filesystem::path onnxPath;     // absolute path to the ONNX directory
    std::string           importedAt;   // ISO 8601 UTC
    ModelCapabilities     capabilities; // populated from model.json at load time
};

class ImportedModelRegistry {
public:
    explicit ImportedModelRegistry(std::filesystem::path registryPath);

    void add(const ImportedModel& model);
    bool exists(const std::string& id) const;
    const std::vector<ImportedModel>& list() const { return models_; }

private:
    void load();
    void save() const;

    std::filesystem::path    registryPath_;
    std::vector<ImportedModel> models_;
};
