#pragma once
#include <optional>
#include <string>
#include <vector>
#include "Preset.hpp"

class ImageGeneratorView;

class PresetManager {
public:
    // Loads presets from filePath on construction; starts empty if file is absent or corrupt.
    explicit PresetManager(std::string filePath = "presets.json");

    // Create a new preset from a Generation snapshot, append and persist it.
    Preset createFromGeneration(const Generation& gen, const std::string& name);

    // Overwrite all fields of an existing preset with gen's data; persist.
    // No-op (with a log warning) if presetId is not found.
    void updateFromGeneration(const std::string& presetId, const Generation& gen);

    // Deep-copy a preset under a new name with a fresh id and createdAt.
    // Logs a warning and returns a default-constructed Preset if presetId is not found.
    Preset duplicatePreset(const std::string& presetId, const std::string& newName);

    std::optional<Preset>      getPreset(const std::string& id) const;
    const std::vector<Preset>& getAllPresets() const;

private:
    void load();
    void save() const;

    std::string         filePath_;
    std::vector<Preset> presets_;
};

// Replaces ALL relevant view fields with preset values (no merge).
// Finds modelId in view.availableModels and sets selectedModelIdx accordingly.
// Logs a warning if modelId is not found; selectedModelIdx is left unchanged.
void applyPresetToSettings(const Preset& preset, ImageGeneratorView& view);
