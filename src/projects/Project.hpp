#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "../config/AppConfig.hpp"
#include "../prompt/Prompt.hpp"

struct AssetType {
    std::string id;
    std::string name;
    Prompt      promptTokens;
};

struct Project {
    std::string            id;
    std::string            name;
    Prompt                 stylePrompt;
    std::string            modelId;
    std::vector<LoraEntry> loraEntries;
    int                    width     = 512;
    int                    height    = 512;
    std::vector<AssetType> assetTypes;
    uint64_t               createdAt = 0;
};

// Full resolved context passed from ProjectController to ImageGeneratorController.
// All strings are empty when no project is active.
struct ResolvedProjectContext {
    std::string projectId;
    std::string projectName;
    std::string assetTypeId;
    std::string assetTypeName;
    Prompt      stylePrompt;
    Prompt      assetTypeTokens;
    std::string            outputSubpath; // sanitised relative path, e.g. "Medieval Dungeon/Wall Tile"
    std::vector<AssetType> allAssetTypes; // all types in the project, used to populate gallery tabs

    bool empty() const { return projectId.empty(); }
};
