#pragma once
#include <string>
#include <vector>
#include "../prompt/Prompt.hpp"
#include "Project.hpp"

struct AssetTypeTemplate {
    std::string id;
    std::string label;
    std::string defaultName;
    Prompt           promptTokens;
    AssetConstraints constraints;
    AssetSpec        spec;
    AssetExportSpec  exportSpec;
    bool             referenceEnabled = false;
    std::string      referenceImagePath;
    float            structureStrength = 0.45f;
    std::vector<std::string> tags;
};

class AssetTypeTemplates {
public:
    static const std::vector<AssetTypeTemplate>& all();
    static const AssetTypeTemplate* findById(const std::string& id);
};
