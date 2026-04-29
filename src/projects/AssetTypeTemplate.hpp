#pragma once
#include <string>
#include <vector>
#include "../prompt/Prompt.hpp"
#include "Project.hpp"

struct AssetTypeTemplate {
    std::string id;
    std::string label;
    std::string defaultName;
    Prompt      promptTokens;
    AssetConstraints constraints;
    std::vector<std::string> tags;
};

class AssetTypeTemplates {
public:
    static const std::vector<AssetTypeTemplate>& all();
    static const AssetTypeTemplate* findById(const std::string& id);
};
