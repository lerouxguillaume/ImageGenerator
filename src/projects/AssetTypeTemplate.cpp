#include "AssetTypeTemplate.hpp"

namespace {
Prompt makePrompt(std::initializer_list<const char*> positive,
                  std::initializer_list<const char*> negative = {}) {
    Prompt p;
    for (const char* token : positive)
        p.positive.push_back({token, 1.0f});
    for (const char* token : negative)
        p.negative.push_back({token, 1.0f});
    return p;
}

const std::vector<AssetTypeTemplate> kTemplates = {
    {
        "wall",
        "Wall",
        "Wall",
        makePrompt(
            {"isometric wall segment", "flat wall panel", "single architectural surface", "clean edges", "minimal detail"},
            {"characters", "background scene", "text", "watermark", "ornate", "complex", "multiple objects", "top view"}
        ),
        AssetConstraints{false, false, true, false},
        {"modular", "tileable", "vertical"}
    },
    {
        "floor_tile",
        "Floor Tile",
        "Floor Tile",
        makePrompt(
            {"isometric floor tile", "top-down readable surface", "modular tile", "clean edges"},
            {"characters", "background scene", "text", "watermark"}
        ),
        AssetConstraints{true, false, true, true},
        {"floor", "tileable", "ground"}
    },
    {
        "corner_wall",
        "Corner Wall",
        "Corner Wall",
        makePrompt(
            {"isometric corner wall", "modular corner piece", "clean silhouette", "readable stone forms"},
            {"characters", "background scene", "text", "watermark"}
        ),
        AssetConstraints{false, false, false, true},
        {"corner", "wall", "modular"}
    },
    {
        "door",
        "Door",
        "Door",
        makePrompt(
            {"isometric door asset", "front-facing door piece", "game-ready prop", "clean frame"},
            {"characters", "background scene", "text", "watermark"}
        ),
        AssetConstraints{false, false, false, true},
        {"entry", "prop"}
    },
    {
        "stairs",
        "Stairs",
        "Stairs",
        makePrompt(
            {"isometric stairs", "modular stair segment", "clean readable steps", "game asset"},
            {"characters", "background scene", "text", "watermark"}
        ),
        AssetConstraints{true, false, false, true},
        {"elevation", "transition"}
    },
    {
        "prop",
        "Prop",
        "Prop",
        makePrompt(
            {"isometric prop", "single object game asset", "centered composition", "clean silhouette"},
            {"characters", "background scene", "text", "watermark"}
        ),
        AssetConstraints{false, false, false, false},
        {"object", "generic"}
    }
};
}

const std::vector<AssetTypeTemplate>& AssetTypeTemplates::all() {
    return kTemplates;
}

const AssetTypeTemplate* AssetTypeTemplates::findById(const std::string& id) {
    for (const auto& tmpl : kTemplates) {
        if (tmpl.id == id)
            return &tmpl;
    }
    return nullptr;
}
