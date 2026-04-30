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

AssetSpec makeSpec(Orientation orient,
                   bool tileable        = false,
                   ShapePolicy shape    = ShapePolicy::Freeform,
                   float targetFill     = 0.6f,
                   float minFill        = 0.3f,
                   float maxFill        = 0.9f) {
    AssetSpec s;
    s.orientation         = orient;
    s.isTileable          = tileable;
    s.shapePolicy         = shape;
    s.requiresTransparency = true;
    s.targetFillRatio     = targetFill;
    s.minFillRatio        = minFill;
    s.maxFillRatio        = maxFill;
    return s;
}

const std::vector<AssetTypeTemplate> kTemplates = {
    {
        "wall", "Wall", "Wall",
        makePrompt(
            {"isometric wall segment", "flat wall panel", "single architectural surface", "clean edges", "minimal detail"},
            {"characters", "background scene", "text", "watermark", "ornate", "complex", "multiple objects", "top view"}
        ),
        AssetConstraints{false, false, true, false},
        makeSpec(Orientation::LeftWall, /*tileable=*/true, ShapePolicy::Bounded, 0.7f, 0.5f, 0.9f),
        {"modular", "tileable", "vertical"}
    },
    {
        "floor_tile", "Floor Tile", "Floor Tile",
        makePrompt(
            {"isometric floor tile", "top-down readable surface", "modular tile", "clean edges"},
            {"characters", "background scene", "text", "watermark"}
        ),
        AssetConstraints{true, false, true, true},
        makeSpec(Orientation::FloorTile, /*tileable=*/true, ShapePolicy::Bounded, 0.85f, 0.6f, 0.95f),
        {"floor", "tileable", "ground"}
    },
    {
        "corner_wall", "Corner Wall", "Corner Wall",
        makePrompt(
            {"isometric corner wall", "modular corner piece", "clean silhouette", "readable stone forms"},
            {"characters", "background scene", "text", "watermark"}
        ),
        AssetConstraints{false, false, false, true},
        makeSpec(Orientation::Unset, /*tileable=*/false, ShapePolicy::Bounded, 0.65f, 0.4f, 0.85f),
        {"corner", "wall", "modular"}
    },
    {
        "door", "Door", "Door",
        makePrompt(
            {"isometric door asset", "front-facing door piece", "game-ready prop", "clean frame"},
            {"characters", "background scene", "text", "watermark"}
        ),
        AssetConstraints{false, false, false, true},
        makeSpec(Orientation::LeftWall, /*tileable=*/false, ShapePolicy::Freeform, 0.6f, 0.4f, 0.8f),
        {"entry", "prop"}
    },
    {
        "stairs", "Stairs", "Stairs",
        makePrompt(
            {"isometric stairs", "modular stair segment", "clean readable steps", "game asset"},
            {"characters", "background scene", "text", "watermark"}
        ),
        AssetConstraints{true, false, false, true},
        makeSpec(Orientation::Unset, /*tileable=*/false, ShapePolicy::Bounded, 0.65f, 0.45f, 0.85f),
        {"elevation", "transition"}
    },
    {
        "prop", "Prop", "Prop",
        makePrompt(
            {"isometric prop", "single object game asset", "centered composition", "clean silhouette"},
            {"characters", "background scene", "text", "watermark"}
        ),
        AssetConstraints{false, false, false, false},
        makeSpec(Orientation::Prop, /*tileable=*/false, ShapePolicy::Freeform, 0.55f, 0.3f, 0.8f),
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
