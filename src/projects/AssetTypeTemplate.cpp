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
    s.validation.enforceAnchor = false;
    return s;
}

AssetSpec makeWallLeftSpec() {
    AssetSpec s = makeSpec(Orientation::LeftWall, /*tileable=*/true,
                           ShapePolicy::Bounded, 0.80f, 0.75f, 0.85f);
    s.canvasWidth  = 512;
    s.canvasHeight = 768;
    s.expectedBounds = {82, 92, 348, 590};
    s.anchor = {256, 682};
    s.validation.enforceAnchor = true;
    return s;
}

AssetExportSpec makeObjectFitExport(int exportW, int exportH, int maxW, int maxH, int paddingPx = 8) {
    AssetExportSpec s;
    s.exportWidth = exportW;
    s.exportHeight = exportH;
    s.maxObjectWidth = maxW;
    s.maxObjectHeight = maxH;
    s.paddingPx = paddingPx;
    s.fitMode = AssetFitMode::ObjectFit;
    s.requireAlpha = true;
    return s;
}

AssetExportSpec makeTileExport(int exportW, int exportH) {
    AssetExportSpec s;
    s.exportWidth = exportW;
    s.exportHeight = exportH;
    s.maxObjectWidth = exportW;
    s.maxObjectHeight = exportH;
    s.paddingPx = 0;
    s.fitMode = AssetFitMode::TileExact;
    s.requireAlpha = true;
    return s;
}

const std::vector<AssetTypeTemplate> kTemplates = {
    {
        "wall_left", "Wall Left", "Wall Left",
        makePrompt(
            {"isometric view", "3/4 angle", "left wall segment", "flat vertical wall panel",
             "single object", "centered composition", "full object visible", "no perspective distortion",
             "clean silhouette", "isolated wall plane", "modular game asset", "tile-safe edges"},
            {"characters", "background scene", "room interior", "text", "watermark", "ornate", "complex",
             "multiple objects", "top view", "cinematic lighting", "wide shot", "full building",
             "floor scene", "floor plane", "ground plane", "ceiling"}
        ),
        AssetConstraints{false, false, true, false},
        makeWallLeftSpec(),
        makeObjectFitExport(128, 192, 112, 176, 8),
        true,
        "assets/references/wall_left.pgm",
        0.34f,
        {"modular", "tileable", "vertical", "left_wall"}
    },
    {
        "floor_tile", "Floor Tile", "Floor Tile",
        makePrompt(
            {"isometric floor tile", "single modular ground tile", "top-down readable surface", "seamless tile edges", "flat gameplay readability"},
            {"characters", "background scene", "room interior", "text", "watermark", "perspective camera", "floating object"}
        ),
        AssetConstraints{true, false, true, true},
        makeSpec(Orientation::FloorTile, /*tileable=*/true, ShapePolicy::Bounded, 0.85f, 0.6f, 0.95f),
        makeTileExport(128, 128),
        false,
        {},
        0.45f,
        {"floor", "tileable", "ground"}
    },
    {
        "corner_wall", "Corner Wall", "Corner Wall",
        makePrompt(
            {"isometric corner wall", "single modular corner piece", "clean silhouette", "readable wall planes", "game asset framing"},
            {"characters", "background scene", "room interior", "text", "watermark", "top view", "distant shot"}
        ),
        AssetConstraints{false, false, false, true},
        makeSpec(Orientation::Unset, /*tileable=*/false, ShapePolicy::Bounded, 0.65f, 0.4f, 0.85f),
        makeObjectFitExport(160, 192, 144, 176, 8),
        false,
        {},
        0.45f,
        {"corner", "wall", "modular"}
    },
    {
        "door", "Door", "Door",
        makePrompt(
            {"isometric left wall door", "single doorway insert", "clean frame", "modular wall attachment", "game asset"},
            {"characters", "background scene", "room interior", "text", "watermark", "top view", "full building", "staircase"}
        ),
        AssetConstraints{false, false, false, true},
        makeSpec(Orientation::LeftWall, /*tileable=*/false, ShapePolicy::Bounded, 0.62f, 0.45f, 0.8f),
        makeObjectFitExport(128, 192, 112, 176, 8),
        false,
        {},
        0.45f,
        {"entry", "prop"}
    },
    {
        "stairs", "Stairs", "Stairs",
        makePrompt(
            {"isometric stairs", "single modular stair segment", "clean readable steps", "game asset framing", "clear top surfaces"},
            {"characters", "background scene", "room interior", "text", "watermark", "top view", "multiple objects"}
        ),
        AssetConstraints{true, false, false, true},
        makeSpec(Orientation::Unset, /*tileable=*/false, ShapePolicy::Bounded, 0.65f, 0.45f, 0.85f),
        makeObjectFitExport(160, 192, 144, 176, 8),
        false,
        {},
        0.45f,
        {"elevation", "transition"}
    },
    {
        "prop", "Prop", "Prop",
        makePrompt(
            {"isometric prop", "single isolated object", "game asset framing", "clean silhouette", "readable materials"},
            {"characters", "background scene", "room interior", "text", "watermark", "multiple objects", "cropped object", "cinematic shot"}
        ),
        AssetConstraints{false, false, false, false},
        makeSpec(Orientation::Prop, /*tileable=*/false, ShapePolicy::Bounded, 0.58f, 0.38f, 0.78f),
        makeObjectFitExport(128, 128, 112, 112, 8),
        false,
        {},
        0.45f,
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
