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
        false,
        {},
        0.34f,
        GenerationWorkflow::PhasedRefinement,
        {"modular", "tileable", "vertical", "left_wall"}
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
