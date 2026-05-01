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
                           ShapePolicy::Bounded, 0.52f, 0.35f, 0.68f);
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
            {"isometric dungeon wall tile", "tall stone wall slab", "flat vertical wall surface",
             "left-facing wall panel, parallelogram shape", "stone masonry, rough stone blocks",
             "flat architectural surface, no attached objects", "game tileset asset, RPG dungeon tile",
             "seamless tiling edges", "isolated object, transparent background",
             "centered composition", "clean hard edges, clear silhouette"},
            {"characters", "props", "furniture", "background scene", "room interior", "landscape",
             "text", "watermark", "multiple objects", "top view", "wide shot", "full building",
             "floor scene", "floor plane", "ground plane", "ceiling", "doorway scene", "interior scene",
             "mechanical parts", "electronic components", "machinery", "robot", "machine",
             "gadget", "device", "pipes", "cables", "circuit board", "industrial equipment", "wires"}
        ),
        AssetConstraints{false, false, true, false},
        makeWallLeftSpec(),
        makeObjectFitExport(128, 192, 112, 176, 8),
        false,
        {},
        0.34f,
        GenerationWorkflow::CandidateRun,
        CandidateRunSettings{},
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
