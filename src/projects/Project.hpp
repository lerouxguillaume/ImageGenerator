#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include "../config/AppConfig.hpp"
#include "../prompt/Prompt.hpp"

// Project-level generation constraints for isometric asset packs.
struct PackConstraints {
    bool transparentBg        = false;
    bool isometricAngle       = false;
    bool centeredComposition  = false;
    bool subjectFullyVisible  = false;
    bool noEnvironmentClutter = false;
    bool noFloorPlane         = false;
};

// Per-asset-type overrides and additions on top of PackConstraints.
struct AssetConstraints {
    bool allowFloorPlane   = false; // overrides PackConstraints::noFloorPlane
    bool allowSceneContext = false; // overrides PackConstraints::noEnvironmentClutter
    bool tileableEdge      = false;
    bool topSurfaceVisible = false;
};

// Isometric plane / orientation class for an asset slot.
enum class Orientation { Unset, LeftWall, RightWall, FloorTile, Prop, Character };

// How strictly the generated asset shape must conform to a reference.
enum class ShapePolicy { Freeform, Bounded, SilhouetteLocked };

// How the generated image is fit to the target canvas.
enum class AssetFitMode { ObjectFit, TileExact, NoResize };

struct Anchor        { int x = 0; int y = 0; };
struct OccupiedBounds { int x = 0; int y = 0; int w = 0; int h = 0; };

struct AssetExportSpec {
    int exportWidth = 128;
    int exportHeight = 128;
    int maxObjectWidth = 112;
    int maxObjectHeight = 112;
    int paddingPx = 8;
    AssetFitMode fitMode = AssetFitMode::ObjectFit;
    bool requireAlpha = true;
};

struct ValidationPolicy {
    bool  enforceCanvasSize      = true;
    bool  enforceTransparency    = true;
    bool  enforceSilhouette      = false;
    bool  enforceAnchor          = false;
    float maxSilhouetteDeviation = 0.0f;
};

// Formal production contract each asset type must satisfy.
struct AssetSpec {
    int canvasWidth   = 0;   // 0 = inherit from project
    int canvasHeight  = 0;

    Anchor         anchor;
    Orientation    orientation    = Orientation::Unset;
    OccupiedBounds expectedBounds;

    float targetFillRatio = 0.6f;
    float minFillRatio    = 0.3f;
    float maxFillRatio    = 0.9f;

    bool         requiresTransparency = true;
    ShapePolicy  shapePolicy          = ShapePolicy::Freeform;
    AssetFitMode fitMode              = AssetFitMode::ObjectFit;
    bool         isTileable           = false;

    ValidationPolicy validation;
};

struct AssetType {
    std::string      id;
    std::string      name;
    Prompt           promptTokens;
    AssetConstraints constraints;
    AssetSpec        spec;
    AssetExportSpec  exportSpec;
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
    PackConstraints        constraints;
};

// Full resolved context passed from ProjectController to ImageGeneratorController.
// All strings are empty when no project is active.
struct ResolvedProjectContext {
    std::string projectId;
    std::string projectName;
    std::string assetTypeId;
    std::string assetTypeName;
    Prompt      stylePrompt;
    Prompt      constraintTokens; // compiled from PackConstraints + AssetConstraints
    Prompt      assetTypeTokens;
    AssetSpec   spec;             // production contract for the active asset type
    AssetExportSpec exportSpec;   // deterministic post-process export contract
    std::string            outputSubpath; // sanitised relative path, e.g. "Medieval Dungeon/Wall Tile"
    std::vector<AssetType> allAssetTypes; // all types in the project, used to populate gallery tabs

    bool empty() const { return projectId.empty(); }
};
