#include "ProjectManager.hpp"
#include "../managers/Logger.hpp"
#include "../prompt/PromptJson.hpp"
#include <chrono>
#include <fstream>
#include <nlohmann/json.hpp>

// ── ID / timestamp helpers ────────────────────────────────────────────────────

static std::string makeProjectId() {
    const auto ms = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    return "project_" + std::to_string(ms);
}

static std::string makeAssetTypeId() {
    const auto ms = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    return "assettype_" + std::to_string(ms);
}

static uint64_t nowSeconds() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
}

// ── JSON helpers ──────────────────────────────────────────────────────────────

static const char* orientationToStr(Orientation o) {
    switch (o) {
        case Orientation::LeftWall:    return "left_wall";
        case Orientation::RightWall:   return "right_wall";
        case Orientation::FloorTile:   return "floor_tile";
        case Orientation::Prop:        return "prop";
        case Orientation::Character:   return "character";
        default:                       return "unset";
    }
}
static Orientation strToOrientation(const std::string& s) {
    if (s == "left_wall")  return Orientation::LeftWall;
    if (s == "right_wall") return Orientation::RightWall;
    if (s == "floor_tile") return Orientation::FloorTile;
    if (s == "prop")       return Orientation::Prop;
    if (s == "character")  return Orientation::Character;
    return Orientation::Unset;
}
static const char* shapePolicyToStr(ShapePolicy p) {
    switch (p) {
        case ShapePolicy::Bounded:          return "bounded";
        case ShapePolicy::SilhouetteLocked: return "silhouette_locked";
        default:                            return "freeform";
    }
}
static ShapePolicy strToShapePolicy(const std::string& s) {
    if (s == "bounded")           return ShapePolicy::Bounded;
    if (s == "silhouette_locked") return ShapePolicy::SilhouetteLocked;
    return ShapePolicy::Freeform;
}
static const char* fitModeToStr(AssetFitMode m) {
    switch (m) {
        case AssetFitMode::TileExact: return "tile_exact";
        case AssetFitMode::NoResize:  return "no_resize";
        default:                      return "object_fit";
    }
}
static AssetFitMode strToFitMode(const std::string& s) {
    if (s == "tile_exact") return AssetFitMode::TileExact;
    if (s == "no_resize")  return AssetFitMode::NoResize;
    return AssetFitMode::ObjectFit;
}
static const char* workflowToStr(GenerationWorkflow w) {
    return w == GenerationWorkflow::CandidateRun ? "candidate_run" : "standard";
}
static GenerationWorkflow strToWorkflow(const std::string& s) {
    return s == "candidate_run" ? GenerationWorkflow::CandidateRun : GenerationWorkflow::Standard;
}

static nlohmann::json alphaCutoutToJson(const AlphaCutoutSpec& s) {
    return {
        {"tolerance", s.tolerance},
        {"featherRadius", s.featherRadius},
        {"defringe", s.defringe}
    };
}

static AlphaCutoutSpec alphaCutoutFromJson(const nlohmann::json& j) {
    AlphaCutoutSpec s;
    s.tolerance = j.value("tolerance", 30.f);
    s.featherRadius = j.value("featherRadius", 3);
    s.defringe = j.value("defringe", true);
    return s;
}

static nlohmann::json candidateRunToJson(const CandidateRunSettings& s) {
    return {
        {"minExploreImages", s.minExploreImages},
        {"candidateCount", s.candidateCount},
        {"refineVariants", s.refineVariants},
        {"scoreThreshold", s.scoreThreshold},
        {"explorationStrength", s.explorationStrength},
        {"refinementStrength", s.refinementStrength}
    };
}

static CandidateRunSettings candidateRunFromJson(const nlohmann::json& j) {
    CandidateRunSettings s;
    s.minExploreImages = j.value("minExploreImages", 8);
    s.candidateCount = j.value("candidateCount", 3);
    s.refineVariants = j.value("refineVariants", 2);
    s.scoreThreshold = j.value("scoreThreshold", 150.0f);
    s.explorationStrength = j.value("explorationStrength", 0.70f);
    s.refinementStrength = j.value("refinementStrength", 0.27f);
    return s;
}

static nlohmann::json exportSpecToJson(const AssetExportSpec& s) {
    return {
        {"exportWidth", s.exportWidth},
        {"exportHeight", s.exportHeight},
        {"maxObjectWidth", s.maxObjectWidth},
        {"maxObjectHeight", s.maxObjectHeight},
        {"paddingPx", s.paddingPx},
        {"fitMode", fitModeToStr(s.fitMode)},
        {"requireAlpha", s.requireAlpha},
        {"alphaCutout", alphaCutoutToJson(s.alphaCutout)}
    };
}

static AssetExportSpec exportSpecFromJson(const nlohmann::json& j) {
    AssetExportSpec s;
    s.exportWidth = j.value("exportWidth", 128);
    s.exportHeight = j.value("exportHeight", 128);
    s.maxObjectWidth = j.value("maxObjectWidth", 112);
    s.maxObjectHeight = j.value("maxObjectHeight", 112);
    s.paddingPx = j.value("paddingPx", 8);
    s.fitMode = strToFitMode(j.value("fitMode", std::string{"object_fit"}));
    s.requireAlpha = j.value("requireAlpha", true);
    if (j.contains("alphaCutout") && j["alphaCutout"].is_object())
        s.alphaCutout = alphaCutoutFromJson(j["alphaCutout"]);
    return s;
}

static nlohmann::json specToJson(const AssetSpec& s) {
    return {
        {"canvasWidth",          s.canvasWidth},
        {"canvasHeight",         s.canvasHeight},
        {"anchor",               {{"x", s.anchor.x}, {"y", s.anchor.y}}},
        {"orientation",          orientationToStr(s.orientation)},
        {"expectedBounds",       {{"x", s.expectedBounds.x}, {"y", s.expectedBounds.y},
                                  {"w", s.expectedBounds.w}, {"h", s.expectedBounds.h}}},
        {"targetFillRatio",      s.targetFillRatio},
        {"minFillRatio",         s.minFillRatio},
        {"maxFillRatio",         s.maxFillRatio},
        {"requiresTransparency", s.requiresTransparency},
        {"shapePolicy",          shapePolicyToStr(s.shapePolicy)},
        {"fitMode",              fitModeToStr(s.fitMode)},
        {"isTileable",           s.isTileable},
        {"validation", {
            {"enforceCanvasSize",      s.validation.enforceCanvasSize},
            {"enforceTransparency",    s.validation.enforceTransparency},
            {"enforceSilhouette",      s.validation.enforceSilhouette},
            {"enforceAnchor",          s.validation.enforceAnchor},
            {"maxSilhouetteDeviation", s.validation.maxSilhouetteDeviation}
        }}
    };
}

static AssetSpec specFromJson(const nlohmann::json& j) {
    AssetSpec s;
    s.canvasWidth  = j.value("canvasWidth",  0);
    s.canvasHeight = j.value("canvasHeight", 0);
    if (j.contains("anchor") && j["anchor"].is_object()) {
        s.anchor.x = j["anchor"].value("x", 0);
        s.anchor.y = j["anchor"].value("y", 0);
    }
    s.orientation = strToOrientation(j.value("orientation", std::string{"unset"}));
    if (j.contains("expectedBounds") && j["expectedBounds"].is_object()) {
        const auto& b      = j["expectedBounds"];
        s.expectedBounds.x = b.value("x", 0);
        s.expectedBounds.y = b.value("y", 0);
        s.expectedBounds.w = b.value("w", 0);
        s.expectedBounds.h = b.value("h", 0);
    }
    s.targetFillRatio      = j.value("targetFillRatio",      0.6f);
    s.minFillRatio         = j.value("minFillRatio",         0.3f);
    s.maxFillRatio         = j.value("maxFillRatio",         0.9f);
    s.requiresTransparency = j.value("requiresTransparency", true);
    s.shapePolicy          = strToShapePolicy(j.value("shapePolicy", std::string{"freeform"}));
    s.fitMode              = strToFitMode(j.value("fitMode", std::string{"object_fit"}));
    s.isTileable           = j.value("isTileable", false);
    if (j.contains("validation") && j["validation"].is_object()) {
        const auto& v = j["validation"];
        s.validation.enforceCanvasSize      = v.value("enforceCanvasSize",      true);
        s.validation.enforceTransparency    = v.value("enforceTransparency",    true);
        s.validation.enforceSilhouette      = v.value("enforceSilhouette",      false);
        s.validation.enforceAnchor          = v.value("enforceAnchor",          false);
        s.validation.maxSilhouetteDeviation = v.value("maxSilhouetteDeviation", 0.0f);
    }
    return s;
}

static nlohmann::json assetTypeToJson(const AssetType& a) {
    nlohmann::json j;
    j["id"]           = a.id;
    j["name"]         = a.name;
    j["promptTokens"] = a.promptTokens;
    j["constraints"]  = {
        {"allowFloorPlane",    a.constraints.allowFloorPlane},
        {"allowSceneContext",  a.constraints.allowSceneContext},
        {"tileableEdge",       a.constraints.tileableEdge},
        {"topSurfaceVisible",  a.constraints.topSurfaceVisible}
    };
    j["spec"] = specToJson(a.spec);
    j["exportSpec"] = exportSpecToJson(a.exportSpec);
    j["referenceEnabled"] = a.referenceEnabled;
    j["referenceImagePath"] = a.referenceImagePath;
    j["structureStrength"] = a.structureStrength;
    j["workflow"] = workflowToStr(a.workflow);
    j["candidateRun"] = candidateRunToJson(a.candidateRun);
    return j;
}

static AssetType assetTypeFromJson(const nlohmann::json& j) {
    AssetType a;
    a.id   = j.value("id",   std::string{});
    a.name = j.value("name", std::string{});
    if (j.contains("promptTokens") && j["promptTokens"].is_object())
        a.promptTokens = j["promptTokens"].get<Prompt>();
    if (j.contains("constraints") && j["constraints"].is_object()) {
        const auto& c           = j["constraints"];
        a.constraints.allowFloorPlane   = c.value("allowFloorPlane",   false);
        a.constraints.allowSceneContext = c.value("allowSceneContext",  false);
        a.constraints.tileableEdge      = c.value("tileableEdge",       false);
        a.constraints.topSurfaceVisible = c.value("topSurfaceVisible",  false);
    }
    if (j.contains("spec") && j["spec"].is_object())
        a.spec = specFromJson(j["spec"]);
    if (j.contains("exportSpec") && j["exportSpec"].is_object())
        a.exportSpec = exportSpecFromJson(j["exportSpec"]);
    a.referenceEnabled = j.value("referenceEnabled", false);
    a.referenceImagePath = j.value("referenceImagePath", std::string{});
    a.structureStrength = j.value("structureStrength", 0.45f);
    a.workflow = strToWorkflow(j.value("workflow", std::string{"standard"}));
    if (j.contains("candidateRun") && j["candidateRun"].is_object())
        a.candidateRun = candidateRunFromJson(j["candidateRun"]);
    return a;
}

static nlohmann::json projectToJson(const Project& p) {
    nlohmann::json j;
    j["id"]          = p.id;
    j["name"]        = p.name;
    j["stylePrompt"] = p.stylePrompt;
    j["modelId"]     = p.modelId;
    j["width"]       = p.width;
    j["height"]      = p.height;
    j["createdAt"]   = p.createdAt;
    j["constraints"] = {
        {"transparentBg",        p.constraints.transparentBg},
        {"isometricAngle",       p.constraints.isometricAngle},
        {"centeredComposition",  p.constraints.centeredComposition},
        {"subjectFullyVisible",  p.constraints.subjectFullyVisible},
        {"noEnvironmentClutter", p.constraints.noEnvironmentClutter},
        {"noFloorPlane",         p.constraints.noFloorPlane}
    };

    nlohmann::json loras = nlohmann::json::array();
    for (const auto& lo : p.loraEntries)
        loras.push_back({{"path", lo.path}, {"scale", lo.scale}});
    j["loraEntries"] = loras;

    nlohmann::json types = nlohmann::json::array();
    for (const auto& at : p.assetTypes)
        types.push_back(assetTypeToJson(at));
    j["assetTypes"] = types;

    return j;
}

static Project projectFromJson(const nlohmann::json& j) {
    Project p;
    p.id        = j.value("id",      std::string{});
    p.name      = j.value("name",    std::string{});
    p.modelId   = j.value("modelId", std::string{});
    p.width     = j.value("width",   512);
    p.height    = j.value("height",  512);
    p.createdAt = j.value("createdAt", uint64_t{0});

    if (j.contains("stylePrompt") && j["stylePrompt"].is_object())
        p.stylePrompt = j["stylePrompt"].get<Prompt>();

    if (j.contains("constraints") && j["constraints"].is_object()) {
        const auto& c              = j["constraints"];
        p.constraints.transparentBg        = c.value("transparentBg",        false);
        p.constraints.isometricAngle       = c.value("isometricAngle",        false);
        p.constraints.centeredComposition  = c.value("centeredComposition",   false);
        p.constraints.subjectFullyVisible  = c.value("subjectFullyVisible",   false);
        p.constraints.noEnvironmentClutter = c.value("noEnvironmentClutter",  false);
        p.constraints.noFloorPlane         = c.value("noFloorPlane",          false);
    }

    if (j.contains("loraEntries") && j["loraEntries"].is_array()) {
        for (const auto& lo : j["loraEntries"]) {
            LoraEntry entry;
            entry.path  = lo.value("path",  std::string{});
            entry.scale = lo.value("scale", 1.0f);
            if (!entry.path.empty())
                p.loraEntries.push_back(entry);
        }
    }

    if (j.contains("assetTypes") && j["assetTypes"].is_array()) {
        for (const auto& at : j["assetTypes"])
            p.assetTypes.push_back(assetTypeFromJson(at));
    }

    return p;
}

// ── ProjectManager ────────────────────────────────────────────────────────────

ProjectManager::ProjectManager(std::string filePath)
    : filePath_(std::move(filePath))
{
    load();
}

void ProjectManager::load() {
    try {
        std::ifstream f(filePath_);
        if (!f.is_open()) {
            Logger::info("projects.json not found — starting with empty project list");
            return;
        }
        const auto j = nlohmann::json::parse(f);
        if (!j.is_array()) {
            Logger::error("projects.json is not a JSON array — starting with empty project list");
            return;
        }
        for (const auto& item : j)
            projects_.push_back(projectFromJson(item));
        Logger::info("Loaded " + std::to_string(projects_.size()) + " project(s) from " + filePath_);
    } catch (const std::exception& e) {
        Logger::error(std::string("projects.json parse error, starting empty: ") + e.what());
        projects_.clear();
    }
}

void ProjectManager::save() const {
    try {
        nlohmann::json arr = nlohmann::json::array();
        for (const auto& p : projects_)
            arr.push_back(projectToJson(p));
        std::ofstream f(filePath_);
        f << arr.dump(4);
        Logger::info("Projects saved to " + filePath_);
    } catch (const std::exception& e) {
        Logger::error(std::string("Project save error: ") + e.what());
    }
}

Project* ProjectManager::findProject(const std::string& id) {
    for (auto& p : projects_)
        if (p.id == id) return &p;
    return nullptr;
}

Project ProjectManager::createProject(const std::string& name) {
    Project p;
    p.id        = makeProjectId();
    p.name      = name;
    p.createdAt = nowSeconds();
    projects_.push_back(p);
    save();
    return p;
}

void ProjectManager::updateProject(const Project& project) {
    Project* p = findProject(project.id);
    if (!p) {
        Logger::info("updateProject: project '" + project.id + "' not found — no-op");
        return;
    }
    *p = project;
    save();
}

void ProjectManager::deleteProject(const std::string& id) {
    const auto it = std::find_if(projects_.begin(), projects_.end(),
                                 [&](const Project& p) { return p.id == id; });
    if (it == projects_.end()) return;
    projects_.erase(it);
    save();
}

AssetType ProjectManager::addAssetType(const std::string& projectId,
                                       const std::string& name,
                                       const Prompt&      promptTokens,
                                       const AssetConstraints& constraints,
                                       const AssetSpec& spec,
                                       const AssetExportSpec& exportSpec,
                                       bool referenceEnabled,
                                       const std::string& referenceImagePath,
                                       float structureStrength,
                                       GenerationWorkflow workflow,
                                       const CandidateRunSettings& candidateRun) {
    Project* p = findProject(projectId);
    if (!p) {
        Logger::info("addAssetType: project '" + projectId + "' not found");
        return {};
    }
    AssetType a;
    a.id           = makeAssetTypeId();
    a.name         = name;
    a.promptTokens = promptTokens;
    a.constraints  = constraints;
    a.spec         = spec;
    a.exportSpec   = exportSpec;
    a.referenceEnabled = referenceEnabled;
    a.referenceImagePath = referenceImagePath;
    a.structureStrength = structureStrength;
    a.workflow = workflow;
    a.candidateRun = candidateRun;
    p->assetTypes.push_back(a);
    save();
    return a;
}

void ProjectManager::updateAssetType(const std::string& projectId, const AssetType& assetType) {
    Project* p = findProject(projectId);
    if (!p) {
        Logger::info("updateAssetType: project '" + projectId + "' not found — no-op");
        return;
    }
    for (auto& a : p->assetTypes) {
        if (a.id == assetType.id) {
            a = assetType;
            save();
            return;
        }
    }
    Logger::info("updateAssetType: assetType '" + assetType.id + "' not found in project '"
                 + projectId + "' — no-op");
}

void ProjectManager::deleteAssetType(const std::string& projectId,
                                     const std::string& assetTypeId) {
    Project* p = findProject(projectId);
    if (!p) return;
    const auto it = std::find_if(p->assetTypes.begin(), p->assetTypes.end(),
                                 [&](const AssetType& a) { return a.id == assetTypeId; });
    if (it == p->assetTypes.end()) return;
    p->assetTypes.erase(it);
    save();
}

std::optional<Project> ProjectManager::getProject(const std::string& id) const {
    for (const auto& p : projects_)
        if (p.id == id) return p;
    return std::nullopt;
}

const std::vector<Project>& ProjectManager::getAllProjects() const {
    return projects_;
}
