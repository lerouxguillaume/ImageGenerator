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
    return j;
}

static AssetType assetTypeFromJson(const nlohmann::json& j) {
    AssetType a;
    a.id   = j.value("id",   std::string{});
    a.name = j.value("name", std::string{});
    if (j.contains("promptTokens") && j["promptTokens"].is_object())
        a.promptTokens = j["promptTokens"].get<Prompt>();
    if (j.contains("constraints") && j["constraints"].is_object()) {
        const auto& c          = j["constraints"];
        a.constraints.allowFloorPlane   = c.value("allowFloorPlane",   false);
        a.constraints.allowSceneContext = c.value("allowSceneContext",  false);
        a.constraints.tileableEdge      = c.value("tileableEdge",       false);
        a.constraints.topSurfaceVisible = c.value("topSurfaceVisible",  false);
    }
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
                                       const Prompt&      promptTokens) {
    Project* p = findProject(projectId);
    if (!p) {
        Logger::info("addAssetType: project '" + projectId + "' not found");
        return {};
    }
    AssetType a;
    a.id           = makeAssetTypeId();
    a.name         = name;
    a.promptTokens = promptTokens;
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
