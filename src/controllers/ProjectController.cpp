#include "ProjectController.hpp"
#include "../managers/Logger.hpp"

static std::string sanitiseName(const std::string& s) {
    std::string r;
    r.reserve(s.size());
    for (char c : s) {
        if (c == '/' || c == '\\' || c == ':' || c == '*' ||
            c == '?' || c == '"' || c == '<' || c == '>' || c == '|')
            r += '_';
        else
            r += c;
    }
    return r;
}

ProjectController::ProjectController(AppConfig& cfg)
    : config_(cfg)
{}

void ProjectController::update(ProjectView& view) {
    view.projects = projectManager_.getAllProjects();
}

void ProjectController::handleEvent(const sf::Event& e, sf::RenderWindow& win,
                                    ProjectView& view, AppScreen& appScreen) {
    if (e.type == sf::Event::Closed) {
        win.close();
        return;
    }
    if (e.type == sf::Event::KeyPressed) {
        if (e.key.code == sf::Keyboard::Escape) {
            if (view.newProjectInputActive) {
                view.newProjectInputActive = false;
                view.newProjectName.clear();
                view.newProjectCursor = 0;
            } else if (view.newAssetTypeInputActive) {
                view.newAssetTypeInputActive = false;
                view.newAssetTypeName.clear();
                view.newAssetTypeCursor = 0;
            } else {
                appScreen = AppScreen::MENU;
            }
            return;
        }
        if (e.key.code == sf::Keyboard::Return) {
            if (view.newProjectInputActive)
                commitNewProject(view);
            else if (view.newAssetTypeInputActive)
                commitNewAssetType(view);
            return;
        }
        if (e.key.code == sf::Keyboard::BackSpace) {
            if (view.newProjectInputActive && !view.newProjectName.empty()) {
                view.newProjectName.pop_back();
                view.newProjectCursor = static_cast<int>(view.newProjectName.size());
            } else if (view.newAssetTypeInputActive && !view.newAssetTypeName.empty()) {
                view.newAssetTypeName.pop_back();
                view.newAssetTypeCursor = static_cast<int>(view.newAssetTypeName.size());
            }
            return;
        }
    }
    if (e.type == sf::Event::TextEntered)
        handleTextInput(e.text.unicode, view);

    if (e.type == sf::Event::MouseButtonPressed && e.mouseButton.button == sf::Mouse::Left)
        handleClick(win.mapPixelToCoords({e.mouseButton.x, e.mouseButton.y}), view, appScreen);
}

void ProjectController::handleClick(sf::Vector2f pos, ProjectView& view, AppScreen& appScreen) {
    if (view.btnBack.contains(pos)) {
        appScreen = AppScreen::MENU;
        return;
    }

    // Project rows: select or delete
    for (const auto& row : view.projectRows) {
        if (row.btnDelete.contains(pos)) {
            projectManager_.deleteProject(row.id);
            if (view.selectedProjectId == row.id)
                view.selectedProjectId.clear();
            return;
        }
        if (row.rect.contains(pos)) {
            view.selectedProjectId = row.id;
            view.newAssetTypeInputActive = false;
            view.newAssetTypeName.clear();
            return;
        }
    }

    // New project button / input commit
    if (view.btnNewProject.contains(pos)) {
        if (!view.newProjectInputActive) {
            view.newProjectInputActive = true;
            view.newProjectName.clear();
            view.newProjectCursor = 0;
        } else {
            commitNewProject(view);
        }
        return;
    }

    // Asset type rows: generate or delete
    for (const auto& row : view.assetTypeRows) {
        if (row.btnDelete.contains(pos)) {
            projectManager_.deleteAssetType(row.projectId, row.assetTypeId);
            return;
        }
        if (row.btnGenerate.contains(pos)) {
            const auto proj = projectManager_.getProject(row.projectId);
            if (!proj) return;
            for (const auto& at : proj->assetTypes) {
                if (at.id != row.assetTypeId) continue;
                ResolvedProjectContext ctx;
                ctx.projectId       = proj->id;
                ctx.projectName     = proj->name;
                ctx.assetTypeId     = at.id;
                ctx.assetTypeName   = at.name;
                ctx.stylePrompt     = proj->stylePrompt;
                ctx.assetTypeTokens = at.promptTokens;
                ctx.outputSubpath   = sanitiseName(proj->name) + "/" + sanitiseName(at.name);
                ctx.allAssetTypes   = proj->assetTypes;
                pendingGeneration_  = ctx;
                Logger::info("ProjectController: generate project='" + proj->name
                             + "' assetType='" + at.name + "'");
                break;
            }
            return;
        }
    }

    // Add asset type button / input commit
    if (!view.selectedProjectId.empty() && view.btnAddAssetType.contains(pos)) {
        if (!view.newAssetTypeInputActive) {
            view.newAssetTypeInputActive = true;
            view.newAssetTypeName.clear();
            view.newAssetTypeCursor = 0;
        } else {
            commitNewAssetType(view);
        }
        return;
    }
}

void ProjectController::handleTextInput(sf::Uint32 unicode, ProjectView& view) {
    if (unicode < 32 || unicode > 126) return;
    const char ch = static_cast<char>(unicode);
    if (view.newProjectInputActive) {
        view.newProjectName.insert(view.newProjectName.begin() + view.newProjectCursor, ch);
        ++view.newProjectCursor;
    } else if (view.newAssetTypeInputActive) {
        view.newAssetTypeName.insert(view.newAssetTypeName.begin() + view.newAssetTypeCursor, ch);
        ++view.newAssetTypeCursor;
    }
}

void ProjectController::commitNewProject(ProjectView& view) {
    const std::string name = view.newProjectName;
    view.newProjectInputActive = false;
    view.newProjectName.clear();
    view.newProjectCursor = 0;
    if (name.empty()) return;
    const Project p = projectManager_.createProject(name);
    view.selectedProjectId = p.id;
    Logger::info("ProjectController: created project '" + name + "' id=" + p.id);
}

void ProjectController::commitNewAssetType(ProjectView& view) {
    const std::string name = view.newAssetTypeName;
    view.newAssetTypeInputActive = false;
    view.newAssetTypeName.clear();
    view.newAssetTypeCursor = 0;
    if (name.empty() || view.selectedProjectId.empty()) return;
    projectManager_.addAssetType(view.selectedProjectId, name);
    Logger::info("ProjectController: added asset type '" + name
                 + "' to project=" + view.selectedProjectId);
}

ResolvedProjectContext ProjectController::consumePendingGeneration() {
    ResolvedProjectContext ctx = pendingGeneration_;
    pendingGeneration_ = {};
    return ctx;
}
