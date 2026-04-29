#include "ProjectController.hpp"
#include "../managers/Logger.hpp"
#include "../prompt/PromptCompiler.hpp"
#include "../prompt/PromptParser.hpp"

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
    , generatorController_(cfg, WorkflowMode::Generate)
{}

void ProjectController::update(ProjectView& view) {
    view.projects = projectManager_.getAllProjects();
    if (!view.selectedProjectId.empty()) {
        const Project* selected = nullptr;
        for (const auto& p : view.projects) {
            if (p.id == view.selectedProjectId) {
                selected = &p;
                break;
            }
        }
        if (!selected) {
            view.selectedProjectId.clear();
            view.selectedAssetTypeId.clear();
        } else if (!view.selectedAssetTypeId.empty()) {
            bool assetFound = false;
            for (const auto& at : selected->assetTypes) {
                if (at.id == view.selectedAssetTypeId) {
                    assetFound = true;
                    break;
                }
            }
            if (!assetFound)
                view.selectedAssetTypeId.clear();
        }
        if (view.selectedAssetTypeId.empty() && selected && !selected->assetTypes.empty())
            view.selectedAssetTypeId = selected->assetTypes.front().id;
    }
    populateEditors(view);
    syncGeneratorSession(view);
    if (!view.selectedProjectId.empty())
        generatorController_.update(view.generatorView);
}

void ProjectController::handleEvent(const sf::Event& e, sf::RenderWindow& win,
                                    ProjectView& view, AppScreen& appScreen) {
    auto commitToolbarField = [&view]() {
        auto& sp = view.generatorView.settingsPanel;
        switch (view.activeToolbarField) {
            case ProjectView::ToolbarField::Steps:
                if (!view.toolbarInput.empty())
                    sp.generationParams.numSteps = std::clamp(std::stoi(view.toolbarInput), 5, 50);
                break;
            case ProjectView::ToolbarField::Cfg:
                if (!view.toolbarInput.empty())
                    sp.generationParams.guidanceScale = std::clamp(std::stof(view.toolbarInput), 1.0f, 20.0f);
                break;
            case ProjectView::ToolbarField::Images:
                if (!view.toolbarInput.empty())
                    sp.generationParams.numImages = std::clamp(std::stoi(view.toolbarInput), 1, 10);
                break;
            case ProjectView::ToolbarField::Seed:
                sp.seedInput = view.toolbarInput;
                break;
            case ProjectView::ToolbarField::None:
                break;
        }
        view.activeToolbarField = ProjectView::ToolbarField::None;
        view.toolbarInput.clear();
        sp.seedInputActive = false;
    };
    if (e.type == sf::Event::Closed) {
        win.close();
        return;
    }
    if (view.generatorView.showSettings) {
        AppScreen nestedScreen = appScreen;
        generatorController_.handleEvent(e, win, view.generatorView, nestedScreen);
        return;
    }
    auto generatorHasFocus = [&view]() {
        const auto& sp = view.generatorView.settingsPanel;
        return sp.seedInputActive
            || view.generatorView.resultPanel.generating;
    };

    if (!view.selectedProjectId.empty() && generatorHasFocus()
        && (e.type == sf::Event::KeyPressed || e.type == sf::Event::TextEntered
            || e.type == sf::Event::MouseWheelScrolled)) {
        AppScreen nestedScreen = appScreen;
        generatorController_.handleEvent(e, win, view.generatorView, nestedScreen);
        return;
    }

    if (e.type == sf::Event::KeyPressed) {
        if (view.activeToolbarField != ProjectView::ToolbarField::None) {
            if (e.key.code == sf::Keyboard::BackSpace && !view.toolbarInput.empty()) {
                view.toolbarInput.pop_back();
                return;
            }
            if (e.key.code == sf::Keyboard::Escape || e.key.code == sf::Keyboard::Return) {
                commitToolbarField();
                return;
            }
        }
        if (view.themePositiveArea.handleEvent(e) || view.themeNegativeArea.handleEvent(e)) {
            view.themeDirty = true;
            return;
        }
        if (view.assetPositiveArea.handleEvent(e) || view.assetNegativeArea.handleEvent(e)) {
            view.assetDirty = true;
            return;
        }
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
                view.themePositiveArea.setActive(false);
                view.themeNegativeArea.setActive(false);
                view.assetPositiveArea.setActive(false);
                view.assetNegativeArea.setActive(false);
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
    if (e.type == sf::Event::TextEntered) {
        if (view.activeToolbarField != ProjectView::ToolbarField::None) {
            const sf::Uint32 unicode = e.text.unicode;
            if (unicode >= '0' && unicode <= '9')
                view.toolbarInput.push_back(static_cast<char>(unicode));
            else if (view.activeToolbarField == ProjectView::ToolbarField::Cfg
                     && unicode == '.' && view.toolbarInput.find('.') == std::string::npos)
                view.toolbarInput.push_back('.');
            return;
        }
        if (view.themePositiveArea.handleEvent(e) || view.themeNegativeArea.handleEvent(e)) {
            view.themeDirty = true;
            return;
        }
        if (view.assetPositiveArea.handleEvent(e) || view.assetNegativeArea.handleEvent(e)) {
            view.assetDirty = true;
            return;
        }
        const sf::Uint32 unicode = e.text.unicode;
        if (unicode >= 32 && unicode <= 126) {
            const char ch = static_cast<char>(unicode);
            if (view.newProjectInputActive) {
                view.newProjectName.insert(view.newProjectName.begin() + view.newProjectCursor, ch);
                ++view.newProjectCursor;
            } else if (view.newAssetTypeInputActive) {
                view.newAssetTypeName.insert(view.newAssetTypeName.begin() + view.newAssetTypeCursor, ch);
                ++view.newAssetTypeCursor;
            }
        }
    }

    if (e.type == sf::Event::MouseWheelScrolled) {
        const sf::Vector2f pos = win.mapPixelToCoords({static_cast<int>(e.mouseWheelScroll.x),
                                                       static_cast<int>(e.mouseWheelScroll.y)});
        if (view.themePositiveArea.getRect().contains(pos))
            view.themePositiveArea.handleScroll(-e.mouseWheelScroll.delta);
        else if (view.themeNegativeArea.getRect().contains(pos))
            view.themeNegativeArea.handleScroll(-e.mouseWheelScroll.delta);
        else if (view.assetPositiveArea.getRect().contains(pos))
            view.assetPositiveArea.handleScroll(-e.mouseWheelScroll.delta);
        else if (view.assetNegativeArea.getRect().contains(pos))
            view.assetNegativeArea.handleScroll(-e.mouseWheelScroll.delta);
        else if (view.assetListViewport.contains(pos))
            view.assetListScroll = std::max(0, view.assetListScroll - static_cast<int>(e.mouseWheelScroll.delta));
    }

    if (e.type == sf::Event::MouseButtonPressed && e.mouseButton.button == sf::Mouse::Left)
        handleClick(win.mapPixelToCoords({e.mouseButton.x, e.mouseButton.y}), win, view, appScreen);
}

void ProjectController::handleClick(sf::Vector2f pos, sf::RenderWindow& win, ProjectView& view, AppScreen& appScreen) {
    const bool clickedThemePos = view.themePositiveArea.getRect().contains(pos);
    const bool clickedThemeNeg = view.themeNegativeArea.getRect().contains(pos);
    const bool clickedAssetPos = view.assetPositiveArea.getRect().contains(pos);
    const bool clickedAssetNeg = view.assetNegativeArea.getRect().contains(pos);
    view.themePositiveArea.setActive(clickedThemePos);
    view.themeNegativeArea.setActive(clickedThemeNeg);
    view.assetPositiveArea.setActive(clickedAssetPos);
    view.assetNegativeArea.setActive(clickedAssetNeg);
    if (clickedThemePos || clickedThemeNeg || clickedAssetPos || clickedAssetNeg) {
        if (clickedThemePos) view.themePositiveArea.handleClick(pos);
        if (clickedThemeNeg) view.themeNegativeArea.handleClick(pos);
        if (clickedAssetPos) view.assetPositiveArea.handleClick(pos);
        if (clickedAssetNeg) view.assetNegativeArea.handleClick(pos);
        return;
    }

    if (view.btnBack.contains(pos)) {
        appScreen = AppScreen::MENU;
        return;
    }
    if (view.btnSettings.contains(pos)) {
        generatorController_.openSettingsDialog(view.generatorView);
        return;
    }

    if (!view.selectedProjectId.empty() && view.btnChooseProject.contains(pos)) {
        view.showProjectBrowser = !view.showProjectBrowser;
        view.newProjectInputActive = false;
        view.newAssetTypeInputActive = false;
        return;
    }

    if (!view.selectedProjectId.empty()
        && view.generatorView.resultPanel.getRect().contains(pos)) {
        sf::Event synthetic;
        synthetic.type = sf::Event::MouseButtonPressed;
        synthetic.mouseButton.button = sf::Mouse::Left;
        synthetic.mouseButton.x = static_cast<int>(pos.x);
        synthetic.mouseButton.y = static_cast<int>(pos.y);
        AppScreen nestedScreen = appScreen;
        generatorController_.handleEvent(synthetic, win, view.generatorView, nestedScreen);
        return;
    }

    auto& sp = view.generatorView.settingsPanel;
    auto activateField = [&view](ProjectView::ToolbarField field, const std::string& value) {
        view.activeToolbarField = field;
        view.toolbarInput = value;
    };
    if (view.stepsField.contains(pos)) {
        activateField(ProjectView::ToolbarField::Steps, std::to_string(sp.generationParams.numSteps));
        return;
    }
    if (view.cfgField.contains(pos)) {
        char cfgBuf[16];
        std::snprintf(cfgBuf, sizeof(cfgBuf), "%.1f", sp.generationParams.guidanceScale);
        activateField(ProjectView::ToolbarField::Cfg, cfgBuf);
        return;
    }
    if (view.imagesField.contains(pos)) {
        activateField(ProjectView::ToolbarField::Images, std::to_string(sp.generationParams.numImages));
        return;
    }
    if (view.seedField.contains(pos)) {
        activateField(ProjectView::ToolbarField::Seed, sp.seedInput);
        sp.seedInputActive = true;
        return;
    }
    view.activeToolbarField = ProjectView::ToolbarField::None;
    view.toolbarInput.clear();
    sp.seedInputActive = false;
    if (view.btnGenerateAsset.contains(pos)) {
        generatorController_.triggerGeneration(view.generatorView);
        return;
    }
    if (view.btnModelCycle.contains(pos) && !sp.availableModels.empty()) {
        sp.selectedModelIdx = (sp.selectedModelIdx + 1) % static_cast<int>(sp.availableModels.size());
        return;
    }
    if (view.btnStepsDown.contains(pos)) {
        sp.generationParams.numSteps = std::max(5, sp.generationParams.numSteps - 1);
        return;
    }
    if (view.btnStepsUp.contains(pos)) {
        sp.generationParams.numSteps = std::min(50, sp.generationParams.numSteps + 1);
        return;
    }
    if (view.btnCfgDown.contains(pos)) {
        sp.generationParams.guidanceScale = std::max(1.0f, sp.generationParams.guidanceScale - 0.5f);
        return;
    }
    if (view.btnCfgUp.contains(pos)) {
        sp.generationParams.guidanceScale = std::min(20.0f, sp.generationParams.guidanceScale + 0.5f);
        return;
    }
    if (view.btnImagesDown.contains(pos)) {
        sp.generationParams.numImages = std::max(1, sp.generationParams.numImages - 1);
        return;
    }
    if (view.btnImagesUp.contains(pos)) {
        sp.generationParams.numImages = std::min(10, sp.generationParams.numImages + 1);
        return;
    }

    // Project rows: select or delete
    for (const auto& row : view.projectRows) {
        if (row.btnDelete.contains(pos)) {
            projectManager_.deleteProject(row.id);
            if (view.selectedProjectId == row.id)
                view.selectedProjectId.clear();
            if (view.loadedProjectId == row.id) {
                view.loadedProjectId.clear();
                view.loadedAssetTypeId.clear();
                view.themeDirty = false;
                view.assetDirty = false;
            }
            return;
        }
        if (row.rect.contains(pos)) {
            if (!view.selectedProjectId.empty() && view.themeDirty)
                saveTheme(view);
            if (!view.selectedAssetTypeId.empty() && view.assetDirty)
                saveAssetType(view);
            view.selectedProjectId = row.id;
            view.selectedAssetTypeId.clear();
            view.showProjectBrowser = false;
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

    if (!view.selectedProjectId.empty() && view.btnSaveTheme.contains(pos)) {
        saveTheme(view);
        return;
    }

    // Asset type rows: select or delete
    for (const auto& row : view.assetTypeRows) {
        if (row.btnDelete.contains(pos)) {
            projectManager_.deleteAssetType(row.projectId, row.assetTypeId);
            if (view.selectedAssetTypeId == row.assetTypeId) {
                view.selectedAssetTypeId.clear();
                view.loadedAssetTypeId.clear();
                view.assetDirty = false;
            }
            return;
        }
        if (row.rect.contains(pos)) {
            if (!view.selectedAssetTypeId.empty() && view.assetDirty)
                saveAssetType(view);
            view.selectedAssetTypeId = row.assetTypeId;
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

    if (!view.selectedAssetTypeId.empty() && view.btnSaveAsset.contains(pos)) {
        saveAssetType(view);
        return;
    }

    (void)appScreen;
}

void ProjectController::commitNewProject(ProjectView& view) {
    const std::string name = view.newProjectName;
    view.newProjectInputActive = false;
    view.newProjectName.clear();
    view.newProjectCursor = 0;
    if (name.empty()) return;
    const Project p = projectManager_.createProject(name);
    view.selectedProjectId = p.id;
    view.selectedAssetTypeId.clear();
    view.showProjectBrowser = false;
    Logger::info("ProjectController: created project '" + name + "' id=" + p.id);
}

void ProjectController::commitNewAssetType(ProjectView& view) {
    const std::string name = view.newAssetTypeName;
    view.newAssetTypeInputActive = false;
    view.newAssetTypeName.clear();
    view.newAssetTypeCursor = 0;
    if (name.empty() || view.selectedProjectId.empty()) return;
    const AssetType assetType = projectManager_.addAssetType(view.selectedProjectId, name);
    view.selectedAssetTypeId = assetType.id;
    Logger::info("ProjectController: added asset type '" + name
                 + "' to project=" + view.selectedProjectId);
}

void ProjectController::populateEditors(ProjectView& view) const {
    if (!view.selectedProjectId.empty() && view.loadedProjectId != view.selectedProjectId) {
        if (const auto proj = projectManager_.getProject(view.selectedProjectId)) {
            view.themePositiveArea.setText(PromptCompiler::compile(proj->stylePrompt, ModelType::SDXL));
            view.themeNegativeArea.setText(PromptCompiler::compileNegative(proj->stylePrompt));
            view.loadedProjectId = proj->id;
            view.themeDirty = false;
        }
    }

    if (view.selectedAssetTypeId.empty()) {
        view.loadedAssetTypeId.clear();
        view.assetDirty = false;
        view.assetPositiveArea.setText({});
        view.assetNegativeArea.setText({});
        return;
    }

    if (view.loadedAssetTypeId != view.selectedAssetTypeId) {
        const auto proj = projectManager_.getProject(view.selectedProjectId);
        if (!proj) return;
        for (const auto& at : proj->assetTypes) {
            if (at.id != view.selectedAssetTypeId) continue;
            view.assetPositiveArea.setText(PromptCompiler::compile(at.promptTokens, ModelType::SDXL));
            view.assetNegativeArea.setText(PromptCompiler::compileNegative(at.promptTokens));
            view.loadedAssetTypeId = at.id;
            view.assetDirty = false;
            return;
        }
    }
}

void ProjectController::syncGeneratorSession(ProjectView& view) {
    const ResolvedProjectContext ctx = buildSelectedContext(view);
    if (ctx.empty()) {
        generatorController_.clearProjectContext();
        return;
    }
    const ResolvedProjectContext current = generatorController_.getProjectContext();
    const bool sameTheme =
        PromptCompiler::compile(current.stylePrompt, ModelType::SDXL)
            == PromptCompiler::compile(ctx.stylePrompt, ModelType::SDXL)
        && PromptCompiler::compileNegative(current.stylePrompt)
            == PromptCompiler::compileNegative(ctx.stylePrompt);
    const bool sameAsset =
        PromptCompiler::compile(current.assetTypeTokens, ModelType::SDXL)
            == PromptCompiler::compile(ctx.assetTypeTokens, ModelType::SDXL)
        && PromptCompiler::compileNegative(current.assetTypeTokens)
            == PromptCompiler::compileNegative(ctx.assetTypeTokens);
    if (current.projectId == ctx.projectId
        && current.assetTypeId == ctx.assetTypeId
        && sameTheme
        && sameAsset) {
        return;
    }
    generatorController_.activateProjectSession(view.generatorView, ctx);
    generatorController_.setBackScreen(AppScreen::Projects);
}

void ProjectController::saveTheme(ProjectView& view, bool clearDirty) {
    if (view.selectedProjectId.empty()) return;
    auto proj = projectManager_.getProject(view.selectedProjectId);
    if (!proj) return;
    proj->stylePrompt = PromptParser::parse(view.themePositiveArea.getText(),
                                            view.themeNegativeArea.getText());
    projectManager_.updateProject(*proj);
    if (clearDirty)
        view.themeDirty = false;
}

void ProjectController::saveAssetType(ProjectView& view, bool clearDirty) {
    if (view.selectedProjectId.empty() || view.selectedAssetTypeId.empty()) return;
    auto proj = projectManager_.getProject(view.selectedProjectId);
    if (!proj) return;
    for (auto& at : proj->assetTypes) {
        if (at.id != view.selectedAssetTypeId) continue;
        at.promptTokens = PromptParser::parse(view.assetPositiveArea.getText(),
                                              view.assetNegativeArea.getText());
        projectManager_.updateAssetType(proj->id, at);
        if (clearDirty)
            view.assetDirty = false;
        return;
    }
}

ResolvedProjectContext ProjectController::consumePendingGeneration() {
    ResolvedProjectContext ctx = pendingGeneration_;
    pendingGeneration_ = {};
    return ctx;
}

ResolvedProjectContext ProjectController::buildSelectedContext(const ProjectView& view) const {
    if (view.selectedProjectId.empty() || view.selectedAssetTypeId.empty())
        return {};
    const auto proj = projectManager_.getProject(view.selectedProjectId);
    if (!proj)
        return {};
    for (const auto& at : proj->assetTypes) {
        if (at.id != view.selectedAssetTypeId) continue;
        ResolvedProjectContext ctx;
        ctx.projectId = proj->id;
        ctx.projectName = proj->name;
        ctx.assetTypeId = at.id;
        ctx.assetTypeName = at.name;
        ctx.stylePrompt = proj->stylePrompt;
        ctx.assetTypeTokens = at.promptTokens;
        ctx.outputSubpath = sanitiseName(proj->name) + "/" + sanitiseName(at.name);
        ctx.allAssetTypes = proj->assetTypes;
        return ctx;
    }
    return {};
}
