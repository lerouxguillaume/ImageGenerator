#pragma once
#include <SFML/Graphics.hpp>
#include "../config/AppConfig.hpp"
#include "../controllers/MenuController.hpp"
#include "ImageGeneratorController.hpp"
#include "../projects/ProjectManager.hpp"
#include "../projects/Project.hpp"
#include "../projects/AssetTypeTemplate.hpp"
#include "../views/ProjectView.hpp"

class ProjectController {
public:
    explicit ProjectController(AppConfig& cfg);

    void handleEvent(const sf::Event& e, sf::RenderWindow& win,
                     ProjectView& view, AppScreen& appScreen);
    void update(ProjectView& view);

    // Returns the resolved context if the user clicked "Generate >" and clears it.
    // Check ctx.empty() before use.
    ResolvedProjectContext consumePendingGeneration();

private:
    void handleClick(sf::Vector2f pos, sf::RenderWindow& win, ProjectView& view, AppScreen& appScreen);
    void commitNewProject(ProjectView& view);
    void commitNewAssetType(ProjectView& view);
    void createAssetTypeFromTemplate(ProjectView& view, const AssetTypeTemplate& assetTemplate);
    void populateEditors(ProjectView& view) const;
    void syncGeneratorSession(ProjectView& view);
    void saveTheme(ProjectView& view, bool clearDirty = true);
    void saveAssetType(ProjectView& view, bool clearDirty = true);
    ResolvedProjectContext buildSelectedContext(const ProjectView& view) const;

    AppConfig&             config_;
    ProjectManager         projectManager_;
    ResolvedProjectContext pendingGeneration_;
    ImageGeneratorController generatorController_;
};
