#pragma once
#include "Screen.hpp"
#include "../projects/Project.hpp"
#include <string>
#include <vector>

class ProjectView : public Screen {
public:
    // ── Data (set by controller each update) ──────────────────────────────────
    std::vector<Project> projects;
    std::string          selectedProjectId;

    // ── Hit rects (written during render, read by controller) ─────────────────
    sf::FloatRect btnBack;
    sf::FloatRect btnNewProject;

    struct ProjectRow {
        std::string   id;
        sf::FloatRect rect;
        sf::FloatRect btnDelete;
    };
    std::vector<ProjectRow> projectRows;

    struct AssetTypeRow {
        std::string   projectId;
        std::string   assetTypeId;
        sf::FloatRect rect;
        sf::FloatRect btnGenerate;
        sf::FloatRect btnDelete;
    };
    std::vector<AssetTypeRow> assetTypeRows;

    sf::FloatRect btnAddAssetType;

    // Inline "new project" name input state
    bool        newProjectInputActive = false;
    std::string newProjectName;
    int         newProjectCursor = 0;

    // Inline "new asset type" name input state (within selected project)
    bool        newAssetTypeInputActive = false;
    std::string newAssetTypeName;
    int         newAssetTypeCursor = 0;

    void render(sf::RenderWindow& win) override;
};
