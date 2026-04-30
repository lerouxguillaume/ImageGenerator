#pragma once
#include "Screen.hpp"
#include "ImageGeneratorView.hpp"
#include "../projects/Project.hpp"
#include "../ui/widgets/MultiLineTextArea.hpp"
#include <array>
#include <string>
#include <vector>

class ProjectView : public Screen {
public:
    enum class ToolbarField { None, Steps, Cfg, Images, Seed };

    std::vector<Project> projects;
    std::string          selectedProjectId;
    std::string          selectedAssetTypeId;

    sf::FloatRect btnBack;
    sf::FloatRect btnSettings;
    sf::FloatRect btnNewProject;
    sf::FloatRect btnChooseProject;
    sf::FloatRect btnSaveTheme;
    sf::FloatRect btnAddAssetType;
    sf::FloatRect btnSaveAsset;
    sf::FloatRect btnGenerateAsset;
    sf::FloatRect btnModelCycle;
    sf::FloatRect btnStepsDown;
    sf::FloatRect btnStepsUp;
    sf::FloatRect btnCfgDown;
    sf::FloatRect btnCfgUp;
    sf::FloatRect btnImagesDown;
    sf::FloatRect btnImagesUp;
    sf::FloatRect stepsField;
    sf::FloatRect cfgField;
    sf::FloatRect imagesField;
    sf::FloatRect seedField;

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
        sf::FloatRect btnDelete;
    };
    std::vector<AssetTypeRow> assetTypeRows;

    struct AssetTemplateOption {
        std::string   id;
        std::string   label;
        sf::FloatRect rect;
    };
    std::vector<AssetTemplateOption> assetTemplateOptions;
    bool                             showAssetTemplatePicker = false;
    sf::FloatRect                    assetTemplatePickerRect;

    bool        newProjectInputActive = false;
    std::string newProjectName;
    int         newProjectCursor = 0;

    bool        newAssetTypeInputActive = false;
    std::string newAssetTypeName;
    int         newAssetTypeCursor = 0;

    MultiLineTextArea themePositiveArea{2000, 4};
    MultiLineTextArea themeNegativeArea{2000, 3};
    MultiLineTextArea assetPositiveArea{1200, 4};
    MultiLineTextArea assetNegativeArea{1200, 3};

    // Constraint toggle hit-rects (set each frame by render).
    // Pack: [0]=transparentBg [1]=isometric [2]=centered [3]=fullVisible [4]=noClutter [5]=noFloor
    std::array<sf::FloatRect, 6> packConstraintToggles = {};
    // Asset: [0]=allowFloor [1]=allowScene [2]=tileable [3]=topSurface
    std::array<sf::FloatRect, 4> assetConstraintToggles = {};
    // Asset spec: orientation [0..5]=Unset/LeftWall/RightWall/FloorTile/Prop/Character
    std::array<sf::FloatRect, 6> assetSpecOrientationToggles = {};
    // Asset spec: misc [0]=requiresTransparency [1]=isTileable
    std::array<sf::FloatRect, 2> assetSpecMiscToggles = {};
    // Asset spec: shape policy [0]=Freeform [1]=Bounded [2]=SilhouetteLocked
    std::array<sf::FloatRect, 3> assetSpecShapePolicyToggles = {};

    bool        themeDirty = false;
    bool        assetDirty = false;
    bool        showProjectBrowser = true;
    int         assetListScroll = 0;
    sf::FloatRect assetListViewport;
    ToolbarField activeToolbarField = ToolbarField::None;
    std::string  toolbarInput;
    std::string loadedProjectId;
    std::string loadedAssetTypeId;
    ImageGeneratorView generatorView{WorkflowMode::Generate};

    void render(sf::RenderWindow& win) override;
};
