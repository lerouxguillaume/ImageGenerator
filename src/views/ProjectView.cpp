#include "ProjectView.hpp"
#include "../enum/constants.hpp"
#include "../ui/Buttons.hpp"
#include "../ui/Helpers.hpp"
#include <filesystem>

static constexpr float VPAD  = 18.f;
static constexpr float ROW_H = 40.f;
static constexpr float GAP   = 8.f;

namespace {
void drawToolbarValue(sf::RenderWindow& win, sf::Font& font,
                      const std::string& label, const std::string& value,
                      float x, float y, float w, bool active = false) {
    Helpers::drawText(win, font, label, Col::Muted, x, y, 11, false);
    Helpers::drawRect(win, {x, y + 16.f, w, 28.f}, Col::Panel, active ? Col::GoldLt : Col::Border, 1.f);
    Helpers::drawTextC(win, font, value, Col::GoldLt, x + w / 2.f, y + 24.f, 12, false);
}
}

void ProjectView::render(sf::RenderWindow& win) {
    const float W = static_cast<float>(win.getSize().x);
    const float H = static_cast<float>(win.getSize().y);

    Helpers::drawRect(win, {0.f, 0.f, W, static_cast<float>(HEADER_H)}, Col::Panel, Col::Border, 1.f);
    Helpers::drawText(win, font, "Projects", Col::GoldLt, VPAD, 18.f, 20, true);

    btnBack = {W - 248.f - VPAD, (HEADER_H - 30.f) / 2.f, 120.f, 30.f};
    drawButton(win, btnBack, "< Back", Col::Panel2, Col::Muted, false, 13, font);
    btnSettings = {W - 120.f - VPAD, (HEADER_H - 30.f) / 2.f, 120.f, 30.f};
    drawButton(win, btnSettings, "Settings", Col::Panel2, Col::Muted, false, 13, font);

    const float bodyY = static_cast<float>(HEADER_H) + VPAD;
    const float bodyH = H - bodyY - VPAD;
    const bool showBrowser = showProjectBrowser || selectedProjectId.empty();
    const float browserW = 260.f;
    const float rightX = showBrowser ? (browserW + VPAD * 2.f) : VPAD;
    const float rightW = showBrowser ? (W - rightX - VPAD) : (W - VPAD * 2.f);

    projectRows.clear();
    btnNewProject = {};
    if (showBrowser) {
        Helpers::drawRect(win, {VPAD, bodyY, browserW, bodyH}, Col::Panel, Col::Border, 1.f);
        Helpers::drawText(win, font, "Asset Packs", Col::Gold, VPAD + 10.f, bodyY + 10.f, 13, true);

        float rowY = bodyY + 36.f;
        for (const auto& proj : projects) {
            const bool selected = proj.id == selectedProjectId;
            const sf::Color bg = selected ? Col::Blue : Col::Panel2;
            const sf::FloatRect rowRect{VPAD + 4.f, rowY, browserW - 8.f, ROW_H};

            Helpers::drawRect(win, rowRect, bg, selected ? Col::BorderHi : Col::Border, 1.f);
            Helpers::drawText(win, font, proj.name, selected ? Col::GoldLt : Col::Text,
                              rowRect.left + 8.f, rowRect.top + 12.f, 13, false);

            const sf::FloatRect delBtn{rowRect.left + rowRect.width - 28.f, rowRect.top + 6.f, 22.f, 28.f};
            drawButton(win, delBtn, "x", Col::Panel, Col::Red, false, 11, font);

            projectRows.push_back({proj.id, rowRect, delBtn});
            rowY += ROW_H + GAP;
        }

        if (newProjectInputActive) {
            const sf::FloatRect inputRect{VPAD + 4.f, rowY, browserW - 8.f, ROW_H};
            Helpers::drawRect(win, inputRect, Col::Panel2, Col::BorderHi, 1.f);
            Helpers::drawText(win, font, newProjectName + "|", Col::Text,
                              inputRect.left + 8.f, inputRect.top + 12.f, 13, false);
            btnNewProject = inputRect;
        } else {
            btnNewProject = {VPAD + 4.f, rowY, browserW - 8.f, ROW_H};
            drawButton(win, btnNewProject, "+ New Project", Col::Panel2, Col::Gold, false, 13, font);
        }
    }

    Helpers::drawRect(win, {rightX, bodyY, rightW, bodyH}, Col::Panel, Col::Border, 1.f);
    if (selectedProjectId.empty()) {
        Helpers::drawTextC(win, font, "Select or create a project", Col::Muted,
                           rightX + rightW / 2.f, bodyY + bodyH / 2.f - 12.f, 16, false);
        Helpers::drawTextC(win, font, "Each project stores the shared theme and per-asset prompts.",
                           Col::Border, rightX + rightW / 2.f, bodyY + bodyH / 2.f + 16.f, 12, false);
        assetTypeRows.clear();
        btnSaveTheme = {};
        btnAddAssetType = {};
        btnSaveAsset = {};
        btnGenerateAsset = {};
        btnChooseProject = {};
        themePositiveArea.setRect({});
        themeNegativeArea.setRect({});
        assetPositiveArea.setRect({});
        assetNegativeArea.setRect({});
        generatorView.resultPanel.setRect({});
        return;
    }

    const Project* proj = nullptr;
    for (const auto& p : projects) {
        if (p.id == selectedProjectId) {
            proj = &p;
            break;
        }
    }
    if (!proj) return;

    btnChooseProject = {rightX + rightW - 132.f, bodyY + 8.f, 120.f, 28.f};
    drawButton(win, btnChooseProject, showBrowser ? "Hide List" : "Choose Project",
               Col::Panel2, Col::Text, false, 12, font);
    Helpers::drawText(win, font, proj->name, Col::GoldLt, rightX + 12.f, bodyY + 10.f, 18, true);

    const float sectionX = rightX + 12.f;
    const float sectionW = rightW - 24.f;
    const float toolbarY = bodyY + 42.f;
    const float toolbarH = 56.f;
    const float railW = std::min(400.f, std::max(320.f, sectionW * 0.34f));
    const float gapX = 14.f;
    const float resultX = sectionX + railW + gapX;
    const float resultW = sectionW - railW - gapX;
    const float railY = toolbarY + toolbarH + 12.f;
    const float railH = bodyY + bodyH - railY;

    Helpers::drawRect(win, {sectionX, toolbarY, sectionW, toolbarH}, Col::Panel2, Col::Border, 1.f);
    Helpers::drawText(win, font, "Generation", Col::Gold, sectionX + 10.f, toolbarY + 10.f, 13, true);

    auto& sp = generatorView.settingsPanel;
    const float rowY = toolbarY + 10.f;
    const float modelW = 170.f;
    const float genW = 106.f;
    const float seedW = 120.f;
    const float statW = 78.f;
    const float gap = 8.f;
    const float rowW = modelW + seedW + statW * 3.f + genW + gap * 5.f;
    const float rowX = sectionX + (sectionW - rowW) / 2.f;

    btnModelCycle = {rowX, rowY + 16.f, modelW, 28.f};
    const std::string modelName = sp.availableModels.empty()
        ? "(no models)"
        : std::filesystem::path(sp.availableModels[static_cast<size_t>(sp.selectedModelIdx)]).filename().string();
    drawToolbarValue(win, font, "Model", modelName, rowX, rowY, modelW);

    seedField = {rowX + modelW + gap, rowY + 16.f, seedW, 28.f};
    Helpers::drawText(win, font, "Seed", Col::Muted, seedField.left, rowY, 11, false);
    Helpers::drawRect(win, seedField, Col::Panel,
                      activeToolbarField == ToolbarField::Seed ? Col::GoldLt : Col::Border, 1.f);
    Helpers::drawText(win, font,
                      (activeToolbarField == ToolbarField::Seed ? toolbarInput + "|" :
                       (sp.seedInput.empty() ? "(random)" : sp.seedInput)),
                      (sp.seedInput.empty() && activeToolbarField != ToolbarField::Seed) ? Col::Border : Col::GoldLt,
                      seedField.left + 8.f, seedField.top + 7.f, 12, false);

    const float stepsX = seedField.left + seedW + gap;
    stepsField = {stepsX, rowY + 16.f, statW, 28.f};
    drawToolbarValue(win, font,
                     "Steps",
                     activeToolbarField == ToolbarField::Steps
                        ? toolbarInput + "|"
                        : std::to_string(sp.generationParams.numSteps),
                     stepsX, rowY, statW, activeToolbarField == ToolbarField::Steps);
    btnStepsDown = {stepsX + 2.f, rowY + 18.f, 16.f, 24.f};
    btnStepsUp   = {stepsX + statW - 18.f, rowY + 18.f, 16.f, 24.f};
    drawButton(win, btnStepsDown, "-", Col::Panel2, Col::Text, false, 12, font);
    drawButton(win, btnStepsUp, "+", Col::Panel2, Col::Text, false, 12, font);

    const float cfgX = stepsX + statW + gap;
    char cfgBuf[16];
    std::snprintf(cfgBuf, sizeof(cfgBuf), "%.1f", sp.generationParams.guidanceScale);
    cfgField = {cfgX, rowY + 16.f, statW, 28.f};
    drawToolbarValue(win, font, "CFG",
                     activeToolbarField == ToolbarField::Cfg ? toolbarInput + "|" : std::string(cfgBuf),
                     cfgX, rowY, statW, activeToolbarField == ToolbarField::Cfg);
    btnCfgDown = {cfgX + 2.f, rowY + 18.f, 16.f, 24.f};
    btnCfgUp   = {cfgX + statW - 18.f, rowY + 18.f, 16.f, 24.f};
    drawButton(win, btnCfgDown, "-", Col::Panel2, Col::Text, false, 12, font);
    drawButton(win, btnCfgUp, "+", Col::Panel2, Col::Text, false, 12, font);

    const float imagesX = cfgX + statW + gap;
    imagesField = {imagesX, rowY + 16.f, statW, 28.f};
    drawToolbarValue(win, font, "Images",
                     activeToolbarField == ToolbarField::Images
                        ? toolbarInput + "|"
                        : std::to_string(sp.generationParams.numImages),
                     imagesX, rowY, statW, activeToolbarField == ToolbarField::Images);
    btnImagesDown = {imagesX + 2.f, rowY + 18.f, 16.f, 24.f};
    btnImagesUp   = {imagesX + statW - 18.f, rowY + 18.f, 16.f, 24.f};
    drawButton(win, btnImagesDown, "-", Col::Panel2, Col::Text, false, 12, font);
    drawButton(win, btnImagesUp, "+", Col::Panel2, Col::Text, false, 12, font);

    btnGenerateAsset = {imagesX + statW + gap, rowY + 16.f, genW, 28.f};
    drawButton(win, btnGenerateAsset, "Generate", Col::Blue, Col::GoldLt, false, 12, font);

    const sf::FloatRect themeBox{sectionX, railY, railW, 150.f};
    Helpers::drawRect(win, themeBox, Col::Panel2, Col::Border, 1.f);
    Helpers::drawText(win, font, "Project Theme", Col::Gold, themeBox.left + 10.f, themeBox.top + 10.f, 13, true);
    Helpers::drawText(win, font, "Shared positive prompt", Col::Muted, themeBox.left + 10.f, themeBox.top + 34.f, 11, false);
    themePositiveArea.setRect({themeBox.left + 10.f, themeBox.top + 52.f, themeBox.width - 20.f, 42.f});
    themePositiveArea.render(win, font);

    Helpers::drawText(win, font, "Shared negative prompt", Col::Muted, themeBox.left + 10.f, themeBox.top + 100.f, 11, false);
    themeNegativeArea.setRect({themeBox.left + 10.f, themeBox.top + 118.f, themeBox.width - 140.f, 22.f});
    themeNegativeArea.setTextColor(Col::Muted);
    themeNegativeArea.render(win, font);

    btnSaveTheme = {themeBox.left + themeBox.width - 122.f, themeBox.top + 114.f, 112.f, 26.f};
    drawButton(win, btnSaveTheme, themeDirty ? "Save Theme *" : "Save Theme",
               Col::Panel, themeDirty ? Col::GoldLt : Col::Text, false, 12, font);

    const float assetWorkspaceY = themeBox.top + themeBox.height + 12.f;
    const float assetWorkspaceH = 240.f;
    const float listW = 134.f;
    const float detailX = sectionX + listW + 14.f;
    const float detailW = railW - listW - 14.f;

    Helpers::drawRect(win, {sectionX, assetWorkspaceY, railW, assetWorkspaceH}, Col::Panel2, Col::Border, 1.f);
    Helpers::drawText(win, font, "Asset Types", Col::Gold, sectionX + 10.f, assetWorkspaceY + 10.f, 13, true);

    assetTypeRows.clear();
    const float listAreaTop = assetWorkspaceY + 36.f;
    const float listAreaBottom = assetWorkspaceY + assetWorkspaceH - 42.f;
    const float rowStep = 38.f;
    const int visibleRows = std::max(1, static_cast<int>((listAreaBottom - listAreaTop) / rowStep));
    const int maxScroll = std::max(0, static_cast<int>(proj->assetTypes.size()) - visibleRows);
    assetListScroll = std::clamp(assetListScroll, 0, maxScroll);
    assetListViewport = {sectionX + 4.f, listAreaTop, listW - 8.f, listAreaBottom - listAreaTop};
    float atY = listAreaTop;
    const int startIdx = assetListScroll;
    const int endIdx = std::min(static_cast<int>(proj->assetTypes.size()), startIdx + visibleRows);
    for (int idx = startIdx; idx < endIdx; ++idx) {
        const auto& at = proj->assetTypes[static_cast<size_t>(idx)];
        const bool selected = at.id == selectedAssetTypeId;
        const sf::FloatRect rowRect{sectionX + 4.f, atY, listW - 8.f, 34.f};
        Helpers::drawRect(win, rowRect, selected ? Col::Blue : Col::Panel, selected ? Col::BorderHi : Col::Border, 1.f);
        Helpers::drawText(win, font, at.name, selected ? Col::GoldLt : Col::Text,
                          rowRect.left + 8.f, rowRect.top + 10.f, 11, false);
        const sf::FloatRect delBtn{rowRect.left + rowRect.width - 24.f, rowRect.top + 5.f, 18.f, 24.f};
        drawButton(win, delBtn, "x", Col::Panel2, Col::Red, false, 11, font);
        assetTypeRows.push_back({proj->id, at.id, rowRect, delBtn});
        atY += 38.f;
    }
    if (maxScroll > 0) {
        Helpers::drawTextR(win, font,
                           std::to_string(startIdx + 1) + "-" + std::to_string(endIdx) + " / "
                               + std::to_string(proj->assetTypes.size()),
                           Col::Border, sectionX + listW - 10.f, assetWorkspaceY + assetWorkspaceH - 32.f, 10);
    }

    if (newAssetTypeInputActive) {
        const sf::FloatRect inputRect{sectionX + 4.f, assetWorkspaceY + assetWorkspaceH - 38.f, listW - 8.f, 34.f};
        Helpers::drawRect(win, inputRect, Col::Panel, Col::BorderHi, 1.f);
        Helpers::drawText(win, font, newAssetTypeName + "|", Col::Text,
                          inputRect.left + 8.f, inputRect.top + 10.f, 11, false);
        btnAddAssetType = inputRect;
    } else {
        btnAddAssetType = {sectionX + 4.f, assetWorkspaceY + assetWorkspaceH - 38.f, listW - 8.f, 34.f};
        drawButton(win, btnAddAssetType, "+ Asset", Col::Panel, Col::Gold, false, 11, font);
    }

    Helpers::drawRect(win, {detailX, assetWorkspaceY + 4.f, detailW, assetWorkspaceH - 8.f}, Col::Panel, Col::Border, 1.f);
    if (selectedAssetTypeId.empty()) {
        Helpers::drawTextC(win, font, "Select an asset type", Col::Muted,
                           detailX + detailW / 2.f, assetWorkspaceY + assetWorkspaceH / 2.f - 10.f, 14, false);
        Helpers::drawTextC(win, font, "This prompt is the subject-specific layer added on top of the project theme.",
                           Col::Border, detailX + detailW / 2.f, assetWorkspaceY + assetWorkspaceH / 2.f + 16.f, 11, false);
        btnSaveAsset = {};
        assetPositiveArea.setRect({});
        assetNegativeArea.setRect({});
    } else {
        const AssetType* selectedAsset = nullptr;
        for (const auto& at : proj->assetTypes) {
            if (at.id == selectedAssetTypeId) {
                selectedAsset = &at;
                break;
            }
        }
        if (!selectedAsset) return;

        Helpers::drawText(win, font, selectedAsset->name, Col::GoldLt, detailX + 10.f, assetWorkspaceY + 14.f, 14, true);
        Helpers::drawText(win, font, "Asset prompt", Col::Muted, detailX + 10.f, assetWorkspaceY + 36.f, 11, false);
        assetPositiveArea.setRect({detailX + 10.f, assetWorkspaceY + 52.f, detailW - 20.f, 58.f});
        assetPositiveArea.render(win, font);

        Helpers::drawText(win, font, "Asset negative prompt", Col::Muted, detailX + 10.f, assetWorkspaceY + 116.f, 11, false);
        assetNegativeArea.setRect({detailX + 10.f, assetWorkspaceY + 132.f, detailW - 20.f, 38.f});
        assetNegativeArea.setTextColor(Col::Muted);
        assetNegativeArea.render(win, font);

        btnSaveAsset = {detailX + 10.f, assetWorkspaceY + 186.f, 112.f, 26.f};
        drawButton(win, btnSaveAsset, assetDirty ? "Save Asset *" : "Save Asset",
                   Col::Panel, assetDirty ? Col::GoldLt : Col::Text, false, 12, font);
        Helpers::drawText(win, font, "Generator below uses this asset type within the current project theme.",
                          Col::BlueLt, detailX + 10.f, assetWorkspaceY + 220.f, 10, false);
    }

    generatorView.resultPanel.showImproveButton = false;
    generatorView.resultPanel.showTabs = false;

    Helpers::drawRect(win, {resultX, railY, resultW, railH}, Col::Panel2, Col::Border, 1.f);
    Helpers::drawText(win, font, "Results", Col::Gold, resultX + 10.f, railY + 10.f, 13, true);
    Helpers::drawText(win, font, "Preview and gallery stay visible while you tweak theme, asset type, and run settings.",
                      Col::Muted, resultX + 72.f, railY + 11.f, 11, false);
    generatorView.resultPanel.setRect({resultX + 1.f, railY + 28.f, resultW - 2.f, railH - 29.f});
    generatorView.resultPanel.render(win, font, generatorView.settingsPanel.generationParams.numSteps);

    if (generatorView.showSettings)
        generatorView.settingsModal.render(win, font);
}
