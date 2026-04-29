#include "ProjectView.hpp"
#include "../ui/Buttons.hpp"
#include "../ui/Helpers.hpp"
#include "../ui/Theme.h"
#include <filesystem>

namespace {
void drawToolbarValue(sf::RenderWindow& win, sf::Font& font,
                      const std::string& label, const std::string& value,
                      float x, float y, float w, const Theme& theme, bool active = false) {
    const auto& colors = theme.colors();
    const auto& metrics = theme.metrics();
    const auto& type = theme.typography();
    Helpers::drawText(win, font, label, colors.muted, x, y, type.compact, false);
    Helpers::drawRect(win, {x, y + metrics.toolbarLabelGap, w, metrics.toolbarFieldHeight},
                      colors.panel, active ? colors.goldLt : colors.border, metrics.borderWidth);
    Helpers::drawTextC(win, font, value, colors.goldLt,
                       x + w / 2.f, y + metrics.toolbarLabelGap + 8.f, type.body, false);
}
}

void ProjectView::render(sf::RenderWindow& win) {
    const auto& theme = Theme::instance();
    const auto& colors = theme.colors();
    const auto& metrics = theme.metrics();
    const auto& type = theme.typography();
    const float W = static_cast<float>(win.getSize().x);
    const float H = static_cast<float>(win.getSize().y);
    const float vPad = metrics.space2xl;
    const float gap = metrics.spaceSm;

    Helpers::drawRect(win, {0.f, 0.f, W, static_cast<float>(metrics.headerHeight)},
                      colors.panel, colors.border, metrics.borderWidth);
    Helpers::drawText(win, font, "Projects", colors.goldLt, vPad, 18.f, type.pageTitle, true);

    btnBack = {W - 248.f - vPad, (metrics.headerHeight - metrics.buttonHeight) / 2.f, 120.f, metrics.buttonHeight};
    drawButton(win, btnBack, "< Back", colors.panel2, colors.muted, false, type.sectionTitle, font);
    btnSettings = {W - 120.f - vPad, (metrics.headerHeight - metrics.buttonHeight) / 2.f, 120.f, metrics.buttonHeight};
    drawButton(win, btnSettings, "Settings", colors.panel2, colors.muted, false, type.sectionTitle, font);

    const float bodyY = static_cast<float>(metrics.headerHeight) + vPad;
    const float bodyH = H - bodyY - vPad;
    const bool showBrowser = showProjectBrowser || selectedProjectId.empty();
    const float browserW = metrics.projectBrowserWidth;
    const float rightX = showBrowser ? (browserW + vPad * 2.f) : vPad;
    const float rightW = showBrowser ? (W - rightX - vPad) : (W - vPad * 2.f);

    projectRows.clear();
    btnNewProject = {};
    if (showBrowser) {
        Helpers::drawRect(win, {vPad, bodyY, browserW, bodyH}, colors.panel, colors.border, metrics.borderWidth);
        Helpers::drawText(win, font, "Asset Packs", colors.gold, vPad + metrics.spaceMd, bodyY + metrics.spaceMd, type.sectionTitle, true);

        float rowY = bodyY + 36.f;
        for (const auto& proj : projects) {
            const bool selected = proj.id == selectedProjectId;
            const sf::Color bg = selected ? colors.blue : colors.panel2;
            const sf::FloatRect rowRect{vPad + metrics.spaceXs, rowY, browserW - metrics.spaceSm, metrics.projectRowHeight};

            Helpers::drawRect(win, rowRect, bg, selected ? colors.borderHi : colors.border, metrics.borderWidth);
            Helpers::drawText(win, font, proj.name, selected ? colors.goldLt : colors.text,
                              rowRect.left + metrics.spaceSm, rowRect.top + 12.f, type.sectionTitle, false);

            const sf::FloatRect delBtn{rowRect.left + rowRect.width - 28.f, rowRect.top + 6.f, 22.f, 28.f};
            drawButton(win, delBtn, "x", colors.panel, colors.red, false, type.compact, font);

            projectRows.push_back({proj.id, rowRect, delBtn});
            rowY += metrics.projectRowHeight + gap;
        }

        if (newProjectInputActive) {
            const sf::FloatRect inputRect{vPad + metrics.spaceXs, rowY, browserW - metrics.spaceSm, metrics.projectRowHeight};
            Helpers::drawRect(win, inputRect, colors.panel2, colors.borderHi, metrics.borderWidth);
            Helpers::drawText(win, font, newProjectName + "|", colors.text,
                              inputRect.left + metrics.spaceSm, inputRect.top + 12.f, type.sectionTitle, false);
            btnNewProject = inputRect;
        } else {
            btnNewProject = {vPad + metrics.spaceXs, rowY, browserW - metrics.spaceSm, metrics.projectRowHeight};
            drawButton(win, btnNewProject, "+ New Project", colors.panel2, colors.gold, false, type.sectionTitle, font);
        }
    }

    Helpers::drawRect(win, {rightX, bodyY, rightW, bodyH}, colors.panel, colors.border, metrics.borderWidth);
    if (selectedProjectId.empty()) {
        Helpers::drawTextC(win, font, "Select or create a project", colors.muted,
                           rightX + rightW / 2.f, bodyY + bodyH / 2.f - 12.f, 16, false);
        Helpers::drawTextC(win, font, "Each project stores the shared theme and per-asset prompts.",
                           colors.border, rightX + rightW / 2.f, bodyY + bodyH / 2.f + 16.f, type.body, false);
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

    btnChooseProject = {rightX + rightW - 132.f, bodyY + metrics.spaceSm, 120.f, metrics.compactButtonHeight};
    drawButton(win, btnChooseProject, showBrowser ? "Hide List" : "Choose Project",
               colors.panel2, colors.text, false, type.body, font);
    Helpers::drawText(win, font, proj->name, colors.goldLt, rightX + metrics.spaceLg, bodyY + metrics.spaceMd, type.projectTitle, true);

    const float sectionX = rightX + metrics.spaceLg;
    const float sectionW = rightW - metrics.spaceLg * 2.f;
    const float toolbarY = bodyY + 42.f;
    const float toolbarH = 56.f;
    const float railW = std::min(metrics.rightRailPreferredWidth,
                                 std::max(metrics.rightRailMinWidth, sectionW * metrics.rightRailRatio));
    const float gapX = metrics.spaceXl;
    const float resultX = sectionX + railW + gapX;
    const float resultW = sectionW - railW - gapX;
    const float railY = toolbarY + toolbarH + metrics.spaceLg;
    const float railH = bodyY + bodyH - railY;

    Helpers::drawRect(win, {sectionX, toolbarY, sectionW, toolbarH}, colors.panel2, colors.border, metrics.borderWidth);
    Helpers::drawText(win, font, "Generation", colors.gold, sectionX + metrics.spaceMd, toolbarY + metrics.spaceMd, type.sectionTitle, true);

    auto& sp = generatorView.settingsPanel;
    const float rowY = toolbarY + metrics.spaceMd;
    const float modelW = 170.f;
    const float genW = 106.f;
    const float seedW = 120.f;
    const float statW = 78.f;
    const float rowW = modelW + seedW + statW * 3.f + genW + gap * 5.f;
    const float rowX = sectionX + (sectionW - rowW) / 2.f;

    btnModelCycle = {rowX, rowY + metrics.toolbarLabelGap, modelW, metrics.toolbarFieldHeight};
    const std::string modelName = sp.availableModels.empty()
        ? "(no models)"
        : std::filesystem::path(sp.availableModels[static_cast<size_t>(sp.selectedModelIdx)]).filename().string();
    drawToolbarValue(win, font, "Model", modelName, rowX, rowY, modelW, theme);

    seedField = {rowX + modelW + gap, rowY + metrics.toolbarLabelGap, seedW, metrics.toolbarFieldHeight};
    Helpers::drawText(win, font, "Seed", colors.muted, seedField.left, rowY, type.compact, false);
    Helpers::drawRect(win, seedField, colors.panel,
                      activeToolbarField == ToolbarField::Seed ? colors.goldLt : colors.border, metrics.borderWidth);
    Helpers::drawText(win, font,
                      (activeToolbarField == ToolbarField::Seed ? toolbarInput + "|" :
                       (sp.seedInput.empty() ? "(random)" : sp.seedInput)),
                      (sp.seedInput.empty() && activeToolbarField != ToolbarField::Seed) ? colors.border : colors.goldLt,
                      seedField.left + metrics.spaceSm, seedField.top + 7.f, type.body, false);

    const float stepsX = seedField.left + seedW + gap;
    stepsField = {stepsX, rowY + metrics.toolbarLabelGap, statW, metrics.toolbarFieldHeight};
    drawToolbarValue(win, font,
                     "Steps",
                     activeToolbarField == ToolbarField::Steps
                        ? toolbarInput + "|"
                        : std::to_string(sp.generationParams.numSteps),
                     stepsX, rowY, statW, theme, activeToolbarField == ToolbarField::Steps);
    btnStepsDown = {stepsX + 2.f, rowY + 18.f, 16.f, 24.f};
    btnStepsUp   = {stepsX + statW - 18.f, rowY + 18.f, 16.f, 24.f};
    drawButton(win, btnStepsDown, "-", colors.panel2, colors.text, false, type.body, font);
    drawButton(win, btnStepsUp, "+", colors.panel2, colors.text, false, type.body, font);

    const float cfgX = stepsX + statW + gap;
    char cfgBuf[16];
    std::snprintf(cfgBuf, sizeof(cfgBuf), "%.1f", sp.generationParams.guidanceScale);
    cfgField = {cfgX, rowY + metrics.toolbarLabelGap, statW, metrics.toolbarFieldHeight};
    drawToolbarValue(win, font, "CFG",
                     activeToolbarField == ToolbarField::Cfg ? toolbarInput + "|" : std::string(cfgBuf),
                     cfgX, rowY, statW, theme, activeToolbarField == ToolbarField::Cfg);
    btnCfgDown = {cfgX + 2.f, rowY + 18.f, 16.f, 24.f};
    btnCfgUp   = {cfgX + statW - 18.f, rowY + 18.f, 16.f, 24.f};
    drawButton(win, btnCfgDown, "-", colors.panel2, colors.text, false, type.body, font);
    drawButton(win, btnCfgUp, "+", colors.panel2, colors.text, false, type.body, font);

    const float imagesX = cfgX + statW + gap;
    imagesField = {imagesX, rowY + metrics.toolbarLabelGap, statW, metrics.toolbarFieldHeight};
    drawToolbarValue(win, font, "Images",
                     activeToolbarField == ToolbarField::Images
                        ? toolbarInput + "|"
                        : std::to_string(sp.generationParams.numImages),
                     imagesX, rowY, statW, theme, activeToolbarField == ToolbarField::Images);
    btnImagesDown = {imagesX + 2.f, rowY + 18.f, 16.f, 24.f};
    btnImagesUp   = {imagesX + statW - 18.f, rowY + 18.f, 16.f, 24.f};
    drawButton(win, btnImagesDown, "-", colors.panel2, colors.text, false, type.body, font);
    drawButton(win, btnImagesUp, "+", colors.panel2, colors.text, false, type.body, font);

    btnGenerateAsset = {imagesX + statW + gap, rowY + metrics.toolbarLabelGap, genW, metrics.toolbarFieldHeight};
    drawButton(win, btnGenerateAsset, "Generate", colors.blue, colors.goldLt, false, type.body, font);

    const sf::FloatRect themeBox{sectionX, railY, railW, 150.f};
    Helpers::drawRect(win, themeBox, colors.panel2, colors.border, metrics.borderWidth);
    Helpers::drawText(win, font, "Project Theme", colors.gold, themeBox.left + metrics.spaceMd, themeBox.top + metrics.spaceMd, type.sectionTitle, true);
    Helpers::drawText(win, font, "Shared positive prompt", colors.muted, themeBox.left + metrics.spaceMd, themeBox.top + 34.f, type.compact, false);
    themePositiveArea.setRect({themeBox.left + metrics.spaceMd, themeBox.top + 52.f, themeBox.width - metrics.spaceLg * 2.f, 42.f});
    themePositiveArea.render(win, font);

    Helpers::drawText(win, font, "Shared negative prompt", colors.muted, themeBox.left + metrics.spaceMd, themeBox.top + 100.f, type.compact, false);
    themeNegativeArea.setRect({themeBox.left + metrics.spaceMd, themeBox.top + 118.f, themeBox.width - 140.f, 22.f});
    themeNegativeArea.setTextColor(colors.muted);
    themeNegativeArea.render(win, font);

    btnSaveTheme = {themeBox.left + themeBox.width - 122.f, themeBox.top + 114.f, 112.f, 26.f};
    drawButton(win, btnSaveTheme, themeDirty ? "Save Theme *" : "Save Theme",
               colors.panel, themeDirty ? colors.goldLt : colors.text, false, type.body, font);

    const float assetWorkspaceY = themeBox.top + themeBox.height + metrics.spaceLg;
    const float assetWorkspaceH = 240.f;
    const float listW = 134.f;
    const float detailX = sectionX + listW + metrics.spaceXl;
    const float detailW = railW - listW - metrics.spaceXl;

    Helpers::drawRect(win, {sectionX, assetWorkspaceY, railW, assetWorkspaceH}, colors.panel2, colors.border, metrics.borderWidth);
    Helpers::drawText(win, font, "Asset Types", colors.gold, sectionX + metrics.spaceMd, assetWorkspaceY + metrics.spaceMd, type.sectionTitle, true);

    assetTypeRows.clear();
    const float listAreaTop = assetWorkspaceY + 36.f;
    const float listAreaBottom = assetWorkspaceY + assetWorkspaceH - 42.f;
    const float rowStep = metrics.assetRowStep;
    const int visibleRows = std::max(1, static_cast<int>((listAreaBottom - listAreaTop) / rowStep));
    const int maxScroll = std::max(0, static_cast<int>(proj->assetTypes.size()) - visibleRows);
    assetListScroll = std::clamp(assetListScroll, 0, maxScroll);
    assetListViewport = {sectionX + metrics.spaceXs, listAreaTop, listW - metrics.spaceSm, listAreaBottom - listAreaTop};
    float atY = listAreaTop;
    const int startIdx = assetListScroll;
    const int endIdx = std::min(static_cast<int>(proj->assetTypes.size()), startIdx + visibleRows);
    for (int idx = startIdx; idx < endIdx; ++idx) {
        const auto& at = proj->assetTypes[static_cast<size_t>(idx)];
        const bool selected = at.id == selectedAssetTypeId;
        const sf::FloatRect rowRect{sectionX + metrics.spaceXs, atY, listW - metrics.spaceSm, metrics.assetRowHeight};
        Helpers::drawRect(win, rowRect, selected ? colors.blue : colors.panel,
                          selected ? colors.borderHi : colors.border, metrics.borderWidth);
        Helpers::drawText(win, font, at.name, selected ? colors.goldLt : colors.text,
                          rowRect.left + metrics.spaceSm, rowRect.top + 10.f, type.compact, false);
        const sf::FloatRect delBtn{rowRect.left + rowRect.width - 24.f, rowRect.top + 5.f, 18.f, 24.f};
        drawButton(win, delBtn, "x", colors.panel2, colors.red, false, type.compact, font);
        assetTypeRows.push_back({proj->id, at.id, rowRect, delBtn});
        atY += rowStep;
    }
    if (maxScroll > 0) {
        Helpers::drawTextR(win, font,
                           std::to_string(startIdx + 1) + "-" + std::to_string(endIdx) + " / "
                               + std::to_string(proj->assetTypes.size()),
                           colors.border, sectionX + listW - metrics.spaceMd, assetWorkspaceY + assetWorkspaceH - 32.f, type.helper);
    }

    if (newAssetTypeInputActive) {
        const sf::FloatRect inputRect{sectionX + metrics.spaceXs, assetWorkspaceY + assetWorkspaceH - 38.f, listW - metrics.spaceSm, metrics.assetRowHeight};
        Helpers::drawRect(win, inputRect, colors.panel, colors.borderHi, metrics.borderWidth);
        Helpers::drawText(win, font, newAssetTypeName + "|", colors.text,
                          inputRect.left + metrics.spaceSm, inputRect.top + 10.f, type.compact, false);
        btnAddAssetType = inputRect;
    } else {
        btnAddAssetType = {sectionX + metrics.spaceXs, assetWorkspaceY + assetWorkspaceH - 38.f, listW - metrics.spaceSm, metrics.assetRowHeight};
        drawButton(win, btnAddAssetType, "+ Asset", colors.panel, colors.gold, false, type.compact, font);
    }

    Helpers::drawRect(win, {detailX, assetWorkspaceY + 4.f, detailW, assetWorkspaceH - 8.f}, colors.panel, colors.border, metrics.borderWidth);
    if (selectedAssetTypeId.empty()) {
        Helpers::drawTextC(win, font, "Select an asset type", colors.muted,
                           detailX + detailW / 2.f, assetWorkspaceY + assetWorkspaceH / 2.f - 10.f, 14, false);
        Helpers::drawTextC(win, font, "This prompt is the subject-specific layer added on top of the project theme.",
                           colors.border, detailX + detailW / 2.f, assetWorkspaceY + assetWorkspaceH / 2.f + 16.f, type.compact, false);
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

        Helpers::drawText(win, font, selectedAsset->name, colors.goldLt, detailX + metrics.spaceMd, assetWorkspaceY + 14.f, type.subsectionTitle, true);
        Helpers::drawText(win, font, "Asset prompt", colors.muted, detailX + metrics.spaceMd, assetWorkspaceY + 36.f, type.compact, false);
        assetPositiveArea.setRect({detailX + metrics.spaceMd, assetWorkspaceY + 52.f, detailW - metrics.spaceLg * 2.f, 58.f});
        assetPositiveArea.render(win, font);

        Helpers::drawText(win, font, "Asset negative prompt", colors.muted, detailX + metrics.spaceMd, assetWorkspaceY + 116.f, type.compact, false);
        assetNegativeArea.setRect({detailX + metrics.spaceMd, assetWorkspaceY + 132.f, detailW - metrics.spaceLg * 2.f, 38.f});
        assetNegativeArea.setTextColor(colors.muted);
        assetNegativeArea.render(win, font);

        btnSaveAsset = {detailX + metrics.spaceMd, assetWorkspaceY + 186.f, 112.f, 26.f};
        drawButton(win, btnSaveAsset, assetDirty ? "Save Asset *" : "Save Asset",
                   colors.panel, assetDirty ? colors.goldLt : colors.text, false, type.body, font);
        Helpers::drawText(win, font, "Generator below uses this asset type within the current project theme.",
                          colors.blueLt, detailX + metrics.spaceMd, assetWorkspaceY + 220.f, type.helper, false);
    }

    generatorView.resultPanel.showImproveButton = false;
    generatorView.resultPanel.showTabs = false;

    Helpers::drawRect(win, {resultX, railY, resultW, railH}, colors.panel2, colors.border, metrics.borderWidth);
    Helpers::drawText(win, font, "Results", colors.gold, resultX + metrics.spaceMd, railY + metrics.spaceMd, type.sectionTitle, true);
    Helpers::drawText(win, font, "Preview and gallery stay visible while you tweak theme, asset type, and run settings.",
                      colors.muted, resultX + 72.f, railY + 11.f, type.compact, false);
    generatorView.resultPanel.setRect({resultX + metrics.panelInset, railY + 28.f,
                                       resultW - metrics.panelInset * 2.f, railH - 29.f});
    generatorView.resultPanel.render(win, font, generatorView.settingsPanel.generationParams.numSteps);

    if (generatorView.showSettings)
        generatorView.settingsModal.render(win, font);
}
