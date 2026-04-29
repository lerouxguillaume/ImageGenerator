#include "ProjectView.hpp"
#include "../enum/constants.hpp"
#include "../ui/Buttons.hpp"
#include "../ui/Helpers.hpp"

static constexpr float VPAD  = 18.f;
static constexpr float ROW_H = 40.f;
static constexpr float GAP   =  8.f;

void ProjectView::render(sf::RenderWindow& win) {
    const float W = static_cast<float>(win.getSize().x);
    const float H = static_cast<float>(win.getSize().y);

    // ── Header bar ────────────────────────────────────────────────────────────
    Helpers::drawRect(win, {0.f, 0.f, W, static_cast<float>(HEADER_H)}, Col::Panel, Col::Border, 1.f);
    Helpers::drawText(win, font, "Projects", Col::GoldLt, VPAD, 18.f, 20, true);

    btnBack = {W - 120.f - VPAD, (HEADER_H - 30.f) / 2.f, 120.f, 30.f};
    drawButton(win, btnBack, "< Back", Col::Panel2, Col::Muted, false, 13, font);

    // ── Two-column layout ─────────────────────────────────────────────────────
    const float bodyY   = static_cast<float>(HEADER_H) + VPAD;
    const float bodyH   = H - bodyY - VPAD;
    const float leftW   = 260.f;
    const float rightX  = leftW + VPAD * 2.f;
    const float rightW  = W - rightX - VPAD;

    // ── Left panel: project list ──────────────────────────────────────────────
    Helpers::drawRect(win, {VPAD, bodyY, leftW, bodyH}, Col::Panel, Col::Border, 1.f);

    Helpers::drawText(win, font, "Projects", Col::Gold, VPAD + 10.f, bodyY + 10.f, 13, true);

    projectRows.clear();
    float rowY = bodyY + 36.f;
    for (const auto& proj : projects) {
        const bool selected = proj.id == selectedProjectId;
        const sf::Color bg  = selected ? Col::Blue : Col::Panel2;
        const sf::FloatRect rowRect{VPAD + 4.f, rowY, leftW - 8.f, ROW_H};

        Helpers::drawRect(win, rowRect, bg, selected ? Col::BorderHi : Col::Border, 1.f);
        Helpers::drawText(win, font, proj.name, selected ? Col::GoldLt : Col::Text,
                          rowRect.left + 8.f, rowRect.top + 12.f, 13, false);

        const sf::FloatRect delBtn{rowRect.left + rowRect.width - 28.f,
                                   rowRect.top + 6.f, 22.f, 28.f};
        drawButton(win, delBtn, "x", Col::Panel, Col::Red, false, 11, font);

        projectRows.push_back({proj.id, rowRect, delBtn});
        rowY += ROW_H + GAP;
    }

    // "New project" row — input or button
    if (newProjectInputActive) {
        const sf::FloatRect inputRect{VPAD + 4.f, rowY, leftW - 8.f, ROW_H};
        Helpers::drawRect(win, inputRect, Col::Panel2, Col::BorderHi, 1.f);
        const std::string display = newProjectName + "|";
        Helpers::drawText(win, font, display, Col::Text, inputRect.left + 8.f,
                          inputRect.top + 12.f, 13, false);
        btnNewProject = inputRect;
    } else {
        btnNewProject = {VPAD + 4.f, rowY, leftW - 8.f, ROW_H};
        drawButton(win, btnNewProject, "+ New Project", Col::Panel2, Col::Gold, false, 13, font);
    }

    // ── Right panel: selected project detail ──────────────────────────────────
    if (selectedProjectId.empty()) {
        Helpers::drawRect(win, {rightX, bodyY, rightW, bodyH}, Col::Panel, Col::Border, 1.f);
        Helpers::drawTextC(win, font, "Select a project", Col::Muted,
                           rightX + rightW / 2.f, bodyY + bodyH / 2.f, 14, false);
        assetTypeRows.clear();
        btnAddAssetType = {};
        return;
    }

    const Project* proj = nullptr;
    for (const auto& p : projects)
        if (p.id == selectedProjectId) { proj = &p; break; }
    if (!proj) return;

    Helpers::drawRect(win, {rightX, bodyY, rightW, bodyH}, Col::Panel, Col::Border, 1.f);

    // Project title
    Helpers::drawText(win, font, proj->name, Col::GoldLt, rightX + 12.f, bodyY + 12.f, 18, true);

    // Asset type list
    Helpers::drawText(win, font, "Asset Types", Col::Gold,
                      rightX + 12.f, bodyY + 48.f, 13, true);

    assetTypeRows.clear();
    float atY = bodyY + 72.f;
    for (const auto& at : proj->assetTypes) {
        const sf::FloatRect atRect{rightX + 4.f, atY, rightW - 8.f, ROW_H};
        Helpers::drawRect(win, atRect, Col::Panel2, Col::Border, 1.f);
        Helpers::drawText(win, font, at.name, Col::Text,
                          atRect.left + 10.f, atRect.top + 12.f, 13, false);

        const sf::FloatRect genBtn{atRect.left + atRect.width - 130.f,
                                   atRect.top + 6.f, 110.f, 28.f};
        const sf::FloatRect delBtn{atRect.left + atRect.width - 148.f,
                                   atRect.top + 6.f, 22.f, 28.f};
        drawButton(win, genBtn, "Generate >", Col::Blue, Col::GoldLt, false, 12, font);
        drawButton(win, delBtn, "x", Col::Panel, Col::Red, false, 11, font);

        assetTypeRows.push_back({proj->id, at.id, atRect, genBtn, delBtn});
        atY += ROW_H + GAP;
    }

    // "Add asset type" row
    if (newAssetTypeInputActive) {
        const sf::FloatRect inputRect{rightX + 4.f, atY, rightW - 8.f, ROW_H};
        Helpers::drawRect(win, inputRect, Col::Panel2, Col::BorderHi, 1.f);
        const std::string display = newAssetTypeName + "|";
        Helpers::drawText(win, font, display, Col::Text,
                          inputRect.left + 10.f, inputRect.top + 12.f, 13, false);
        btnAddAssetType = inputRect;
    } else {
        btnAddAssetType = {rightX + 4.f, atY, rightW - 8.f, ROW_H};
        drawButton(win, btnAddAssetType, "+ Add Asset Type", Col::Panel2, Col::Gold,
                   false, 13, font);
    }
}
