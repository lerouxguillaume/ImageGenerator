#include "MenuView.hpp"

#include <filesystem>
#include "../enum/constants.hpp"
#include "../ui/Buttons.hpp"
#include "../ui/Helpers.hpp"
#include "../ui/Logo.hpp"
#include "../ui/Theme.h"

void MenuView::render(sf::RenderWindow& win) {
    const auto& theme = Theme::instance();
    const auto& colors = theme.colors();
    const auto& type = theme.typography();
    const float cx = win.getSize().x / 2.f;
    const float cy = win.getSize().y / 2.f;
    const float winW = static_cast<float>(win.getSize().x);
    const float winH = static_cast<float>(win.getSize().y);

    win.clear(colors.bg);

    Helpers::drawRect(win, {48.f, 42.f, winW - 96.f, winH - 84.f}, colors.surfaceInset, colors.border, 1.f);
    Helpers::drawRect(win, {72.f, 70.f, winW - 144.f, winH - 140.f}, colors.panel, colors.border, 1.f);

    Logo::draw(win, cx, cy - 196.f, 52.f);
    Helpers::drawTextC(win, font, "Image Generator", colors.text, cx, cy - 112.f, 40, true);
    Helpers::drawTextC(win, font, "Generate themed assets, iterate edits, and manage packs in one place.",
                       colors.muted, cx, cy - 64.f, type.sectionTitle, false);

    const float cardW = 348.f;
    const float cardH = 190.f;
    const float cardX = cx - cardW / 2.f;
    const float cardY = cy - 8.f;
    Helpers::drawRect(win, {cardX, cardY, cardW, cardH}, colors.panel2, colors.border, 1.f);
    Helpers::drawText(win, font, "Workspace", colors.gold, cardX + 18.f, cardY + 16.f, type.sectionTitle, true);

    const float bw = cardW - 40.f;
    const float bh = 42.f;
    const float gap = 12.f;
    btnImageGen  = {cardX + 20.f, cardY + 48.f,               bw, bh};
    btnImageEdit = {cardX + 18.f, btnImageGen.top  + bh + gap,    bw, bh};
    btnProjects  = {cardX + 18.f, btnImageEdit.top + bh + gap,    bw, bh};
    btnImageEdit = {cardX + 20.f, btnImageGen.top  + bh + gap,    bw, bh};
    btnProjects  = {cardX + 20.f, btnImageEdit.top + bh + gap,    bw, bh};
    drawButton(win, btnImageGen,  "Generate Images", colors.panel, colors.text, false, 13, font);
    drawButton(win, btnImageEdit, "Edit Image",      colors.panel, colors.text, false, 13, font);
    drawButton(win, btnProjects,  "Projects",        colors.blue,  colors.goldLt,  false, 13, font);

}
