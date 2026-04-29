#include "MenuView.hpp"

#include <filesystem>
#include "../enum/constants.hpp"
#include "../ui/Buttons.hpp"
#include "../ui/Helpers.hpp"
#include "../ui/Logo.hpp"

void MenuView::render(sf::RenderWindow& win) {
    const float cx = win.getSize().x / 2.f;
    const float cy = win.getSize().y / 2.f;

    Logo::draw(win, cx, cy - 220.f, 52.f);
    Helpers::drawTextC(win, font, "Image generator", Col::GoldLt, cx, cy - 120, 48, true);

    const float bw = 220.f, bh = 44.f, gap = 12.f;
    btnImageGen = {cx - bw / 2.f, cy + (bh + gap) * 1.5f, bw, bh};
    btnImageEdit = {cx - bw / 2.f, btnImageGen.top + bh + gap, bw, bh};
    drawButton(win, btnImageGen, "Generate Images", Col::Panel2, Col::Muted, false, 14, font);
    drawButton(win, btnImageEdit, "Edit Image", Col::Panel2, Col::Muted, false, 14, font);

}
