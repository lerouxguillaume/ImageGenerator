#include "MenuView.hpp"

#include <filesystem>
#include "../enum/constants.hpp"
#include "../ui/Buttons.hpp"
#include "../ui/Helpers.hpp"
#include "../ui/Logo.hpp"

void MenuView::render(sf::RenderWindow& win) {
    const float cx = WIN_W / 2.f;
    const float cy = WIN_H / 2.f;

    Logo::draw(win, cx, cy - 220.f, 52.f);
    Helpers::drawTextC(win, font, "Image generator", Col::GoldLt, cx, cy - 120, 48, true);

    const float bw = 200.f, bh = 44.f, gap = 12.f;
    btnImageGen = {cx - bw / 2.f, cy + (bh + gap) * 2.f, bw, bh};
    drawButton(win, btnImageGen, "Image Generator", Col::Panel2, Col::Muted, false, 14, font);

}
