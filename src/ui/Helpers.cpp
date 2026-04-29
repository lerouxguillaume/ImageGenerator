#include <string>
#include <SFML/Graphics/Color.hpp>
#include <sstream>

#include "Helpers.hpp"
#include "Theme.h"
#include "../enum/constants.hpp"

namespace Helpers {
// ─── Drawing helpers ──────────────────────────────────────────────────────────
void drawRect(sf::RenderTarget& rt, sf::FloatRect r, sf::Color fill, sf::Color border, float bw) {
    sf::RectangleShape s({r.width, r.height});
    s.setPosition(r.left, r.top);
    s.setFillColor(fill);
    if (bw > 0.f) { s.setOutlineColor(border); s.setOutlineThickness(-bw); }
    rt.draw(s);
}

void drawDividers(sf::RenderWindow& win) {
    const auto& theme = Theme::instance();
    for (const float x : {static_cast<float>(LEFT_W)}) {
        sf::RectangleShape line({1.f, static_cast<float>(BODY_H)});
        line.setPosition(x, HEADER_H);
        line.setFillColor(theme.colors().border);
        win.draw(line);
    }
}

void drawBar(sf::RenderTarget& rt, float x, float y, float w, float h, float pct,
             sf::Color fg, sf::Color bg) {
    drawRect(rt, {x, y, w, h}, bg);
    drawRect(rt, {x, y, std::max(2.f, w * std::min(1.f, pct)), h}, fg);
}

void drawText(sf::RenderTarget& rt, sf::Font& font, const std::string& str,
              sf::Color color, float x, float y, unsigned size, bool bold) {
    sf::Text t;
    t.setFont(font);
    t.setString(str);
    t.setCharacterSize(size);
    t.setFillColor(color);
    t.setStyle(bold ? sf::Text::Bold : sf::Text::Regular);
    t.setPosition(x, y);
    rt.draw(t);
}

void drawTextC(sf::RenderTarget& rt, sf::Font& font, const std::string& str,
               sf::Color color, float cx, float y, unsigned size, bool bold) {
    sf::Text t;
    t.setFont(font); t.setString(str); t.setCharacterSize(size);
    t.setFillColor(color);
    t.setStyle(bold ? sf::Text::Bold : sf::Text::Regular);
    t.setPosition(cx - t.getLocalBounds().width / 2.f, y);
    rt.draw(t);
}

void drawTextR(sf::RenderTarget& rt, sf::Font& font, const std::string& str,
               sf::Color color, float rx, float y, unsigned size) {
    sf::Text t;
    t.setFont(font); t.setString(str); t.setCharacterSize(size);
    t.setFillColor(color);
    t.setPosition(rx - t.getLocalBounds().width, y);
    rt.draw(t);
}

float drawWrapped(sf::RenderTarget& rt, sf::Font& font, const std::string& str,
                 sf::Color color, float x, float y, float maxW, unsigned size) {
    sf::Text t;
    t.setFont(font); t.setCharacterSize(size); t.setFillColor(color);
    std::istringstream ss(str);
    std::string word, line;
    float lineH = size + 4.f, totalH = 0;
    auto flush = [&]() {
        if (line.empty()) return;
        t.setString(line);
        t.setPosition(x, y + totalH);
        rt.draw(t);
        totalH += lineH;
        line.clear();
    };
    while (ss >> word) {
        std::string test = line.empty() ? word : line + " " + word;
        t.setString(test);
        if (t.getLocalBounds().width > maxW && !line.empty()) { flush(); line = word; }
        else line = test;
    }
    flush();
    return totalH;
}

sf::Color diffColor(int d) {
    const auto& colors = Theme::instance().colors();
    switch (d) {
        case 1:  return colors.greenLt;
        case 2:  return colors.gold;
        case 3:  return colors.redLt;
        default: return colors.purpleLt;
    }
}
} // namespace Helpers
