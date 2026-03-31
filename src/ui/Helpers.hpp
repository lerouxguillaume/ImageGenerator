#pragma once
#include <string>
#include <SFML/Graphics/Font.hpp>
#include <SFML/Graphics/RenderTarget.hpp>
#include <SFML/Graphics/RenderWindow.hpp>

#include "../enum/enums.hpp"

// ─── Drawing helpers ──────────────────────────────────────────────────────────
namespace Helpers {
    void drawRect(sf::RenderTarget& rt, sf::FloatRect r, sf::Color fill,
                  sf::Color border = sf::Color::Transparent, float bw = 0.f);
    void drawDividers(sf::RenderWindow& win);
    void drawBar(sf::RenderTarget& rt, float x, float y, float w, float h, float pct,
                 sf::Color fg, sf::Color bg = {40,30,15,255});
    void drawText(sf::RenderTarget& rt, sf::Font& font, const std::string& str,
                  sf::Color color, float x, float y, unsigned size = 14, bool bold = false);
    void drawTextC(sf::RenderTarget& rt, sf::Font& font, const std::string& str,
                   sf::Color color, float cx, float y, unsigned size = 14, bool bold = false);
    void drawTextR(sf::RenderTarget& rt, sf::Font& font, const std::string& str,
                   sf::Color color, float rx, float y, unsigned size = 14);
    float drawWrapped(sf::RenderTarget& rt, sf::Font& font, const std::string& str,
                      sf::Color color, float x, float y, float maxW, unsigned size = 12);
    sf::Color   diffColor(int d);
    std::string diffStars(int d);
    std::string statusStr(Status s);
    sf::Color   statusColor(Status s);
}
