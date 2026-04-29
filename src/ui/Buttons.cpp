#include "Buttons.hpp"

#include <SFML/Graphics/Text.hpp>

#include "Helpers.hpp"
#include "Theme.h"

void drawButton(sf::RenderTarget& rt, sf::FloatRect r, const std::string& label,
                sf::Color bg, sf::Color tc, bool disabled, unsigned sz, sf::Font& font) {
    const auto& theme = Theme::instance();
    const auto& colors = theme.colors();
    const auto& metrics = theme.metrics();
    const sf::Color actual = disabled ? colors.buttonDisabledBg : bg;
    Helpers::drawRect(rt,
                      {r.left, r.top + metrics.shadowOffset, r.width, r.height},
                      colors.shadow,
                      sf::Color::Transparent,
                      0.f);
    Helpers::drawRect(rt, r, actual, colors.border, metrics.borderWidth);
    Helpers::drawRect(rt,
                      {r.left + 1.f, r.top + 1.f, r.width - 2.f, 1.f},
                      disabled ? colors.border : colors.borderHi,
                      sf::Color::Transparent,
                      0.f);
    sf::Text t;
    t.setFont(font); t.setString(label); t.setCharacterSize(sz);
    t.setFillColor(disabled ? colors.muted : tc);
    t.setStyle(sf::Text::Bold);
    t.setPosition(r.left + (r.width  - t.getLocalBounds().width)  / 2.f,
                  r.top  + (r.height - t.getLocalBounds().height) / 2.f - 2.f);
    rt.draw(t);
}
