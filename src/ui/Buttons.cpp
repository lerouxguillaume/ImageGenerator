#include "Buttons.hpp"

#include <SFML/Graphics/Text.hpp>

#include "Helpers.hpp"
#include "../enum/constants.hpp"

void drawButton(sf::RenderTarget& rt, sf::FloatRect r, const std::string& label,
                sf::Color bg, sf::Color tc, bool disabled, unsigned sz, sf::Font& font) {
    const sf::Color actual = disabled ? sf::Color{60,50,35} : bg;
    Helpers::drawRect(rt, r, actual, Col::Border, 1.f);
    sf::Text t;
    t.setFont(font); t.setString(label); t.setCharacterSize(sz);
    t.setFillColor(disabled ? Col::Muted : tc);
    t.setStyle(sf::Text::Bold);
    t.setPosition(r.left + (r.width  - t.getLocalBounds().width)  / 2.f,
                  r.top  + (r.height - t.getLocalBounds().height) / 2.f - 2.f);
    rt.draw(t);
}
