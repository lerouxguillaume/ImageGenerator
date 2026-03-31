#pragma once
#include <optional>
#include <SFML/Graphics.hpp>

class Screen
{
public:
    Screen();
    virtual void render(sf::RenderWindow& win) = 0;

    sf::Vector2f localPosMid(sf::Vector2f screenPos) const;
    bool         inMid(sf::Vector2f screenPos) const;

    virtual ~Screen() = default;

protected:
    sf::Font      font;
    sf::FloatRect midRect;
};
