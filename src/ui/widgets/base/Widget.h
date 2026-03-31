#pragma once
#include <SFML/Graphics.hpp>

class Widget {
public:
    virtual ~Widget() = default;
    virtual void render(sf::RenderWindow& win, sf::Font& font) = 0;
    virtual void setRect(const sf::FloatRect& rect) = 0;
};
