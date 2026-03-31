#pragma once
#include <SFML/Graphics.hpp>
#include "WidgetEvent.h"

class WidgetController {
public:
    virtual ~WidgetController() = default;
    virtual WidgetEvent handleEvent(const sf::Event& e, const sf::RenderWindow& win) = 0;
    virtual void update() { }
};
