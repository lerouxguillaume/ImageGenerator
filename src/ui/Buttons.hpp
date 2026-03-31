#pragma once
#include <string>
#include <SFML/Graphics/Font.hpp>
#include <SFML/Graphics/RenderTarget.hpp>

void drawButton(sf::RenderTarget& rt, sf::FloatRect r, const std::string& label,
                sf::Color bg, sf::Color tc, bool disabled, unsigned sz, sf::Font& font);
