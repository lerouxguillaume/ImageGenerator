#pragma once
#include <string>
#include <vector>
#include <SFML/Graphics.hpp>
#include "Screen.hpp"

class MenuView : public Screen {
public:
    sf::FloatRect btnImageGen;
    sf::FloatRect btnImageEdit;
    sf::FloatRect btnProjects;
    sf::FloatRect btnImportModel;

    void render(sf::RenderWindow& win) override;

private:
};
