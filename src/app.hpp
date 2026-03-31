#pragma once
#include <SFML/Graphics.hpp>

#include "views/MenuView.hpp"
#include "views/ImageGeneratorView.hpp"
#include "controllers/MenuController.hpp"
#include "controllers/ImageGeneratorController.hpp"

class App {
    sf::RenderWindow win;

    AppScreen             screen = AppScreen::MENU;
    MenuView              menuScreen;
    ImageGeneratorView    imageGenScreen;
    MenuController        menuController;
    ImageGeneratorController imageGenController;

public:
    App();
    void run();
};
