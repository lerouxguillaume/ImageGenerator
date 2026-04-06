#pragma once
#include <SFML/Graphics.hpp>

#include "config/AppConfig.hpp"
#include "views/MenuView.hpp"
#include "views/ImageGeneratorView.hpp"
#include "controllers/MenuController.hpp"
#include "controllers/ImageGeneratorController.hpp"

class App {
    // config must be declared before imageGenController so it is initialised first.
    AppConfig             config;
    sf::RenderWindow      win;

    AppScreen             screen = AppScreen::MENU;
    MenuView              menuScreen;
    ImageGeneratorView    imageGenScreen;
    MenuController        menuController;
    ImageGeneratorController imageGenController;

public:
    App();
    void run();
};
