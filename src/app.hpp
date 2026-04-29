#pragma once
#include <SFML/Graphics.hpp>

#include "config/AppConfig.hpp"
#include "views/MenuView.hpp"
#include "views/ImageGeneratorView.hpp"
#include "views/ProjectView.hpp"
#include "controllers/MenuController.hpp"
#include "controllers/ImageGeneratorController.hpp"
#include "controllers/ProjectController.hpp"

class App {
    // config must be declared before controllers so it is initialised first.
    AppConfig             config;
    sf::RenderWindow      win;

    AppScreen             screen = AppScreen::MENU;
    AppScreen             imageEditBackScreen = AppScreen::MENU;
    MenuView              menuScreen;
    ImageGeneratorView    imageGenScreen{WorkflowMode::Generate};
    ImageGeneratorView    imageEditScreen{WorkflowMode::Edit};
    ProjectView           projectScreen;
    MenuController        menuController;
    ImageGeneratorController imageGenController;
    ImageGeneratorController imageEditController;
    ProjectController     projectController;

public:
    App();
    void run();
};
