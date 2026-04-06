#include "app.hpp"

#include "config/AppConfig.hpp"
#include "enum/constants.hpp"
#include "managers/Logger.hpp"
#include "ui/Logo.hpp"

App::App()
    : config(AppConfig::load("config.json"))
    , win(sf::VideoMode(WIN_W, WIN_H), "Image generator", sf::Style::Close)
    , imageGenController(config)
{
    Logger::info("app constructor");
    win.setFramerateLimit(60);

    const sf::Image icon = Logo::makeIconImage(128);
    win.setIcon(128, 128, icon.getPixelsPtr());
}

void App::run() {
    Logger::info("Application run started");
    while (win.isOpen()) {
        sf::Event e;
        win.clear(Col::Bg);
        if (screen == AppScreen::ImageGenerator) {
            imageGenController.update(imageGenScreen);
            while (win.pollEvent(e))
                imageGenController.handleEvent(e, win, imageGenScreen, screen);
            imageGenScreen.render(win);
        } else {
            while (win.pollEvent(e))
                menuController.handleEvent(e, win, menuScreen, screen);
            menuScreen.render(win);
        }
        win.display();
    }
}
