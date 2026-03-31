#include "app.hpp"

#include "enum/constants.hpp"
#include "managers/Logger.hpp"

App::App() : win(sf::VideoMode(WIN_W, WIN_H), "Guild Master", sf::Style::Close) {
    Logger::info("app constructor");
    win.setFramerateLimit(60);

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
        }
        win.display();
    }
}
