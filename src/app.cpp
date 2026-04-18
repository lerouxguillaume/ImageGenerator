#include "app.hpp"

#include "config/AppConfig.hpp"
#include "enum/constants.hpp"
#include "managers/Logger.hpp"
#include "ui/Logo.hpp"

static constexpr unsigned MIN_WIN_W = 700u;
static constexpr unsigned MIN_WIN_H = 550u;

App::App()
    : config(AppConfig::load("config.json"))
    , win(sf::VideoMode(WIN_W, WIN_H), "Image generator", sf::Style::Close | sf::Style::Resize)
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

        if (screen == AppScreen::ImageGenerator)
            imageGenController.update(imageGenScreen);

        while (win.pollEvent(e)) {
            if (e.type == sf::Event::Resized) {
                const unsigned w = std::max(e.size.width,  MIN_WIN_W);
                const unsigned h = std::max(e.size.height, MIN_WIN_H);
                if (w != e.size.width || h != e.size.height)
                    win.setSize({w, h});
                win.setView(sf::View(sf::FloatRect(0.f, 0.f,
                    static_cast<float>(win.getSize().x),
                    static_cast<float>(win.getSize().y))));
                continue;
            }
            if (screen == AppScreen::ImageGenerator)
                imageGenController.handleEvent(e, win, imageGenScreen, screen);
            else
                menuController.handleEvent(e, win, menuScreen, screen);
        }

        if (screen == AppScreen::ImageGenerator)
            imageGenScreen.render(win);
        else
            menuScreen.render(win);

        win.display();
    }
}
