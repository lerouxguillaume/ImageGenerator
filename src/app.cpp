#include "app.hpp"

#include "config/AppConfig.hpp"
#include "ui/Theme.h"
#include "managers/Logger.hpp"
#include "ui/Logo.hpp"

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX               // keep std::max/std::min (used below) usable
#  include <windows.h>
#endif

static constexpr unsigned MIN_WIN_W = 700u;
static constexpr unsigned MIN_WIN_H = 550u;

// Open maximized: fills the screen but keeps the title bar, window controls and
// taskbar (unlike sf::Style::Fullscreen). SFML 2.x has no portable maximize, so
// use the native call on Windows and fall back to sizing to the desktop mode.
// The initial 1280x800 VideoMode remains the "restore" size when un-maximized.
static void maximizeWindow(sf::RenderWindow& win) {
#ifdef _WIN32
    ShowWindow(win.getSystemHandle(), SW_MAXIMIZE);
#else
    const sf::VideoMode dm = sf::VideoMode::getDesktopMode();
    win.setPosition({0, 0});
    win.setSize({dm.width, dm.height});
#endif
}

App::App()
    : config(AppConfig::load("config.json"))
    , win(sf::VideoMode(Theme::instance().metrics().windowWidth,
                        Theme::instance().metrics().windowHeight),
          "Image generator", sf::Style::Close | sf::Style::Resize)
    , imageGenController(config)
{
    Logger::info("app constructor");
    win.setFramerateLimit(60);

    const sf::Image icon = Logo::makeIconImage(128);
    win.setIcon(128, 128, icon.getPixelsPtr());

    maximizeWindow(win);
}

void App::run() {
    Logger::info("Application run started");
    while (win.isOpen()) {
        sf::Event e;
        win.clear(Theme::instance().colors().bg);

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
        else {
            menuScreen.render(win);
            menuController.renderOverlay(win);
        }

        win.display();
    }
}
