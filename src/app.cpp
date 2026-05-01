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
    , imageGenController(config, WorkflowMode::Generate)
    , imageEditController(config, WorkflowMode::Edit)
    , projectController(config)
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
        else if (screen == AppScreen::ImageEditor)
            imageEditController.update(imageEditScreen);
        else if (screen == AppScreen::Projects)
            projectController.update(projectScreen);

        while (win.pollEvent(e)) {
            const AppScreen screenBeforeEvent = screen;
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
            else if (screen == AppScreen::ImageEditor)
                imageEditController.handleEvent(e, win, imageEditScreen, screen);
            else if (screen == AppScreen::Projects)
                projectController.handleEvent(e, win, projectScreen, screen);
            else
                menuController.handleEvent(e, win, menuScreen, screen);

            const std::string editTarget = imageGenController.consumePendingEditNavigation();
            if (!editTarget.empty()) {
                imageEditBackScreen = AppScreen::ImageGenerator;
                imageEditController.setBackScreen(imageEditBackScreen);
                const ResolvedProjectContext projectCtx = imageGenController.getProjectContext();
                if (!projectCtx.empty())
                    imageEditController.activateProjectSession(imageEditScreen, projectCtx);
                else
                    imageEditController.clearProjectContext();
                imageEditController.prepareEditSession(imageEditScreen, editTarget);
                screen = AppScreen::ImageEditor;
            } else if (screenBeforeEvent == AppScreen::MENU && screen == AppScreen::ImageEditor) {
                imageEditBackScreen = AppScreen::MENU;
                imageEditController.setBackScreen(imageEditBackScreen);
                imageEditController.clearProjectContext();
            }

            const ResolvedProjectContext pendingCtx = projectController.consumePendingGeneration();
            if (!pendingCtx.empty()) {
                imageGenController.activateProjectSession(imageGenScreen, pendingCtx);
                imageGenController.setBackScreen(AppScreen::Projects);
                screen = AppScreen::ImageGenerator;
            }

            // Clear project context when navigating away from the generator to anywhere
            // other than the edit screen (which inherits the session).
            if (screenBeforeEvent == AppScreen::ImageGenerator
                && screen != AppScreen::ImageGenerator
                && screen != AppScreen::ImageEditor) {
                imageGenController.clearProjectContext();
            }
        }

        if (screen == AppScreen::ImageGenerator)
            imageGenScreen.render(win);
        else if (screen == AppScreen::ImageEditor)
            imageEditScreen.render(win);
        else if (screen == AppScreen::Projects)
            projectScreen.render(win);
        else {
            menuScreen.render(win);
            menuController.renderOverlay(win);
        }

        win.display();
    }
}
