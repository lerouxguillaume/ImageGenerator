#include "MenuController.hpp"

void MenuController::handleEvent(const sf::Event& e, sf::RenderWindow& win,
                                 MenuView& screen, AppScreen& appScreen) {
    if (e.type == sf::Event::Closed) win.close();
    if (e.type == sf::Event::KeyPressed && e.key.code == sf::Keyboard::Escape) {
        win.close();
    }
    if (e.type == sf::Event::MouseButtonPressed && e.mouseButton.button == sf::Mouse::Left)
        handleClick(win.mapPixelToCoords({e.mouseButton.x, e.mouseButton.y}), screen, appScreen);
}

void MenuController::handleClick(sf::Vector2f pos, MenuView& screen, AppScreen& appScreen) {
    if (screen.btnImageGen.contains(pos)) {
        appScreen = AppScreen::ImageGenerator;
    }
}
