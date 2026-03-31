#include "MenuController.hpp"

void MenuController::handleEvent(const sf::Event& e, sf::RenderWindow& win,
                                 MenuView& screen, AppScreen& appScreen) {
    if (e.type == sf::Event::Closed) win.close();
    if (e.type == sf::Event::KeyPressed && e.key.code == sf::Keyboard::Escape) {
        if (screen.showDeleteConfirm) { presenter.closeDeleteConfirm(screen); return; }
        if (screen.showModal)         { presenter.closeNewGameModal(screen); return; }
        if (screen.showLoadModal)     { presenter.closeLoadModal(screen); return; }
        win.close();
    }
    if (e.type == sf::Event::MouseButtonPressed && e.mouseButton.button == sf::Mouse::Left)
        handleClick(win.mapPixelToCoords({e.mouseButton.x, e.mouseButton.y}), screen, appScreen);
    if (screen.showModal) {
        if (e.type == sf::Event::KeyPressed && e.key.code == sf::Keyboard::BackSpace) {
            if (!screen.guildName.empty())
                screen.guildName.pop_back();
        }
        if (e.type == sf::Event::TextEntered) {
            const auto c = e.text.unicode;
            if (c >= 32 && c < 127 && screen.guildName.size() < 30)
                screen.guildName += static_cast<char>(c);
        }
    }
}

void MenuController::handleClick(sf::Vector2f pos, MenuView& screen, AppScreen& appScreen) {
    if (screen.btnImageGen.contains(pos)) {
        appScreen = AppScreen::ImageGenerator;
    }
}
