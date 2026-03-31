#include "MenuController.hpp"
#include "../services/GameService.hpp"
#include "../managers/SaveManager.hpp"

void MenuController::handleEvent(const sf::Event& e, sf::RenderWindow& win,
                                 std::optional<Game>& game, MenuView& screen, AppScreen& appScreen) {
    if (e.type == sf::Event::Closed) win.close();
    if (e.type == sf::Event::KeyPressed && e.key.code == sf::Keyboard::Escape) {
        if (screen.showDeleteConfirm) { presenter.closeDeleteConfirm(screen); return; }
        if (screen.showModal)         { presenter.closeNewGameModal(screen); return; }
        if (screen.showLoadModal)     { presenter.closeLoadModal(screen); return; }
        win.close();
    }
    if (e.type == sf::Event::MouseButtonPressed && e.mouseButton.button == sf::Mouse::Left)
        handleClick(win.mapPixelToCoords({e.mouseButton.x, e.mouseButton.y}), game, screen, appScreen);
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

void MenuController::handleClick(sf::Vector2f pos, std::optional<Game>& game, MenuView& screen, AppScreen& appScreen) {
    if (screen.showLoadModal) {
        if (screen.showDeleteConfirm) {
            if (screen.btnDeleteConfirm.contains(pos)) {
                SaveManager saveManager;
                saveManager.deleteSave(screen.saves[screen.pendingDeleteIdx].slot);
                screen.saves = saveManager.listSaves();
                presenter.closeDeleteConfirm(screen);
            } else if (screen.btnDeleteCancel.contains(pos)) {
                presenter.closeDeleteConfirm(screen);
            }
            return;
        }

        for (int i = 0; i < (int)screen.saveSlotBtns.size(); ++i) {
            if (screen.saveSlotBtns[i].contains(pos)) {
                SaveManager saveManager;
                saveManager.loadSave(screen.saves[i].slot);
                screen.showLoadModal = false;
                appScreen = AppScreen::Playing;
                return;
            }
            if (screen.saveDeleteBtns[i].contains(pos)) {
                presenter.openDeleteConfirm(screen, i);
                return;
            }
        }
        if (screen.btnLoadCancel.contains(pos))
            presenter.closeLoadModal(screen);
        return;
    }

    if (screen.showModal) {
        if (screen.btnConfirm.contains(pos)) {
            SaveManager saveManager;
            GameService gameService;
            const std::string name = screen.guildName.empty() ? "My Guild" : screen.guildName;
            saveManager.createSave(name);
            game.emplace(gameService.createNewGame(name.c_str()));
            appScreen = AppScreen::Playing;
        } else if (screen.btnCancel.contains(pos)) {
            presenter.closeNewGameModal(screen);
        }
        return;
    }

    if (screen.btnStartGame.contains(pos)) {
        presenter.openNewGameModal(screen);
    } else if (screen.btnLoadGame.contains(pos)) {
        const SaveManager saveManager;
        screen.saves = saveManager.listSaves();
        presenter.openLoadModal(screen);
    } else if (screen.btnImageGen.contains(pos)) {
        appScreen = AppScreen::ImageGenerator;
    }
}
