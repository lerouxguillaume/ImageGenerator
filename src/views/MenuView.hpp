#pragma once
#include <string>
#include <vector>
#include <future>
#include <SFML/Graphics.hpp>
#include "Screen.hpp"

class MenuView : public Screen {
public:
    sf::FloatRect btnStartGame;
    sf::FloatRect btnLoadGame;
    sf::FloatRect btnImageGen;

    // New game modal
    bool          showModal  = false;
    std::string   guildName  = "My Guild";
    sf::FloatRect modalBox;
    sf::FloatRect modalInput;
    sf::FloatRect btnConfirm;
    sf::FloatRect btnCancel;

    // Load game modal
    bool                               showLoadModal = false;
    std::vector<sf::FloatRect>         saveSlotBtns;
    std::vector<sf::FloatRect>         saveDeleteBtns;
    sf::FloatRect                      btnLoadCancel;

    // Delete confirmation modal
    bool          showDeleteConfirm = false;
    int           pendingDeleteIdx  = -1;
    sf::FloatRect btnDeleteConfirm;
    sf::FloatRect btnDeleteCancel;

    void render(sf::RenderWindow& win) override;

private:
    void drawModal(sf::RenderWindow& win);
    void drawLoadModal(sf::RenderWindow& win);
    void drawDeleteConfirm(sf::RenderWindow& win);
};
