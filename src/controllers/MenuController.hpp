#pragma once
#include <optional>
#include <cstdint>
#include <SFML/Graphics.hpp>

#include "../entities/Game.hpp"
#include "../presenters/MenuPresenter.hpp"
#include "../views/MenuView.hpp"

enum class AppScreen : std::uint8_t { MENU, Playing, ImageGenerator };

class MenuController {
public:
    void handleEvent(const sf::Event& event, sf::RenderWindow& win,
                     std::optional<Game>& game, MenuView& screen, AppScreen& appScreen);
private:
    void handleClick(sf::Vector2f pos, std::optional<Game>& game, MenuView& screen, AppScreen& appScreen);
    MenuPresenter presenter;
};
