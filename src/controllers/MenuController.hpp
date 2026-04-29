#pragma once
#include <optional>
#include <cstdint>
#include <SFML/Graphics.hpp>

#include "../presenters/MenuPresenter.hpp"
#include "../views/MenuView.hpp"

enum class AppScreen : std::uint8_t { MENU, Playing, ImageGenerator, ImageEditor };

class MenuController {
public:
    void handleEvent(const sf::Event& event, sf::RenderWindow& win,
                     MenuView& screen, AppScreen& appScreen);
private:
    void handleClick(sf::Vector2f pos, MenuView& screen, AppScreen& appScreen);
    MenuPresenter presenter;
};
