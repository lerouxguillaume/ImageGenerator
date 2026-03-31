#pragma once
#include <SFML/Graphics.hpp>
#include "../presenters/ImageGeneratorPresenter.hpp"
#include "../views/ImageGeneratorView.hpp"
#include "MenuController.hpp"

class ImageGeneratorController {
public:
    void handleEvent(const sf::Event& event, sf::RenderWindow& win,
                     ImageGeneratorView& screen, AppScreen& appScreen);
    void update(ImageGeneratorView& screen);

private:
    void handleClick(sf::Vector2f pos, sf::RenderWindow& win,
                     ImageGeneratorView& view, AppScreen& appScreen);
    ImageGeneratorPresenter presenter;
};
