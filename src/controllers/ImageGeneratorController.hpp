#pragma once
#include <SFML/Graphics.hpp>
#include "../presenters/ImageGeneratorPresenter.hpp"
#include "../views/ImageGeneratorView.hpp"
#include "MenuController.hpp"

// Handles all input and per-frame logic for the image generator screen.
// Owns an ImageGeneratorPresenter for state mutations and polls the generation
// atomics each frame to detect completion and load the result texture.
class ImageGeneratorController {
public:
    // Route an SFML event (key presses, mouse clicks, text entered, etc.) to the
    // appropriate handler. May change appScreen to navigate away from this screen.
    void handleEvent(const sf::Event& event, sf::RenderWindow& win,
                     ImageGeneratorView& screen, AppScreen& appScreen);

    // Called every frame. Checks whether background generation has completed and,
    // if so, loads the output image into the view's result texture.
    void update(ImageGeneratorView& screen);

private:
    // Dispatch a mouse click at pos to whichever hit-rect it lands in.
    void handleClick(sf::Vector2f pos, sf::RenderWindow& win,
                     ImageGeneratorView& view, AppScreen& appScreen);

    ImageGeneratorPresenter presenter;
};
