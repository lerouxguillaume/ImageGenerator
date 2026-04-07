#pragma once
#include <future>
#include <memory>
#include <SFML/Graphics.hpp>
#include "../config/AppConfig.hpp"
#include "../llm/IPromptEnhancer.hpp"
#include "../llm/PromptEnhancerFactory.hpp"
#include "../presenters/ImageGeneratorPresenter.hpp"
#include "../views/ImageGeneratorView.hpp"
#include "MenuController.hpp"

class ImageGeneratorController {
public:
    explicit ImageGeneratorController(AppConfig cfg)
        : config(std::move(cfg))
        , enhancer(PromptEnhancerFactory::create(config.promptEnhancer.enabled,
                                                  config.promptEnhancer.modelDir))
    {}

    void handleEvent(const sf::Event& event, sf::RenderWindow& win,
                     ImageGeneratorView& screen, AppScreen& appScreen);

    void update(ImageGeneratorView& screen);

private:
    void handleClick(sf::Vector2f pos, sf::RenderWindow& win,
                     ImageGeneratorView& view, AppScreen& appScreen);

    // Open the settings modal, copying current config values into the view fields.
    void openSettings(ImageGeneratorView& view);

    // Persist edits from the view fields back to config and close the modal.
    void saveSettings(ImageGeneratorView& view);

    // Apply the active model's defaults (falling back to global config defaults).
    void applyModelDefaults(ImageGeneratorView& view);

    AppConfig                        config;
    ImageGeneratorPresenter          presenter;
    std::unique_ptr<IPromptEnhancer> enhancer;
    bool                             modelsDirty     = true;
    bool                             viewInitialized = false;
    int                              lastModelIdx    = -1;

    // Async folder browser (zenity runs on a thread; result polled in update())
    std::future<std::string> browseFuture;
    bool                     browsingForModel = true; // which field receives the result
};
