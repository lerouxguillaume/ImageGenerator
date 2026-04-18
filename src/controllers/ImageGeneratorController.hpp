#pragma once
#include <future>
#include <memory>
#include <thread>
#include <SFML/Graphics.hpp>
#include "../config/AppConfig.hpp"
#include "../enum/enums.hpp"
#include "../llm/IPromptEnhancer.hpp"
#include "../llm/PromptEnhancerFactory.hpp"
#include "../presets/PresetManager.hpp"
#include "../views/ImageGeneratorView.hpp"
#include "MenuController.hpp"

class ImageGeneratorController {
public:
    explicit ImageGeneratorController(AppConfig cfg)
        : config(std::move(cfg))
        , enhancer(std::make_unique<NullPromptEnhancer>())
    {
        if (!config.promptEnhancer.modelDir.empty())
            startLlmLoad(config.promptEnhancer.modelDir);
    }

    void handleEvent(const sf::Event& event, sf::RenderWindow& win,
                     ImageGeneratorView& screen, AppScreen& appScreen);

    void update(ImageGeneratorView& screen);

private:
    // ── Settings helpers ──────────────────────────────────────────────────────
    void openSettings(ImageGeneratorView& view);
    void saveSettings(ImageGeneratorView& view);

    // ── Model defaults ────────────────────────────────────────────────────────
    void applyModelDefaults(ImageGeneratorView& view);

    // ── LLM async load ────────────────────────────────────────────────────────
    void startLlmLoad(const std::string& modelDir);

    // ── Generation ────────────────────────────────────────────────────────────
    void launchGeneration(ImageGeneratorView& view);
    void launchEnhancement(ImageGeneratorView& view);

    // ── State ─────────────────────────────────────────────────────────────────
    AppConfig                        config;
    PresetManager                    presetManager;
    std::unique_ptr<IPromptEnhancer> enhancer;
    bool                             modelsDirty     = true;
    bool                             lorasDirty      = true;
    bool                             viewInitialized = false;
    int                              lastModelIdx     = -1;
    ModelType                        cachedModelType_ = ModelType::SDXL;
    std::string                      lastPositiveText_;
    std::string                      lastNegativeText_;
    bool                             dslDirty_        = true;

    // Async LLM model load
    std::future<std::unique_ptr<IPromptEnhancer>> llmLoadFuture;

    // Generation thread (jthread: auto-requests-stop + joins on destruction/reassignment)
    std::jthread generationThread_;

    // Async folder browser
    enum class BrowseTarget { ModelDir, OutputDir, LlmDir, LoraDir };
    std::future<std::string> browseFuture;
    BrowseTarget             browseTarget = BrowseTarget::ModelDir;
};
