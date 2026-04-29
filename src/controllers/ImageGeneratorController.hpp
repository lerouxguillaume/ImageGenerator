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
#include "../projects/Project.hpp"
#include "../views/ImageGeneratorView.hpp"
#include "MenuController.hpp"

class ImageGeneratorController {
public:
    ImageGeneratorController(AppConfig& cfg, WorkflowMode workflowMode = WorkflowMode::Generate)
        : config(cfg)
        , mode_(workflowMode)
        , enhancer(std::make_shared<NullPromptEnhancer>())
    {
        if (!config.promptEnhancer.modelDir.empty())
            startLlmLoad(config.promptEnhancer.modelDir);
    }

    void handleEvent(const sf::Event& event, sf::RenderWindow& win,
                     ImageGeneratorView& screen, AppScreen& appScreen);

    void update(ImageGeneratorView& screen);
    void prepareEditSession(ImageGeneratorView& screen, const std::string& imagePath);
    std::string consumePendingEditNavigation();
    void setBackScreen(AppScreen screen);
    void activateProjectSession(ImageGeneratorView& view, const ResolvedProjectContext& ctx);
    ResolvedProjectContext getProjectContext() const;
    void triggerGeneration(ImageGeneratorView& view);
    void openSettingsDialog(ImageGeneratorView& view);

    // Sets the active project context for the next generation session.
    // Resets the gallery to the project/asset-type subfolder.
    void setProjectContext(const ResolvedProjectContext& ctx);
    void clearProjectContext();

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
    void refreshGallery(ImageGeneratorView& view, const std::string& preferredSelection = {});
    void selectGalleryImage(ImageGeneratorView& view, int index);

    // ── State ─────────────────────────────────────────────────────────────────
    AppConfig&                       config;
    PresetManager                    presetManager;
    WorkflowMode                     mode_;
    AppScreen                        backScreen_ = AppScreen::MENU;
    std::shared_ptr<IPromptEnhancer> enhancer;
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

    // Async LLM enhancement (owned — result polled in update())
    std::future<LLMResponse> enhancementFuture_;

    // Generation thread (jthread: auto-requests-stop + joins on destruction/reassignment)
    std::jthread generationThread_;

    // Async folder browser
    enum class BrowseTarget { ModelDir, OutputDir, LlmDir, LoraDir };
    std::future<std::string> browseFuture;
    BrowseTarget             browseTarget = BrowseTarget::ModelDir;

    // Active project context (empty = no project active)
    ResolvedProjectContext projectContext_;

    // Async thumbnail loading (load+resize on background thread, create texture on main thread)
    struct PendingThumb {
        std::string            path;
        std::future<sf::Image> imageFuture;
    };
    std::vector<PendingThumb> pendingThumbs_;
    void flushPendingThumbs(ImageGeneratorView& view);
    std::string pendingEditNavigationPath_;
};
