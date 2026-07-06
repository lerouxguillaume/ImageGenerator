#pragma once
#include <atomic>
#include <string>
#include <vector>
#include <SFML/Graphics.hpp>
#include "../../portraits/PortraitGeneratorAi.hpp"  // GenerationParams, LoraEntry
#include "../../prompt/Prompt.hpp"
#include "../../enum/enums.hpp"
#include "../../import/ImportedModelRegistry.hpp"   // ModelCapabilities
#include "MultiLineTextArea.hpp"
#include "../SliderTypes.hpp"

// Left panel: model selector, LoRA panel, prompts, sliders, seed input.
// Owns its own rendering and all input handling for this region.
class SettingsPanel {
public:
    // ── Prompt fields ─────────────────────────────────────────────────────────
    MultiLineTextArea positiveArea{2000};
    MultiLineTextArea negativeArea{2000, 3};

    // ── Generation parameters ─────────────────────────────────────────────────
    GenerationParams generationParams;

    // ── Model selection ───────────────────────────────────────────────────────
    // Single source of truth for the model list, populated by the controller
    // from ImportedModelRegistry. Persisted references (presets, modelConfigs)
    // key on ModelEntry::id — the stable id, never the display name.
    struct ModelEntry {
        std::string       id;           // stable id == folder name under models/imported/<id>/
        std::string       displayName;  // shown in the UI
        std::string       path;         // onnx directory path (see getSelectedModelDir)
        ModelType         type = ModelType::SD15;
        ModelCapabilities capabilities;
    };
    std::vector<ModelEntry> models;
    int  selectedModelIdx = 0;
    bool showModelDropdown = false;
    std::string activePresetId; // empty when no preset is active

    // Returns the selected entry, or nullptr when no model is available.
    const ModelEntry* currentModel() const {
        const auto idx = static_cast<size_t>(selectedModelIdx);
        return idx < models.size() ? &models[idx] : nullptr;
    }
    bool currentModelVaeEncoderAvailable() const {
        const auto* m = currentModel();
        return !m || m->capabilities.vaeEncoderAvailable;
    }
    bool currentModelLoraCompatible() const {
        const auto* m = currentModel();
        return !m || m->capabilities.loraCompatible;
    }
    ModelType currentModelType() const {
        const auto* m = currentModel();
        return m ? m->type : ModelType::SD15;
    }

    // ── LoRA selection ────────────────────────────────────────────────────────
    std::vector<std::string> availableLoras;
    std::vector<bool>        loraSelected;
    std::vector<float>       loraScales;
    std::vector<std::string> loraScaleInputs;
    bool showLoraPanel      = false;
    int  activeLoraScaleIdx = -1;

    // ── Seed input ────────────────────────────────────────────────────────────
    std::string seedInput;
    int  seedInputCursor = 0;
    bool seedInputActive = false;

    // ── DSL display (set by controller each update) ───────────────────────────
    Prompt      currentDsl;       // parsed DSL — drives token chip display
    std::string compiledPreview;  // non-empty only for SD1.5 (shows compiled positive)

    // ── Interface ─────────────────────────────────────────────────────────────
    void setRect(const sf::FloatRect& rect);
    void render(sf::RenderWindow& win, sf::Font& font);

    // Returns true if the event was consumed.
    // Also handles mouse-wheel scroll on text areas.
    bool handleEvent(const sf::Event& e);

    // Returns the full path of the selected model dir, or "" if no model is available.
    std::string getSelectedModelDir() const;
    sf::FloatRect getRect() const;

private:
    sf::FloatRect rect_;

    // Hit rects (written during render, read by handleEvent)
    sf::FloatRect btnModelDropdown_;
    sf::FloatRect btnLoraPanel_;
    std::vector<sf::FloatRect> modelDropdownItems_;
    std::vector<sf::FloatRect> loraRowToggleRects_;
    std::vector<sf::FloatRect> loraScaleRects_;
    sf::FloatRect stepsSliderTrack_;
    sf::FloatRect cfgSliderTrack_;
    sf::FloatRect imagesSliderTrack_;
    sf::FloatRect strengthSliderTrack_;
    sf::FloatRect btnClearInit_;
    sf::FloatRect btnStrengthSubtle_;
    sf::FloatRect btnStrengthMedium_;
    sf::FloatRect btnStrengthStrong_;
    sf::FloatRect seedField_;

    // Slider drag state
    DraggingSlider draggingSlider_ = DraggingSlider::None;

    void drawSlider(sf::RenderWindow& win, sf::Font& font,
                    const sf::FloatRect& track, float normalised,
                    const std::string& label, const std::string& valueStr);

    void drawSingleLineField(sf::RenderWindow& win, sf::Font& font,
                             const sf::FloatRect& field,
                             const std::string& text,
                             int cursor, bool active);

    // Returns true if a click at pos was handled internally.
    bool handleClick(sf::Vector2f pos);
};
