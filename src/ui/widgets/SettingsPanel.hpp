#pragma once
#include <atomic>
#include <string>
#include <vector>
#include <SFML/Graphics.hpp>
#include "../../portraits/PortraitGeneratorAi.hpp"  // GenerationParams, LoraEntry
#include "MultiLineTextArea.hpp"
#include "../SliderTypes.hpp"

// Left panel: model selector, LoRA panel, prompts, sliders, seed input.
// Owns its own rendering and all input handling for this region.
class SettingsPanel {
public:
    // ── Prompt fields ─────────────────────────────────────────────────────────
    MultiLineTextArea positiveArea{2000};
    MultiLineTextArea negativeArea{2000};

    // ── Generation parameters ─────────────────────────────────────────────────
    GenerationParams generationParams;

    // ── Model selection ───────────────────────────────────────────────────────
    std::vector<std::string> availableModels; // set by controller after scan
    int  selectedModelIdx = 0;
    bool showModelDropdown = false;
    std::string activePresetId; // empty when no preset is active

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

    // ── Interface ─────────────────────────────────────────────────────────────
    void setRect(const sf::FloatRect& rect);
    void render(sf::RenderWindow& win, sf::Font& font);

    // Returns true if the event was consumed.
    // Also handles mouse-wheel scroll on text areas.
    bool handleEvent(const sf::Event& e);

    // Returns the full path of the selected model dir, or "models" if none.
    std::string getSelectedModelDir() const;

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
