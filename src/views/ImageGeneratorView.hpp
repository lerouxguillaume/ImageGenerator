#pragma once
#include <string>
#include <vector>
#include <atomic>
#include <SFML/Graphics.hpp>
#include "Screen.hpp"
#include "../portraits/PortraitGeneratorAi.hpp"
#include "../ui/SliderTypes.hpp"


class ImageGeneratorView : public Screen {
public:
    std::string positivePrompt;
    std::string negativePrompt;
    int positiveCursor = 0;
    int negativeCursor = 0;
    bool positiveActive = true;
    bool negativeActive = false;

    // Visual line layout cache — updated each render frame
    struct VisualLine { int start, end; }; // [start, end) byte indices into the prompt string
    std::vector<VisualLine> positiveLines;
    std::vector<VisualLine> negativeLines;
    int  positiveScrollLine  = 0;
    int  negativeScrollLine  = 0;
    bool positiveAllSelected = false;
    bool negativeAllSelected = false;

    GenerationParams generationParams;
    bool showAdvancedParams = false;

    bool              generating    = false;
    std::atomic<bool> generationDone{false};
    std::atomic<bool> cancelToken{false};
    std::atomic<int>  generationStep{0};
    std::atomic<int>  generationImageNum{0};   // 1-based current image
    std::atomic<int>  generationTotalImages{1};
    std::string       lastImagePath;

    std::atomic<int> generationId{0};

    sf::Texture resultTexture;
    bool        resultLoaded = false;

    // Model selection
    std::vector<std::string> availableModels;
    int selectedModelIdx = 0;

    // Hit rects (written during render, read by controller)
    sf::FloatRect positiveField;
    sf::FloatRect negativeField;
    sf::FloatRect btnGenerate;
    sf::FloatRect btnBack;
    sf::FloatRect btnAdvanced;
    sf::FloatRect btnCancelGenerate;
    sf::FloatRect btnModelPrev;
    sf::FloatRect btnModelNext;
    sf::FloatRect stepsSliderTrack;
    sf::FloatRect cfgSliderTrack;
    sf::FloatRect imagesSliderTrack;
    DraggingSlider draggingSlider = DraggingSlider::None;

    void render(sf::RenderWindow& win) override;

private:
    void drawPromptField(sf::RenderWindow& win,
                         const sf::FloatRect& field,
                         const std::string& text,
                         int cursor, bool active, bool allSelected,
                         sf::Color textColor,
                         std::vector<VisualLine>& outLines,
                         int& scrollLine);
    void drawSlider(sf::RenderWindow& win,
                    const sf::FloatRect& track, float normalised,
                    const std::string& label, const std::string& valueStr);
    void drawGeneratingOverlay(sf::RenderWindow& win);
};
