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
    bool positiveActive = true;
    bool negativeActive = false;

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
    void drawSlider(sf::RenderWindow& win,
                    const sf::FloatRect& track, float normalised,
                    const std::string& label, const std::string& valueStr);
    void drawGeneratingOverlay(sf::RenderWindow& win);
};
