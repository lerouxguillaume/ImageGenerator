#include "ImageGeneratorView.hpp"
#include "../enum/constants.hpp"
#include "../ui/Buttons.hpp"
#include "../ui/Helpers.hpp"

#include <cstdio>

using namespace Helpers;

static constexpr float FIELD_W  = 700.f;
static constexpr float FIELD_H  = 40.f;
static constexpr float LEFT_X   = (WIN_W - FIELD_W) / 2.f;
static constexpr float cx       = WIN_W / 2.f;

// Truncate long strings for display — show the tail so typing feedback is visible
static std::string displayText(const std::string& s, size_t maxChars = 80) {
    if (s.size() <= maxChars) return s;
    return "..." + s.substr(s.size() - (maxChars - 3));
}

void ImageGeneratorView::render(sf::RenderWindow& win) {
    float y = static_cast<float>(PAD) * 2.f;

    // Back button + title
    btnBack = {LEFT_X, y, 80.f, 28.f};
    drawButton(win, btnBack, "< Back", Col::Panel2, Col::Muted, false, 12, font);
    drawTextC(win, font, "Image Generator", Col::GoldLt, cx, y + 5.f, 18, true);
    y += 44.f;

    // Model selector
    {
        constexpr float arrowW = 24.f;
        constexpr float arrowH = 22.f;
        const std::string modelName = availableModels.empty()
            ? "(no models found)"
            : availableModels[selectedModelIdx];
        drawText(win, font, "Model:", Col::Muted, LEFT_X, y + 4.f, 12);
        const float arrowX = LEFT_X + 60.f;
        btnModelPrev = {arrowX, y, arrowW, arrowH};
        drawButton(win, btnModelPrev, "<", Col::Panel2, Col::Muted, false, 12, font);
        drawText(win, font, modelName, Col::Text, arrowX + arrowW + 8.f, y + 4.f, 12);
        btnModelNext = {arrowX + arrowW + 8.f + 200.f, y, arrowW, arrowH};
        drawButton(win, btnModelNext, ">", Col::Panel2, Col::Muted, false, 12, font);
    }
    y += 34.f;

    // Positive prompt
    drawText(win, font, "Positive prompt:", Col::Muted, LEFT_X, y, 12);
    y += 18.f;
    positiveField = {LEFT_X, y, FIELD_W, FIELD_H};
    const sf::Color posiBorder = positiveActive ? Col::BorderHi : Col::Border;
    drawRect(win, positiveField, Col::Panel2, posiBorder, 1.f);
    const std::string posiDisplay = displayText(positivePrompt) + (positiveActive ? "|" : "");
    drawText(win, font, posiDisplay, Col::Text, LEFT_X + 8.f, y + 11.f, 13);
    y += FIELD_H + static_cast<float>(PAD) * 2.f;

    // Negative prompt
    drawText(win, font, "Negative prompt:", Col::Muted, LEFT_X, y, 12);
    y += 18.f;
    negativeField = {LEFT_X, y, FIELD_W, FIELD_H};
    const sf::Color negBorder = negativeActive ? Col::BorderHi : Col::Border;
    drawRect(win, negativeField, Col::Panel2, negBorder, 1.f);
    const std::string negDisplay = displayText(negativePrompt) + (negativeActive ? "|" : "");
    drawText(win, font, negDisplay, Col::Muted, LEFT_X + 8.f, y + 11.f, 13);
    y += FIELD_H + static_cast<float>(PAD) * 2.f;

    // Advanced toggle
    const std::string toggleLabel = showAdvancedParams ? "v Advanced" : "> Advanced";
    btnAdvanced = {LEFT_X, y, 100.f, 22.f};
    drawButton(win, btnAdvanced, toggleLabel, Col::Panel, Col::Muted, false, 11, font);
    y += 30.f;

    if (showAdvancedParams) {
        constexpr float sliderWidth = FIELD_W;
        const float sliderH = 18.f;

        // Steps slider (range 5–30)
        stepsSliderTrack = {LEFT_X, y + 14.f, sliderWidth, sliderH};
        const float stepsNorm = (generationParams.numSteps - 5.f) / 45.f;
        drawSlider(win, stepsSliderTrack, stepsNorm,
                   "Steps", std::to_string(generationParams.numSteps));
        y += 34.f;

        // CFG Scale slider (range 1.0–15.0)
        cfgSliderTrack = {LEFT_X, y + 14.f, sliderWidth, sliderH};
        const float cfgNorm = (generationParams.guidanceScale - 1.0f) / 19.0f;
        char cfgBuf[16];
        std::snprintf(cfgBuf, sizeof(cfgBuf), "%.1f", generationParams.guidanceScale);
        drawSlider(win, cfgSliderTrack, cfgNorm, "CFG Scale", cfgBuf);
        y += 34.f;

        // Images slider (range 1–10)
        imagesSliderTrack = {LEFT_X, y + 14.f, sliderWidth, sliderH};
        const float imagesNorm = (generationParams.numImages - 1.f) / 19.f;
        drawSlider(win, imagesSliderTrack, imagesNorm,
                   "Images", std::to_string(generationParams.numImages));
        y += 44.f;
    }

    // Generate button
    btnGenerate = {cx - 80.f, y, 160.f, 38.f};
    drawButton(win, btnGenerate, "Generate", Col::Panel2, Col::GoldLt, false, 14, font);
    y += 54.f;

    // Result image
    if (resultLoaded) {
        constexpr float imgSize = 256.f;
        const float imgX = cx - imgSize / 2.f;
        sf::Sprite sprite(resultTexture);
        const auto texSize = resultTexture.getSize();
        sprite.setScale(imgSize / static_cast<float>(texSize.x),
                        imgSize / static_cast<float>(texSize.y));
        sprite.setPosition(imgX, y);
        win.draw(sprite);
    }

    if (generating)
        drawGeneratingOverlay(win);
}

void ImageGeneratorView::drawSlider(sf::RenderWindow& win,
                                    const sf::FloatRect& track, float normalised,
                                    const std::string& label, const std::string& valueStr) {
    constexpr float visualH = 6.f;
    constexpr float thumbW  = 12.f;
    constexpr float thumbH  = 18.f;

    const float barY = track.top + (track.height - visualH) / 2.f;
    const sf::FloatRect visualBar = {track.left, barY, track.width, visualH};
    drawRect(win, visualBar, Col::Panel2, Col::Border, 1.f);
    if (normalised > 0.f)
        drawRect(win, {track.left, barY, track.width * normalised, visualH}, Col::Gold);

    const float thumbX = track.left + track.width * normalised - thumbW / 2.f;
    const float thumbY = track.top  + (track.height - thumbH)  / 2.f;
    drawRect(win, {thumbX, thumbY, thumbW, thumbH}, Col::GoldLt, Col::Border, 1.f);

    drawText(win,  font, label,    Col::Muted,  track.left,               track.top - 13.f, 11);
    drawTextR(win, font, valueStr, Col::GoldLt, track.left + track.width, track.top - 13.f, 11);
}

void ImageGeneratorView::drawGeneratingOverlay(sf::RenderWindow& win) {
    sf::RectangleShape overlay({static_cast<float>(WIN_W), static_cast<float>(WIN_H)});
    overlay.setFillColor(Col::Overlay);
    win.draw(overlay);

    constexpr float modalWidth  = 340.f;
    constexpr float modalHeight = 148.f;
    const sf::FloatRect modalBox = {cx - modalWidth / 2.f,
                                    WIN_H / 2.f - modalHeight / 2.f,
                                    modalWidth, modalHeight};
    drawRect(win, modalBox, Col::Panel, Col::Border, 1.f);

    const int imgNum   = generationImageNum.load();
    const int imgTotal = generationTotalImages.load();
    const std::string imgLabel = imgTotal > 1
        ? "Generating image " + std::to_string(imgNum) + " / " + std::to_string(imgTotal) + "..."
        : "Generating image...";
    drawTextC(win, font, imgLabel, Col::GoldLt, cx, modalBox.top + 16.f, 15, true);

    const int   currentStep = generationStep.load();
    const int   totalSteps  = generationParams.numSteps;
    const float progress    = totalSteps > 0 ? static_cast<float>(currentStep) / static_cast<float>(totalSteps) : 0.f;
    constexpr float barPad    = 20.f;
    constexpr float barHeight = 10.f;
    const float barWidth = modalWidth - barPad * 2.f;
    const float barX     = modalBox.left + barPad;
    const float barY     = modalBox.top + 48.f;
    drawRect(win, {barX, barY, barWidth, barHeight}, Col::Panel2, Col::Border, 1.f);
    if (progress > 0.f)
        drawRect(win, {barX, barY, barWidth * progress, barHeight}, Col::Gold);

    const std::string stepLabel = "Step " + std::to_string(currentStep) + " / " + std::to_string(totalSteps);
    drawTextC(win, font, stepLabel, Col::Muted, cx, modalBox.top + 72.f, 11);

    btnCancelGenerate = {cx - 55.f, modalBox.top + modalHeight - 34.f, 110.f, 26.f};
    drawButton(win, btnCancelGenerate, "Cancel", Col::Panel2, Col::RedLt, false, 12, font);
}
