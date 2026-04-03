#include "ImageGeneratorView.hpp"
#include "../enum/constants.hpp"
#include "../ui/Buttons.hpp"
#include "../ui/Helpers.hpp"

#include <algorithm>
#include <cstdio>

using namespace Helpers;

static constexpr float FIELD_W       = 700.f;
static constexpr float FIELD_H       = 86.f;   // 4 visible lines × 17px + 18px padding
static constexpr float FIELD_PAD_X   = 8.f;
static constexpr float FIELD_PAD_Y   = 6.f;
static constexpr float FIELD_LINE_H  = 17.f;   // font size 13 + 4px leading
static constexpr int   FIELD_VISIBLE = 4;
static constexpr unsigned FIELD_FONT = 13;
static constexpr float LEFT_X   = (WIN_W - FIELD_W) / 2.f;
static constexpr float cx       = WIN_W / 2.f;

// Compute pixel-accurate word-wrapped line layout for a prompt string.
// Returns a list of [start, end) byte index pairs, one per visual line.
using VisualLine = ImageGeneratorView::VisualLine;
static std::vector<VisualLine> computeLines(
    const std::string& s, sf::Font& font, unsigned charSize, float maxW)
{
    std::vector<VisualLine> lines;
    if (s.empty()) { lines.push_back({0, 0}); return lines; }

    sf::Text t;
    t.setFont(font);
    t.setCharacterSize(charSize);

    // Build word list with original byte offsets
    struct Word { int start, end; std::string text; };
    std::vector<Word> words;
    int i = 0;
    const int n = static_cast<int>(s.size());
    while (i < n) {
        while (i < n && s[i] == ' ') ++i;
        if (i >= n) break;
        const int ws = i;
        while (i < n && s[i] != ' ') ++i;
        words.push_back({ws, i, s.substr(ws, i - ws)});
    }

    if (words.empty()) { lines.push_back({0, 0}); return lines; }

    int lineStart = words[0].start;
    int lineEnd   = words[0].end;
    std::string lineText = words[0].text;

    for (int wi = 1; wi < static_cast<int>(words.size()); ++wi) {
        const std::string test = lineText + " " + words[wi].text;
        t.setString(test);
        if (t.getLocalBounds().width > maxW) {
            lines.push_back({lineStart, lineEnd});
            lineStart = words[wi].start;
            lineEnd   = words[wi].end;
            lineText  = words[wi].text;
        } else {
            lineEnd  = words[wi].end;
            lineText = test;
        }
    }
    lines.push_back({lineStart, lineEnd});
    return lines;
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
    drawPromptField(win, positiveField, positivePrompt, positiveCursor,
                    positiveActive, positiveAllSelected, Col::Text, positiveLines, positiveScrollLine);
    y += FIELD_H + static_cast<float>(PAD) * 2.f;

    // Negative prompt
    drawText(win, font, "Negative prompt:", Col::Muted, LEFT_X, y, 12);
    y += 18.f;
    negativeField = {LEFT_X, y, FIELD_W, FIELD_H};
    drawPromptField(win, negativeField, negativePrompt, negativeCursor,
                    negativeActive, negativeAllSelected, Col::Muted, negativeLines, negativeScrollLine);
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

void ImageGeneratorView::drawPromptField(
    sf::RenderWindow& win,
    const sf::FloatRect& field,
    const std::string& text,
    int cursor, bool active, bool allSelected,
    sf::Color textColor,
    std::vector<VisualLine>& outLines,
    int& scrollLine)
{
    drawRect(win, field, Col::Panel2, active ? Col::BorderHi : Col::Border, 1.f);

    // Selection highlight (drawn before text so text renders on top)
    if (active && allSelected) {
        sf::RectangleShape sel({field.width - 2.f, field.height - 2.f});
        sel.setPosition(field.left + 1.f, field.top + 1.f);
        sel.setFillColor(sf::Color(42, 85, 138, 110));
        win.draw(sel);
    }

    const float maxW = field.width - FIELD_PAD_X * 2.f - 6.f; // reserve 6px for scrollbar
    outLines = computeLines(text, font, FIELD_FONT, maxW);

    // Find cursor's visual line (cursor on line l if cursor < next line's start)
    int cursorLine = static_cast<int>(outLines.size()) - 1;
    for (int l = 0; l + 1 < static_cast<int>(outLines.size()); ++l) {
        if (cursor < outLines[l + 1].start) { cursorLine = l; break; }
    }

    // Auto-scroll so cursor stays visible when field is active
    if (active) {
        if (cursorLine < scrollLine) scrollLine = cursorLine;
        if (cursorLine >= scrollLine + FIELD_VISIBLE) scrollLine = cursorLine - FIELD_VISIBLE + 1;
    }
    scrollLine = std::clamp(scrollLine, 0,
                            std::max(0, static_cast<int>(outLines.size()) - FIELD_VISIBLE));

    // Draw visible lines
    const int last = std::min(static_cast<int>(outLines.size()), scrollLine + FIELD_VISIBLE);
    for (int l = scrollLine; l < last; ++l) {
        const auto [lStart, lEnd] = outLines[l];
        std::string lineText = text.substr(static_cast<size_t>(lStart),
                                           static_cast<size_t>(lEnd - lStart));
        if (active && !allSelected && l == cursorLine) {
            const int col = std::clamp(cursor - lStart, 0, static_cast<int>(lineText.size()));
            lineText.insert(static_cast<size_t>(col), "|");
        }
        const float ly = field.top + FIELD_PAD_Y + static_cast<float>(l - scrollLine) * FIELD_LINE_H;
        drawText(win, font, lineText, textColor, field.left + FIELD_PAD_X, ly, FIELD_FONT);
    }

    // Scrollbar (only when content overflows)
    if (static_cast<int>(outLines.size()) > FIELD_VISIBLE) {
        const float trackH = field.height - FIELD_PAD_Y * 2.f;
        const float ratio  = static_cast<float>(FIELD_VISIBLE) / static_cast<float>(outLines.size());
        const float thumbH = std::max(8.f, trackH * ratio);
        const float thumbY = field.top + FIELD_PAD_Y
                           + trackH * static_cast<float>(scrollLine)
                                     / static_cast<float>(outLines.size());
        const float trackX = field.left + field.width - FIELD_PAD_X + 1.f;
        drawRect(win, {trackX, field.top + FIELD_PAD_Y, 4.f, trackH}, Col::Panel);
        drawRect(win, {trackX, thumbY, 4.f, thumbH}, Col::Border);
    }
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
