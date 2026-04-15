#include "ImageGeneratorView.hpp"
#include "../enum/constants.hpp"
#include "../ui/Buttons.hpp"
#include "../ui/Helpers.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>

using namespace Helpers;

static constexpr float FIELD_W      = 700.f;
static constexpr float FIELD_H      = 86.f;   // 4 visible lines × 17px + 18px padding
static constexpr float FIELD_H_SM   = 46.f;   // 2 visible lines × 17px + 12px padding
static constexpr float LEFT_X  = (WIN_W - FIELD_W) / 2.f;
static constexpr float cx      = WIN_W / 2.f;

void ImageGeneratorView::render(sf::RenderWindow& win) {
    float y = static_cast<float>(PAD) * 2.f;

    // Back button + title + settings button
    btnBack = {LEFT_X, y, 80.f, 28.f};
    drawButton(win, btnBack, "< Back", Col::Panel2, Col::Muted, false, 12, font);
    drawTextC(win, font, "Image Generator", Col::GoldLt, cx, y + 5.f, 18, true);
    btnSettings = {WIN_W - LEFT_X - 80.f, y, 80.f, 28.f};
    drawButton(win, btnSettings, "Settings", Col::Panel2, Col::Muted, false, 12, font);
    y += 44.f;

    // Model selector (dropdown) + LoRA button on the same row
    {
        const std::string displayName = availableModels.empty()
            ? "(no models found)"
            : std::filesystem::path(availableModels[selectedModelIdx]).filename().string();
        const std::string label = displayName + (showModelDropdown ? "  ^" : "  v");
        drawText(win, font, "Model:", Col::Muted, LEFT_X, y + 4.f, 12);
        btnModelDropdown = {LEFT_X + 60.f, y, 300.f, 22.f};
        drawButton(win, btnModelDropdown, label, Col::Panel2,
                   showModelDropdown ? Col::GoldLt : Col::Text, false, 12, font);

        const int numSel = static_cast<int>(
            std::count(loraSelected.begin(), loraSelected.end(), true));
        const std::string loraLabel = "LoRA (" + std::to_string(numSel) + ")";
        btnLoraPanel = {LEFT_X + 370.f, y, 110.f, 22.f};
        drawButton(win, btnLoraPanel, loraLabel, Col::Panel2,
                   showLoraPanel ? Col::GoldLt : Col::Text, false, 12, font);
    }
    y += 28.f;

    // Positive prompt
    drawText(win, font, "Positive prompt:", Col::Muted, LEFT_X, y, 12);
    y += 18.f;
    positiveArea.setRect({LEFT_X, y, FIELD_W, FIELD_H});
    positiveArea.render(win, font);
    y += FIELD_H + static_cast<float>(PAD) * 2.f;

    // Negative prompt
    drawText(win, font, "Negative prompt:", Col::Muted, LEFT_X, y, 12);
    y += 18.f;
    negativeArea.setRect({LEFT_X, y, FIELD_W, FIELD_H});
    negativeArea.setTextColor(Col::Muted);
    negativeArea.render(win, font);
    y += FIELD_H + static_cast<float>(PAD) * 2.f;

    // Instruction field — only rendered when a LLM model is ready or loading
    if (promptEnhancerAvailable || llmLoading) {
        drawText(win, font, "Instruction (optional):", Col::Muted, LEFT_X, y, 12);
        y += 18.f;
        instructionArea.setRect({LEFT_X, y, FIELD_W, FIELD_H_SM});
        instructionArea.render(win, font);
        y += FIELD_H_SM + static_cast<float>(PAD);
    }

    // Advanced toggle + optional Enhance button on the same row
    const std::string toggleLabel = showAdvancedParams ? "v Advanced" : "> Advanced";
    btnAdvanced = {LEFT_X, y, 100.f, 22.f};
    drawButton(win, btnAdvanced, toggleLabel, Col::Panel, Col::Muted, false, 11, font);

    if (llmLoading) {
        btnEnhance = {};
        drawButton(win, {LEFT_X + FIELD_W - 130.f, y, 130.f, 22.f},
                   "LLM loading...", Col::Panel, Col::Muted, false, 11, font);
    } else if (promptEnhancerAvailable) {
        const std::string enhLabel = enhancing ? "Enhancing..." : "Enhance prompts";
        const sf::Color   enhCol   = enhancing ? Col::Muted : Col::GoldLt;
        btnEnhance = {LEFT_X + FIELD_W - 130.f, y, 130.f, 22.f};
        drawButton(win, btnEnhance, enhLabel, Col::Panel, enhCol, false, 11, font);
    }
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

        // Seed input
        drawText(win, font, "Seed:", Col::Muted, LEFT_X, y + 6.f, 12);
        drawText(win, font, "(empty = random)", Col::Border, LEFT_X + 44.f, y + 6.f, 11);
        constexpr float seedFieldW = 180.f;
        constexpr float seedFieldH = 24.f;
        seedField = {LEFT_X + sliderWidth - seedFieldW, y, seedFieldW, seedFieldH};
        drawSingleLineField(win, seedField, seedInput, seedInputCursor, seedInputActive);
        y += 34.f;
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

    // Error banner — shown when the last generation failed; cleared on next Generate
    if (generationFailed.load() && !generating) {
        std::string msg = generationErrorMsg;
        constexpr size_t kMaxLen = 120;
        if (msg.size() > kMaxLen)
            msg = msg.substr(0, kMaxLen) + "... (see log)";

        constexpr float bannerW = 480.f;
        constexpr float bannerH = 54.f;
        const sf::FloatRect banner = {cx - bannerW / 2.f, y, bannerW, bannerH};
        drawRect(win, banner, Col::Panel2, Col::RedLt, 1.f);
        drawTextC(win, font, "Generation failed", Col::RedLt, cx, banner.top + 8.f, 12, true);
        drawTextC(win, font, msg, Col::Muted, cx, banner.top + 28.f, 10);
    }

    // Model dropdown list (drawn on top of all other UI)
    if (showModelDropdown && !availableModels.empty()) {
        constexpr float itemH   = 22.f;
        constexpr float listPad = 2.f;
        const float listX = btnModelDropdown.left;
        const float listW = btnModelDropdown.width;
        const float listY = btnModelDropdown.top + btnModelDropdown.height + 2.f;
        const int   count = static_cast<int>(availableModels.size());
        const float listH = listPad * 2.f + itemH * static_cast<float>(count);

        drawRect(win, {listX, listY, listW, listH}, Col::Panel2, Col::BorderHi, 1.f);

        modelDropdownItems.resize(static_cast<size_t>(count));
        for (int i = 0; i < count; ++i) {
            const float itemY = listY + listPad + static_cast<float>(i) * itemH;
            modelDropdownItems[static_cast<size_t>(i)] = {listX, itemY, listW, itemH};
            const bool selected = (i == selectedModelIdx);
            if (selected)
                drawRect(win, modelDropdownItems[static_cast<size_t>(i)], Col::Panel);
            const std::string name =
                std::filesystem::path(availableModels[static_cast<size_t>(i)]).filename().string();
            drawText(win, font, name,
                     selected ? Col::GoldLt : Col::Text,
                     listX + 6.f, itemY + 4.f, 12);
        }
    }

    // LoRA panel (drawn on top of all other UI, like the model dropdown)
    if (showLoraPanel) {
        constexpr float panelW   = 340.f;
        constexpr float rowH     = 24.f;
        constexpr float scaleW   = 50.f;
        constexpr float listPad  = 4.f;
        const float panelX = btnLoraPanel.left;
        const float panelY = btnLoraPanel.top + btnLoraPanel.height + 2.f;
        const int   count  = static_cast<int>(availableLoras.size());

        if (count == 0) {
            drawRect(win, {panelX, panelY, panelW, 30.f}, Col::Panel2, Col::BorderHi, 1.f);
            drawText(win, font, "No .safetensors found in LoRA dir",
                     Col::Muted, panelX + 8.f, panelY + 8.f, 11);
        } else {
            const float panelH = listPad * 2.f + rowH * static_cast<float>(count);
            drawRect(win, {panelX, panelY, panelW, panelH}, Col::Panel2, Col::BorderHi, 1.f);

            loraRowToggleRects.resize(static_cast<size_t>(count));
            loraScaleRects.resize(static_cast<size_t>(count));

            for (int i = 0; i < count; ++i) {
                const float rowY2 = panelY + listPad + static_cast<float>(i) * rowH;
                const bool  sel   = (i < static_cast<int>(loraSelected.size())) && loraSelected[static_cast<size_t>(i)];

                // Highlight selected rows
                if (sel)
                    drawRect(win, {panelX + 1.f, rowY2, panelW - 2.f, rowH}, Col::Panel);

                // Checkbox
                constexpr float cbSize = 10.f;
                const float cbX = panelX + 8.f;
                const float cbY = rowY2 + (rowH - cbSize) / 2.f;
                drawRect(win, {cbX, cbY, cbSize, cbSize}, sel ? Col::GoldLt : Col::Panel2,
                         Col::Border, 1.f);

                // LoRA name (strip path and .safetensors)
                std::string name = std::filesystem::path(
                    availableLoras[static_cast<size_t>(i)]).stem().string();
                drawText(win, font, name, sel ? Col::GoldLt : Col::Text,
                         cbX + cbSize + 6.f, rowY2 + 5.f, 11);

                // Toggle rect covers name+checkbox area
                loraRowToggleRects[static_cast<size_t>(i)] = {
                    panelX, rowY2, panelW - scaleW - 4.f, rowH};

                // Scale text field
                const float scaleX = panelX + panelW - scaleW - 4.f;
                loraScaleRects[static_cast<size_t>(i)] = {scaleX, rowY2 + 2.f, scaleW, rowH - 4.f};
                const bool        scaleActive = (activeLoraScaleIdx == i);
                const std::string scaleText   = (i < static_cast<int>(loraScaleInputs.size()))
                                                    ? loraScaleInputs[static_cast<size_t>(i)] : "1";
                const int scaleCursor = scaleActive ? static_cast<int>(scaleText.size()) : 0;
                drawSingleLineField(win, loraScaleRects[static_cast<size_t>(i)],
                                    scaleText, scaleCursor, scaleActive);
            }
        }
    }

    if (generating)
        drawGeneratingOverlay(win);

    if (showSettings)
        drawSettingsModal(win);
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

void ImageGeneratorView::drawSingleLineField(sf::RenderWindow& win,
                                              const sf::FloatRect& field,
                                              const std::string& text,
                                              int cursor, bool active) {
    drawRect(win, field, Col::Panel, active ? Col::GoldLt : Col::Border, 1.f);

    constexpr float     padX     = 6.f;
    constexpr unsigned  fontSize = 13;
    const float         textY    = field.top + (field.height - static_cast<float>(fontSize)) / 2.f - 1.f;
    const float         maxW     = field.width - padX * 2.f;

    // Compute pixel offset of cursor so it stays visible as text grows.
    sf::Text measure;
    measure.setFont(font);
    measure.setCharacterSize(fontSize);
    measure.setString(text.substr(0, static_cast<size_t>(cursor)));
    const float cursorPx  = measure.getLocalBounds().width;
    const float scrollX   = std::max(0.f, cursorPx - maxW);

    sf::Text textObj;
    textObj.setFont(font);
    textObj.setCharacterSize(fontSize);
    textObj.setFillColor(Col::Text);
    textObj.setString(text);
    textObj.setPosition(field.left + padX - scrollX, textY);
    win.draw(textObj);

    if (active) {
        sf::RectangleShape cur({1.f, static_cast<float>(fontSize) + 3.f});
        cur.setFillColor(Col::GoldLt);
        cur.setPosition(field.left + padX + cursorPx - scrollX, textY);
        win.draw(cur);
    }
}

void ImageGeneratorView::drawSettingsModal(sf::RenderWindow& win) {
    // Dim the background
    sf::RectangleShape overlay({static_cast<float>(WIN_W), static_cast<float>(WIN_H)});
    overlay.setFillColor(Col::Overlay);
    win.draw(overlay);

    // Modal panel
    constexpr float boxW = 560.f, boxH = 330.f;
    const float boxX = (WIN_W - boxW) / 2.f;
    const float boxY = (WIN_H - boxH) / 2.f;
    drawRect(win, {boxX, boxY, boxW, boxH}, Col::Panel2, Col::BorderHi, 2.f);
    drawTextC(win, font, "Settings", Col::GoldLt, WIN_W / 2.f, boxY + 14.f, 15, true);

    constexpr float padX      = 20.f;
    constexpr float labelW    = 140.f;
    constexpr float browseW   = 38.f;
    constexpr float browseGap = 4.f;
    constexpr float fieldW    = 318.f;   // 360 - browseW - browseGap
    constexpr float fieldH    = 26.f;
    const float     fieldX    = boxX + padX + labelW;

    // Row 1: model directory
    float rowY = boxY + 52.f;
    drawText(win, font, "Model directory:", Col::Muted, boxX + padX, rowY + 6.f, 12);
    settingsModelDirField = {fieldX, rowY, fieldW, fieldH};
    drawSingleLineField(win, settingsModelDirField, settingsModelDir,
                        settingsModelDirCursor, settingsModelDirActive);
    settingsBtnBrowseModel = {fieldX + fieldW + browseGap, rowY, browseW, fieldH};
    drawButton(win, settingsBtnBrowseModel, "...", Col::Panel2, Col::Muted, false, 12, font);

    // Row 2: output directory
    rowY += 48.f;
    drawText(win, font, "Output directory:", Col::Muted, boxX + padX, rowY + 6.f, 12);
    settingsOutputDirField = {fieldX, rowY, fieldW, fieldH};
    drawSingleLineField(win, settingsOutputDirField, settingsOutputDir,
                        settingsOutputDirCursor, settingsOutputDirActive);
    settingsBtnBrowseOutput = {fieldX + fieldW + browseGap, rowY, browseW, fieldH};
    drawButton(win, settingsBtnBrowseOutput, "...", Col::Panel2, Col::Muted, false, 12, font);

    // Row 3: LLM model directory
    rowY += 48.f;
    drawText(win, font, "LLM model dir:", Col::Muted, boxX + padX, rowY + 6.f, 12);
    settingsLlmModelDirField = {fieldX, rowY, fieldW, fieldH};
    drawSingleLineField(win, settingsLlmModelDirField, settingsLlmModelDir,
                        settingsLlmModelDirCursor, settingsLlmModelDirActive);
    settingsBtnBrowseLlm = {fieldX + fieldW + browseGap, rowY, browseW, fieldH};
    drawButton(win, settingsBtnBrowseLlm, "...", Col::Panel2, Col::Muted, false, 12, font);
    if (llmLoading)
        drawText(win, font, "Loading...", Col::Muted, fieldX, rowY + fieldH + 2.f, 10);

    // Row 4: LoRA directory
    rowY += 48.f;
    drawText(win, font, "LoRA directory:", Col::Muted, boxX + padX, rowY + 6.f, 12);
    settingsLoraDirField = {fieldX, rowY, fieldW, fieldH};
    drawSingleLineField(win, settingsLoraDirField, settingsLoraDir,
                        settingsLoraDirCursor, settingsLoraDirActive);
    settingsBtnBrowseLora = {fieldX + fieldW + browseGap, rowY, browseW, fieldH};
    drawButton(win, settingsBtnBrowseLora, "...", Col::Panel2, Col::Muted, false, 12, font);

    // Buttons
    constexpr float btnW = 100.f, btnH = 28.f;
    const float btnY = boxY + boxH - btnH - 16.f;
    settingsBtnCancel = {fieldX + fieldW - btnW * 2.f - 8.f, btnY, btnW, btnH};
    settingsBtnSave   = {fieldX + fieldW - btnW,              btnY, btnW, btnH};
    drawButton(win, settingsBtnCancel, "Cancel", Col::Panel2, Col::Muted,  false, 12, font);
    drawButton(win, settingsBtnSave,   "Save",   Col::Panel,  Col::GoldLt, false, 12, font);
}
