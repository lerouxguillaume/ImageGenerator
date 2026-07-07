#include "SettingsPanel.hpp"
#include "../../ui/Buttons.hpp"
#include "../../ui/Helpers.hpp"
#include "../../ui/Theme.h"
#include <SFML/Window/Clipboard.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>

using namespace Helpers;

void SettingsPanel::setRect(const sf::FloatRect& rect) {
    rect_ = rect;
}

std::string SettingsPanel::getSelectedModelDir() const {
    const auto* m = currentModel();
    return m ? m->path : std::string{};
}

sf::FloatRect SettingsPanel::getRect() const {
    return rect_;
}

// ── Draw helpers ──────────────────────────────────────────────────────────────

void SettingsPanel::drawSlider(sf::RenderWindow& win, sf::Font& font,
                               const sf::FloatRect& track, float normalised,
                               const std::string& label, const std::string& valueStr) {
    const auto& theme = Theme::instance();
    const auto& colors = theme.colors();
    const auto& metrics = theme.metrics();
    const auto& type = theme.typography();
    constexpr float visualH = 6.f;
    constexpr float thumbW  = 12.f;
    constexpr float thumbH  = 18.f;

    const float barY = track.top + (track.height - visualH) / 2.f;
    drawRect(win, {track.left, barY, track.width, visualH}, colors.surfaceInset, colors.border, metrics.borderWidth);
    if (normalised > 0.f)
        drawRect(win, {track.left, barY, track.width * normalised, visualH}, colors.gold);

    const float thumbX = track.left + track.width * normalised - thumbW / 2.f;
    const float thumbY = track.top  + (track.height - thumbH)  / 2.f;
    drawRect(win, {thumbX, thumbY, thumbW, thumbH}, colors.goldLt, colors.border, metrics.borderWidth);

    drawText(win,  font, label,    colors.muted,  track.left,               track.top - 13.f, type.compact);
    drawTextR(win, font, valueStr, colors.goldLt, track.left + track.width, track.top - 13.f, type.compact);
}

void SettingsPanel::drawSingleLineField(sf::RenderWindow& win, sf::Font& font,
                                        const sf::FloatRect& field,
                                        const std::string& text,
                                        int cursor, bool active) {
    const auto& theme = Theme::instance();
    const auto& colors = theme.colors();
    const auto& metrics = theme.metrics();
    drawRect(win, field, colors.surfaceInset, active ? colors.borderHi : colors.border, metrics.borderWidth);

    constexpr float    padX     = 6.f;
    constexpr unsigned fontSize = 13;
    const float        textY    = field.top + (field.height - static_cast<float>(fontSize)) / 2.f - 1.f;
    const float        maxW     = field.width - padX * 2.f;

    sf::Text measure;
    measure.setFont(font);
    measure.setCharacterSize(fontSize);
    measure.setString(text.substr(0, static_cast<size_t>(cursor)));
    const float cursorPx = measure.getLocalBounds().width;
    const float scrollX  = std::max(0.f, cursorPx - maxW);

    sf::Text textObj;
    textObj.setFont(font);
    textObj.setCharacterSize(fontSize);
    textObj.setFillColor(theme.colors().text);
    textObj.setString(text);
    textObj.setPosition(field.left + padX - scrollX, textY);
    win.draw(textObj);

    if (active) {
        sf::RectangleShape cur({1.f, static_cast<float>(fontSize) + 3.f});
        cur.setFillColor(theme.colors().goldLt);
        cur.setPosition(field.left + padX + cursorPx - scrollX, textY);
        win.draw(cur);
    }
}

// ── Render ────────────────────────────────────────────────────────────────────

void SettingsPanel::render(sf::RenderWindow& win, sf::Font& font) {
    const auto& theme = Theme::instance();
    const auto& colors = theme.colors();
    const auto& metrics = theme.metrics();
    const float x  = rect_.left;
    const float pw = rect_.width;       // panel width
    const float pad  = static_cast<float>(metrics.pad);

    // Panel background with right border
    drawRect(win, rect_, colors.panel2, colors.border, metrics.borderWidth);
    drawRect(win, {x + 1.f, rect_.top + 1.f, pw - 2.f, rect_.height - 2.f}, colors.panel, sf::Color::Transparent, 0.f);

    // Card geometry: cards are inset from the rail edges; content is inset again.
    const float cardX = x + pad;
    const float cardW = pw - pad * 2.f;
    const float cinX  = cardX + 12.f;   // inner content x
    const float cinW  = cardW - 24.f;   // inner content width
    constexpr float cardGap = 10.f;

    // Uppercase micro-label + framed body. Returns the content-start y (below header).
    auto drawCard = [&](float top, float h, const std::string& title) -> float {
        drawRect(win, {cardX, top, cardW, h}, colors.panel2, colors.border, metrics.borderWidth);
        drawText(win, font, title, colors.muted, cinX, top + 11.f, 10);
        return top + 32.f;
    };
    // Rectangular pill toggle (SFML has no rounded rects; matches the app's style).
    auto drawSwitch = [&](const sf::FloatRect& box, bool on, bool enabled) {
        drawRect(win, box, on ? colors.panel : colors.surfaceInset,
                 on ? colors.borderHi : colors.border, 1.f);
        const float knob = box.height - 4.f;
        const float kx   = on ? box.left + box.width - knob - 2.f : box.left + 2.f;
        drawRect(win, {kx, box.top + 2.f, knob, knob},
                 on ? colors.gold : (enabled ? colors.muted : colors.border),
                 sf::Color::Transparent, 0.f);
    };

    float y = rect_.top + pad + 2.f;

    // ── Model & LoRA card ─────────────────────────────────────────────────────
    {
        const float top   = y;
        const float inner = drawCard(top, 106.f, "MODEL & LORA");
        const ModelEntry* selected = currentModel();

        // Full-width dropdown styled as an input, with an inline arch badge.
        btnModelDropdown_ = {cinX, inner, cinW, 30.f};
        drawRect(win, btnModelDropdown_, colors.surfaceInset,
                 showModelDropdown ? colors.borderHi : colors.border, 1.f);
        const std::string dn = selected ? selected->displayName : "(no models found)";
        drawText(win, font, dn, showModelDropdown ? colors.goldLt : colors.text,
                 cinX + 10.f, inner + 8.f, 13);
        drawTextR(win, font, showModelDropdown ? "\xe2\x96\xb4" : "\xe2\x96\xbe",
                  colors.muted, cinX + cinW - 10.f, inner + 8.f, 11);
        if (selected) {
            const std::string badge = selected->type == ModelType::SDXL ? "SDXL" : "SD1.5";
            drawTextR(win, font, badge, colors.gold, cinX + cinW - 26.f, inner + 8.f, 10);
        }

        const float ly = inner + 38.f;
        if (currentModelLoraCompatible()) {
            const int numSel = static_cast<int>(std::count(loraSelected.begin(), loraSelected.end(), true));
            btnLoraPanel_ = {cinX, ly, 150.f, 26.f};
            drawButton(win, btnLoraPanel_, "LoRA (" + std::to_string(numSel) + ")",
                       colors.surfaceInset, showLoraPanel ? colors.goldLt : colors.text, false, 12, font);
        } else {
            btnLoraPanel_ = {};
            showLoraPanel = false;
            drawText(win, font, "No LoRA for this model", colors.muted, cinX, ly + 6.f, 11);
        }
        y = top + 106.f + cardGap;
    }

    // ── Prompt card ───────────────────────────────────────────────────────────
    constexpr float fieldH    = 82.f;
    constexpr float fieldH_sm = 56.f;
    {
        // Build the chip list first so the card height can account for it.
        struct Chip { std::string label; sf::Color border; sf::Color text; };
        std::vector<Chip> chips;
        if (currentDsl.subject) {
            const auto& subj = *currentDsl.subject;
            std::string lbl  = subj.value;
            if (std::abs(subj.weight - 1.0f) > 0.01f) {
                char buf[16]; std::snprintf(buf, sizeof(buf), " %.2g\xc3\x97", subj.weight); lbl += buf;
            }
            chips.push_back({lbl, colors.gold, colors.goldLt});
        }
        for (const auto& tok : currentDsl.positive) {
            std::string lbl = tok.value;
            if (std::abs(tok.weight - 1.0f) > 0.01f) {
                char buf[16]; std::snprintf(buf, sizeof(buf), " %.2g\xc3\x97", tok.weight); lbl += buf;
            }
            const sf::Color border = (tok.weight > 1.f) ? colors.blueLt
                                   : (tok.weight < 1.f) ? colors.muted : colors.border;
            chips.push_back({lbl, border, colors.text});
        }

        constexpr float chipH = 20.f, chipStep = 22.f, chipPadX = 6.f, chipGap = 4.f;
        sf::Text measure; measure.setFont(font); measure.setCharacterSize(11);
        int chipRows = 0;
        if (!chips.empty()) {
            chipRows = 1;
            float cxp = cinX;
            for (const auto& c : chips) {
                measure.setString(c.label);
                const float cw = measure.getLocalBounds().width + chipPadX * 2.f;
                if (cxp + cw > cinX + cinW && cxp > cinX) {
                    if (++chipRows > 2) { chipRows = 2; break; }
                    cxp = cinX;
                }
                cxp += cw + chipGap;
            }
        }

        const float chipsBlockH = (chipRows > 0) ? (6.f + static_cast<float>(chipRows) * chipStep) : 6.f;
        const bool  hasCompiled = !compiledPreview.empty();
        const float contentH = 16.f + fieldH + chipsBlockH + 16.f + fieldH_sm + (hasCompiled ? 18.f : 0.f);
        const float cardH    = 32.f + contentH + 10.f;

        const float top   = y;
        const float inner = drawCard(top, cardH, "PROMPT");

        float iy = inner;
        drawText(win, font, "Positive", colors.muted, cinX, iy, 11);
        iy += 16.f;
        positiveArea.setRect({cinX, iy, cinW, fieldH});
        positiveArea.render(win, font);
        iy += fieldH;

        if (chipRows > 0) {
            iy += 6.f;
            float cx = cinX, cy = iy; int rows = 1;
            for (const auto& chip : chips) {
                measure.setString(chip.label);
                const float cw = measure.getLocalBounds().width + chipPadX * 2.f;
                if (cx + cw > cinX + cinW && cx > cinX) {
                    cy += chipStep; cx = cinX;
                    if (++rows > 2) break;
                }
                drawRect(win, {cx, cy, cw, chipH}, colors.panel, chip.border, 1.f);
                drawText(win, font, chip.label, chip.text, cx + chipPadX, cy + 4.f, 11);
                cx += cw + chipGap;
            }
            iy += static_cast<float>(chipRows) * chipStep;
        } else {
            iy += 6.f;
        }

        drawText(win, font, "Negative", colors.muted, cinX, iy, 11);
        iy += 16.f;
        negativeArea.setRect({cinX, iy, cinW, fieldH_sm});
        negativeArea.setTextColor(colors.muted);
        negativeArea.render(win, font);
        iy += fieldH_sm;

        if (hasCompiled) {
            measure.setCharacterSize(10);
            std::string txt = compiledPreview;
            measure.setString(txt);
            while (!txt.empty() && measure.getLocalBounds().width > cinW - 14.f) {
                txt.pop_back(); measure.setString(txt);
            }
            if (txt.size() < compiledPreview.size()) txt += "\xe2\x80\xa6";
            drawText(win, font, "\xe2\x86\x92 " + txt, colors.muted, cinX, iy + 2.f, 10);
        }
        y = top + cardH + cardGap;
    }

    const float sliderH = 16.f;

    btnStrengthSubtle_ = {};
    btnStrengthMedium_ = {};
    btnStrengthStrong_ = {};

    // ── Editing source card (img2img — shown only with an input image) ────────
    if (!generationParams.initImagePath.empty()) {
        const bool hasVae = currentModelVaeEncoderAvailable();
        const float cardH = hasVae ? 32.f + 28.f + 34.f + 10.f : 32.f + 22.f + 10.f;
        const float top   = y;
        const float inner = drawCard(top, cardH, "EDITING SOURCE");

        const std::string stem =
            std::filesystem::path(generationParams.initImagePath).filename().string();
        const std::string truncated = stem.size() > 20 ? stem.substr(0, 18) + "\xe2\x80\xa6" : stem;
        btnClearInit_ = {cardX + cardW - 30.f, top + 9.f, 20.f, 18.f};
        drawButton(win, btnClearInit_, "\xc3\x97", colors.panel, colors.muted, false, 12, font);
        drawTextR(win, font, truncated, colors.blueLt, btnClearInit_.left - 8.f, top + 11.f, 10);

        if (hasVae) {
            const sf::Color subtleCol = generationParams.strength <= 0.35f ? colors.goldLt : colors.text;
            const sf::Color mediumCol = (generationParams.strength > 0.35f && generationParams.strength < 0.7f)
                ? colors.goldLt : colors.text;
            const sf::Color strongCol = generationParams.strength >= 0.7f ? colors.goldLt : colors.text;
            const float segW = (cinW - 16.f) / 3.f;
            btnStrengthSubtle_ = {cinX,                       inner, segW, 24.f};
            btnStrengthMedium_ = {cinX + segW + 8.f,          inner, segW, 24.f};
            btnStrengthStrong_ = {cinX + (segW + 8.f) * 2.f,  inner, segW, 24.f};
            drawButton(win, btnStrengthSubtle_, "Subtle", colors.surfaceInset, subtleCol, false, 11, font);
            drawButton(win, btnStrengthMedium_, "Medium", colors.surfaceInset, mediumCol, false, 11, font);
            drawButton(win, btnStrengthStrong_, "Strong", colors.surfaceInset, strongCol, false, 11, font);

            strengthSliderTrack_ = {cinX, inner + 28.f + 14.f, cinW, sliderH};
            const float strengthNorm = (generationParams.strength - 0.05f) / 0.95f;
            char strBuf[8]; std::snprintf(strBuf, sizeof(strBuf), "%.2f", generationParams.strength);
            drawSlider(win, font, strengthSliderTrack_, std::clamp(strengthNorm, 0.f, 1.f), "Strength", strBuf);
        } else {
            btnStrengthSubtle_ = {}; btnStrengthMedium_ = {}; btnStrengthStrong_ = {};
            strengthSliderTrack_ = {};
            drawText(win, font, "img2img not supported by this model", colors.muted, cinX, inner + 4.f, 11);
        }
        y = top + cardH + cardGap;
    } else {
        btnClearInit_        = {};
        strengthSliderTrack_ = {};
    }

    // ── Sampling card ─────────────────────────────────────────────────────────
    {
        const float top   = y;
        const float inner = drawCard(top, 32.f + 34.f * 3.f + 8.f, "SAMPLING");
        float sy = inner + 14.f;

        stepsSliderTrack_ = {cinX, sy, cinW, sliderH};
        drawSlider(win, font, stepsSliderTrack_,
                   std::clamp((generationParams.numSteps - 5.f) / 45.f, 0.f, 1.f),
                   "Steps", std::to_string(generationParams.numSteps));
        sy += 34.f;

        cfgSliderTrack_ = {cinX, sy, cinW, sliderH};
        char cfgBuf[16]; std::snprintf(cfgBuf, sizeof(cfgBuf), "%.1f", generationParams.guidanceScale);
        drawSlider(win, font, cfgSliderTrack_,
                   std::clamp((generationParams.guidanceScale - 1.0f) / 19.0f, 0.f, 1.f), "CFG scale", cfgBuf);
        sy += 34.f;

        imagesSliderTrack_ = {cinX, sy, cinW, sliderH};
        drawSlider(win, font, imagesSliderTrack_,
                   std::clamp((generationParams.numImages - 1.f) / 9.f, 0.f, 1.f),
                   "Images", std::to_string(generationParams.numImages));
        y = top + 32.f + 34.f * 3.f + 8.f + cardGap;
    }

    // ── Hires fix card ────────────────────────────────────────────────────────
    // Gate mirrors the LoRA pattern: hiresCapable (derived from the model's graphs
    // at import) → live controls; a model lacking the flag → disabled + re-import
    // hint. Scale max is per-arch — SDXL capped at kSdxlMaxHiresScale (VRAM ceiling).
    if (currentModel() != nullptr) {
        const bool  capable  = currentModelHiresCapable();
        const bool  isXL     = currentModelType() == ModelType::SDXL;
        const float scaleMax = isXL ? kSdxlMaxHiresScale : 2.0f;
        auto&       hires    = generationParams.hires;
        const bool  on       = capable && hires.enabled;

        const float cardH = on ? 32.f + 24.f + 8.f + 34.f + 34.f + 30.f + 8.f
                               : 32.f + 24.f + 8.f;
        const float top   = y;
        const float inner = drawCard(top, cardH, "HIRES FIX");

        // Whole row is the hit target; switch pill sits on the left.
        btnHiresToggle_ = {cinX, inner, cinW, 22.f};
        drawSwitch({cinX, inner, 38.f, 21.f}, on, capable);
        drawText(win, font, "Upscale after denoise", capable ? colors.text : colors.muted,
                 cinX + 48.f, inner + 4.f, 12);
        if (!capable)
            drawTextR(win, font, "re-import to enable", colors.muted, cinX + cinW, inner + 5.f, 10);

        if (on) {
            float hy = inner + 24.f + 8.f + 14.f;
            const float shownScale = std::min(hires.scale, scaleMax);
            hiresScaleSliderTrack_ = {cinX, hy, cinW, sliderH};
            char scBuf[16]; std::snprintf(scBuf, sizeof(scBuf), "%.1f\xc3\x97", shownScale);
            drawSlider(win, font, hiresScaleSliderTrack_,
                       std::clamp((shownScale - 1.0f) / (scaleMax - 1.0f), 0.f, 1.f), "Hires scale", scBuf);
            hy += 34.f;

            hiresStrengthSliderTrack_ = {cinX, hy, cinW, sliderH};
            char hsBuf[8]; std::snprintf(hsBuf, sizeof(hsBuf), "%.2f", hires.strength);
            drawSlider(win, font, hiresStrengthSliderTrack_,
                       std::clamp((hires.strength - 0.3f) / 0.4f, 0.f, 1.f), "Hires strength", hsBuf);
            hy += 30.f;

            const bool pixelCapable = currentModelPixelHiresCapable();
            const UpscaleMode effMode = pixelCapable ? hires.mode : UpscaleMode::Latent;
            drawText(win, font, "Upscale", colors.muted, cinX, hy + 5.f, 10);
            btnHiresModePixel_  = {cinX + 62.f,  hy, 64.f, 22.f};
            btnHiresModeLatent_ = {cinX + 132.f, hy, 64.f, 22.f};
            const sf::Color pixelCol  = !pixelCapable ? colors.muted
                                        : (effMode == UpscaleMode::Pixel ? colors.goldLt : colors.text);
            const sf::Color latentCol = effMode == UpscaleMode::Latent ? colors.goldLt : colors.text;
            drawButton(win, btnHiresModePixel_,  "Pixel",  colors.surfaceInset, pixelCol,  !pixelCapable, 11, font);
            drawButton(win, btnHiresModeLatent_, "Latent", colors.surfaceInset, latentCol, false, 11, font);
            if (!pixelCapable)
                drawTextR(win, font, "re-import for Pixel", colors.muted, cinX + cinW, hy + 5.f, 10);
        } else {
            hiresScaleSliderTrack_    = {};
            hiresStrengthSliderTrack_ = {};
            btnHiresModePixel_        = {};
            btnHiresModeLatent_       = {};
        }
        y = top + cardH + cardGap;
    } else {
        btnHiresToggle_           = {};
        hiresScaleSliderTrack_    = {};
        hiresStrengthSliderTrack_ = {};
        btnHiresModePixel_        = {};
        btnHiresModeLatent_       = {};
    }

    // ── Seed card ─────────────────────────────────────────────────────────────
    {
        const float top   = y;
        const float inner = drawCard(top, 32.f + 30.f + 10.f, "SEED");
        seedField_ = {cinX, inner, cinW, 30.f};
        drawSingleLineField(win, font, seedField_, seedInput, seedInputCursor, seedInputActive);
        if (seedInput.empty() && !seedInputActive)
            drawText(win, font, "empty = random", colors.muted, cinX + 8.f, inner + 8.f, 12);
        y = top + 32.f + 30.f + 10.f + cardGap;
    }

    // ── Primary action: Generate / Edit Image (pinned to the rail bottom) ──────
    {
        constexpr float genH = 44.f;
        const float genY = std::max(y, rect_.top + rect_.height - genH - pad);
        btnGenerate_ = {cardX, genY, cardW, genH};
        const char* label = generationParams.initImagePath.empty() ? "Generate" : "Edit Image";
        drawButton(win, btnGenerate_, label, colors.blue, colors.goldLt, false, 14, font);
    }

    // ── Model dropdown list (overlay, drawn last so it renders on top) ────────
    if (showModelDropdown && !models.empty()) {
        constexpr float itemH   = 22.f;
        constexpr float listPad = 2.f;
        const float listX = btnModelDropdown_.left;
        const float listW = btnModelDropdown_.width;
        const float listY = btnModelDropdown_.top + btnModelDropdown_.height + 2.f;
        const int   count = static_cast<int>(models.size());
        const float listH = listPad * 2.f + itemH * static_cast<float>(count);

        drawRect(win, {listX, listY, listW, listH}, colors.panel2, colors.borderHi, 1.f);
        modelDropdownItems_.resize(static_cast<size_t>(count));
        for (int i = 0; i < count; ++i) {
            const float itemY2 = listY + listPad + static_cast<float>(i) * itemH;
            modelDropdownItems_[static_cast<size_t>(i)] = {listX, itemY2, listW, itemH};
            const bool selected = (i == selectedModelIdx);
            if (selected)
                drawRect(win, modelDropdownItems_[static_cast<size_t>(i)], colors.panel);
            const ModelEntry& entry = models[static_cast<size_t>(i)];
            drawText(win, font, entry.displayName, selected ? colors.goldLt : colors.text,
                     listX + 6.f, itemY2 + 4.f, 12);
            const std::string badge = entry.type == ModelType::SDXL ? "SDXL" : "SD1.5";
            drawTextR(win, font, badge, selected ? colors.goldLt : colors.muted,
                      listX + listW - 8.f, itemY2 + 5.f, 11);
        }
    }

    // ── LoRA panel (overlay) ──────────────────────────────────────────────────
    if (showLoraPanel) {
        constexpr float panelW  = 340.f;
        constexpr float rowH    = 24.f;
        constexpr float scaleW  = 50.f;
        constexpr float listPad = 4.f;
        const float panelX = btnLoraPanel_.left;
        const float panelY = btnLoraPanel_.top + btnLoraPanel_.height + 2.f;
        const int   count  = static_cast<int>(availableLoras.size());

        if (count == 0) {
            drawRect(win, {panelX, panelY, panelW, 30.f}, colors.panel2, colors.borderHi, 1.f);
            drawText(win, font, "No .safetensors found in LoRA dir",
                     colors.muted, panelX + 8.f, panelY + 8.f, 11);
        } else {
            const float panelH = listPad * 2.f + rowH * static_cast<float>(count);
            drawRect(win, {panelX, panelY, panelW, panelH}, colors.panel2, colors.borderHi, 1.f);

            loraRowToggleRects_.resize(static_cast<size_t>(count));
            loraScaleRects_.resize(static_cast<size_t>(count));

            for (int i = 0; i < count; ++i) {
                const float rowY2 = panelY + listPad + static_cast<float>(i) * rowH;
                const bool  sel   = (i < static_cast<int>(loraSelected.size())) && loraSelected[static_cast<size_t>(i)];

                if (sel)
                    drawRect(win, {panelX + 1.f, rowY2, panelW - 2.f, rowH}, colors.panel);

                constexpr float cbSize = 10.f;
                const float cbX = panelX + 8.f;
                const float cbY = rowY2 + (rowH - cbSize) / 2.f;
                drawRect(win, {cbX, cbY, cbSize, cbSize}, sel ? colors.goldLt : colors.panel2, colors.border, 1.f);

                const std::string name = std::filesystem::path(
                    availableLoras[static_cast<size_t>(i)]).stem().string();
                drawText(win, font, name, sel ? colors.goldLt : colors.text,
                         cbX + cbSize + 6.f, rowY2 + 5.f, 11);

                loraRowToggleRects_[static_cast<size_t>(i)] = {
                    panelX, rowY2, panelW - scaleW - 4.f, rowH};

                const float scaleX = panelX + panelW - scaleW - 4.f;
                loraScaleRects_[static_cast<size_t>(i)] = {scaleX, rowY2 + 2.f, scaleW, rowH - 4.f};
                const bool scaleActive = (activeLoraScaleIdx == i);
                const std::string scaleText = (i < static_cast<int>(loraScaleInputs.size()))
                    ? loraScaleInputs[static_cast<size_t>(i)] : "1";
                drawSingleLineField(win, font, loraScaleRects_[static_cast<size_t>(i)],
                                    scaleText, scaleActive ? static_cast<int>(scaleText.size()) : 0, scaleActive);
            }
        }
    }

    (void)y; // suppress unused-variable warning
}

// ── Event handling ────────────────────────────────────────────────────────────

bool SettingsPanel::handleClick(sf::Vector2f pos) {
    // Model dropdown (check open list first — it renders on top)
    if (btnModelDropdown_.contains(pos)) {
        showModelDropdown = !showModelDropdown;
        return true;
    }
    if (showModelDropdown) {
        for (int i = 0; i < static_cast<int>(modelDropdownItems_.size()); ++i) {
            if (modelDropdownItems_[static_cast<size_t>(i)].contains(pos)) {
                selectedModelIdx  = i;
                showModelDropdown = false;
                return true;
            }
        }
        showModelDropdown = false; // click outside closes
        return true;
    }

    // LoRA panel
    if (btnLoraPanel_.contains(pos)) {
        showLoraPanel = !showLoraPanel;
        activeLoraScaleIdx = -1;
        return true;
    }
    if (showLoraPanel) {
        for (int i = 0; i < static_cast<int>(loraScaleRects_.size()); ++i) {
            if (loraScaleRects_[static_cast<size_t>(i)].contains(pos)) {
                activeLoraScaleIdx = i;
                seedInputActive    = false;
                return true;
            }
        }
        for (int i = 0; i < static_cast<int>(loraRowToggleRects_.size()); ++i) {
            if (loraRowToggleRects_[static_cast<size_t>(i)].contains(pos)) {
                if (i < static_cast<int>(loraSelected.size()))
                    loraSelected[static_cast<size_t>(i)] = !loraSelected[static_cast<size_t>(i)];
                activeLoraScaleIdx = -1;
                return true;
            }
        }
    }

    // Primary action button
    if (btnGenerate_.contains(pos)) { generateRequested = true; return true; }

    // Slider tracks — initiate drag
    if (stepsSliderTrack_.contains(pos))    { draggingSlider_ = DraggingSlider::Steps;    return true; }
    if (cfgSliderTrack_.contains(pos))      { draggingSlider_ = DraggingSlider::Cfg;      return true; }
    if (imagesSliderTrack_.contains(pos))   { draggingSlider_ = DraggingSlider::Images;   return true; }
    if (strengthSliderTrack_.contains(pos)) { draggingSlider_ = DraggingSlider::Strength; return true; }
    if (hiresScaleSliderTrack_.contains(pos))    { draggingSlider_ = DraggingSlider::HiresScale;    return true; }
    if (hiresStrengthSliderTrack_.contains(pos)) { draggingSlider_ = DraggingSlider::HiresStrength; return true; }

    // Hires-fix toggle (checkbox). Only interactive when the model is hires-capable.
    if (btnHiresToggle_.contains(pos)) {
        if (currentModelHiresCapable())
            generationParams.hires.enabled = !generationParams.hires.enabled;
        return true;
    }

    // Upscale-mode toggle. Pixel is selectable only on a pixel-hires-capable model
    // (dynamic VAE encoder); Latent is always available. Consumes the click either
    // way so a disabled Pixel press is a no-op, not a fall-through.
    if (btnHiresModePixel_.contains(pos)) {
        if (currentModelPixelHiresCapable())
            generationParams.hires.mode = UpscaleMode::Pixel;
        return true;
    }
    if (btnHiresModeLatent_.contains(pos)) {
        generationParams.hires.mode = UpscaleMode::Latent;
        return true;
    }

    // Clear init image — return to plain text-to-image
    if (btnClearInit_.contains(pos)) {
        generationParams.initImagePath.clear();
        return true;
    }
    if (btnStrengthSubtle_.contains(pos)) {
        generationParams.strength = 0.25f;
        return true;
    }
    if (btnStrengthMedium_.contains(pos)) {
        generationParams.strength = 0.5f;
        return true;
    }
    if (btnStrengthStrong_.contains(pos)) {
        generationParams.strength = 0.8f;
        return true;
    }

    // Text field focus
    if (positiveArea.getRect().contains(pos)) {
        positiveArea.handleClick(pos);
        negativeArea.setActive(false);
        seedInputActive = false;
        return true;
    }
    if (negativeArea.getRect().contains(pos)) {
        negativeArea.handleClick(pos);
        positiveArea.setActive(false);
        seedInputActive = false;
        return true;
    }
    if (seedField_.contains(pos)) {
        seedInputActive = true;
        positiveArea.setActive(false);
        negativeArea.setActive(false);
        return true;
    }

    return false;
}

bool SettingsPanel::handleEvent(const sf::Event& e) {
    // Mouse button released — stop slider drag
    if (e.type == sf::Event::MouseButtonReleased && e.mouseButton.button == sf::Mouse::Left) {
        draggingSlider_ = DraggingSlider::None;
        // don't consume — other panels may need release too
    }

    // Mouse moved — update dragged slider
    if (e.type == sf::Event::MouseMoved && draggingSlider_ != DraggingSlider::None) {
        const sf::Vector2f mousePos{static_cast<float>(e.mouseMove.x),
                                    static_cast<float>(e.mouseMove.y)};
        if (draggingSlider_ == DraggingSlider::Steps) {
            const float t = std::clamp((mousePos.x - stepsSliderTrack_.left) / stepsSliderTrack_.width, 0.f, 1.f);
            generationParams.numSteps = static_cast<int>(std::round(5.f + t * 45.f));
        } else if (draggingSlider_ == DraggingSlider::Cfg) {
            const float t   = std::clamp((mousePos.x - cfgSliderTrack_.left) / cfgSliderTrack_.width, 0.f, 1.f);
            const float raw = 1.0f + t * 19.0f;
            generationParams.guidanceScale = std::round(raw * 2.f) / 2.f;
        } else if (draggingSlider_ == DraggingSlider::Images) {
            const float t = std::clamp((mousePos.x - imagesSliderTrack_.left) / imagesSliderTrack_.width, 0.f, 1.f);
            generationParams.numImages = static_cast<int>(std::round(1.f + t * 9.f));
        } else if (draggingSlider_ == DraggingSlider::Strength) {
            const float t = std::clamp((mousePos.x - strengthSliderTrack_.left) / strengthSliderTrack_.width, 0.f, 1.f);
            const float raw = 0.05f + t * 0.95f;
            generationParams.strength = std::round(raw / 0.05f) * 0.05f;
        } else if (draggingSlider_ == DraggingSlider::HiresScale) {
            const float t = std::clamp((mousePos.x - hiresScaleSliderTrack_.left) / hiresScaleSliderTrack_.width, 0.f, 1.f);
            // Per-arch max: SDXL capped at kSdxlMaxHiresScale (VRAM), SD1.5 up to 2.0.
            const float scaleMax = currentModelType() == ModelType::SDXL ? kSdxlMaxHiresScale : 2.0f;
            const float raw = 1.0f + t * (scaleMax - 1.0f);
            generationParams.hires.scale = std::round(raw / 0.1f) * 0.1f;
        } else if (draggingSlider_ == DraggingSlider::HiresStrength) {
            const float t = std::clamp((mousePos.x - hiresStrengthSliderTrack_.left) / hiresStrengthSliderTrack_.width, 0.f, 1.f);
            const float raw = 0.3f + t * 0.4f;                 // 0.3–0.7
            generationParams.hires.strength = std::round(raw / 0.05f) * 0.05f;
        }
        return true;
    }

    // Mouse wheel scroll on prompt areas
    if (e.type == sf::Event::MouseWheelScrolled) {
        const sf::Vector2f pos{static_cast<float>(e.mouseWheelScroll.x),
                               static_cast<float>(e.mouseWheelScroll.y)};
        const float delta = e.mouseWheelScroll.delta > 0 ? -1.f : 1.f;
        if (positiveArea.getRect().contains(pos)) { positiveArea.handleScroll(delta); return true; }
        if (negativeArea.getRect().contains(pos)) { negativeArea.handleScroll(delta); return true; }
    }

    // Mouse click
    if (e.type == sf::Event::MouseButtonPressed && e.mouseButton.button == sf::Mouse::Left) {
        const sf::Vector2f pos{static_cast<float>(e.mouseButton.x),
                               static_cast<float>(e.mouseButton.y)};
        if (!rect_.contains(pos)) return false;
        return handleClick(pos);
    }

    // Keyboard / text input
    if (activeLoraScaleIdx >= 0) {
        const size_t idx = static_cast<size_t>(activeLoraScaleIdx);
        if (e.type == sf::Event::KeyPressed) {
            auto& s = loraScaleInputs[idx];
            if (e.key.code == sf::Keyboard::Escape || e.key.code == sf::Keyboard::Tab
                || e.key.code == sf::Keyboard::Enter) {
                activeLoraScaleIdx = -1;
            } else if (e.key.code == sf::Keyboard::BackSpace && !s.empty()) {
                s.pop_back();
                try { loraScales[idx] = std::stof(s); } catch (...) {}
            } else if (e.key.code == sf::Keyboard::Delete) {
                s.clear(); loraScales[idx] = 0.f;
            }
            return true;
        }
        if (e.type == sf::Event::TextEntered) {
            const auto ch = static_cast<char>(e.text.unicode);
            const bool isDigit = (ch >= '0' && ch <= '9');
            const bool isDot   = (ch == '.' && loraScaleInputs[idx].find('.') == std::string::npos);
            if ((isDigit || isDot) && loraScaleInputs[idx].size() < 8) {
                loraScaleInputs[idx] += ch;
                try { loraScales[idx] = std::stof(loraScaleInputs[idx]); } catch (...) {}
            }
            return true;
        }
        return false;
    }

    if (seedInputActive) {
        auto& s = seedInput;
        auto& c = seedInputCursor;
        if (e.type == sf::Event::KeyPressed) {
            if (e.key.code == sf::Keyboard::Left  && c > 0) { --c; }
            else if (e.key.code == sf::Keyboard::Right && c < static_cast<int>(s.size())) { ++c; }
            else if (e.key.code == sf::Keyboard::Home) { c = 0; }
            else if (e.key.code == sf::Keyboard::End)  { c = static_cast<int>(s.size()); }
            else if (e.key.code == sf::Keyboard::BackSpace && c > 0) { s.erase(static_cast<size_t>(--c), 1); }
            else if (e.key.code == sf::Keyboard::Delete && c < static_cast<int>(s.size())) { s.erase(static_cast<size_t>(c), 1); }
            else if (e.key.code == sf::Keyboard::Escape || e.key.code == sf::Keyboard::Tab) {
                seedInputActive = false;
            } else if (e.key.control && e.key.code == sf::Keyboard::C) {
                sf::Clipboard::setString(s);
            } else if (e.key.control && e.key.code == sf::Keyboard::V) {
                const std::string clip = sf::Clipboard::getString().toAnsiString();
                for (size_t i = 0; i < clip.size() && s.size() < 20; ++i) {
                    const char ch   = clip[i];
                    const bool dig  = (ch >= '0' && ch <= '9');
                    const bool neg  = (ch == '-' && c == 0 && s.empty());
                    if (dig || neg) { s.insert(static_cast<size_t>(c), 1, ch); ++c; }
                }
            }
            return true;
        }
        if (e.type == sf::Event::TextEntered) {
            const auto ch   = e.text.unicode;
            const bool dig  = (ch >= '0' && ch <= '9');
            const bool neg  = (ch == '-' && seedInputCursor == 0 && seedInput.empty());
            if ((dig || neg) && seedInput.size() < 20) {
                seedInput.insert(static_cast<size_t>(seedInputCursor), 1, static_cast<char>(ch));
                ++seedInputCursor;
            }
            return true;
        }
        return false;
    }

    // Tab: cycle focus between the positive and negative prompt fields.
    if (e.type == sf::Event::KeyPressed && e.key.code == sf::Keyboard::Tab) {
        if (positiveArea.isActive()) {
            positiveArea.setActive(false);
            negativeArea.setActive(true);
        } else if (negativeArea.isActive()) {
            negativeArea.setActive(false);
            positiveArea.setActive(true);
        }
        return true;
    }

    // Delegate text input to the active prompt field
    if (positiveArea.isActive() && positiveArea.handleEvent(e)) return true;
    if (negativeArea.isActive() && negativeArea.handleEvent(e)) return true;

    return false;
}
