#include "SettingsPanel.hpp"
#include "../../enum/constants.hpp"
#include "../../ui/Buttons.hpp"
#include "../../ui/Helpers.hpp"
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
    if (availableModels.empty()) return "models";
    return availableModels[static_cast<size_t>(selectedModelIdx)];
}

// ── Draw helpers ──────────────────────────────────────────────────────────────

void SettingsPanel::drawSlider(sf::RenderWindow& win, sf::Font& font,
                               const sf::FloatRect& track, float normalised,
                               const std::string& label, const std::string& valueStr) {
    constexpr float visualH = 6.f;
    constexpr float thumbW  = 12.f;
    constexpr float thumbH  = 18.f;

    const float barY = track.top + (track.height - visualH) / 2.f;
    drawRect(win, {track.left, barY, track.width, visualH}, Col::Panel2, Col::Border, 1.f);
    if (normalised > 0.f)
        drawRect(win, {track.left, barY, track.width * normalised, visualH}, Col::Gold);

    const float thumbX = track.left + track.width * normalised - thumbW / 2.f;
    const float thumbY = track.top  + (track.height - thumbH)  / 2.f;
    drawRect(win, {thumbX, thumbY, thumbW, thumbH}, Col::GoldLt, Col::Border, 1.f);

    drawText(win,  font, label,    Col::Muted,  track.left,               track.top - 13.f, 11);
    drawTextR(win, font, valueStr, Col::GoldLt, track.left + track.width, track.top - 13.f, 11);
}

void SettingsPanel::drawSingleLineField(sf::RenderWindow& win, sf::Font& font,
                                        const sf::FloatRect& field,
                                        const std::string& text,
                                        int cursor, bool active) {
    drawRect(win, field, Col::Panel, active ? Col::GoldLt : Col::Border, 1.f);

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

// ── Render ────────────────────────────────────────────────────────────────────

void SettingsPanel::render(sf::RenderWindow& win, sf::Font& font) {
    const float x  = rect_.left;
    const float pw = rect_.width;       // panel width
    constexpr float pad  = static_cast<float>(PAD);
    const float fw = pw - pad * 2.f;    // field width inside padding

    // Panel background with right border
    drawRect(win, rect_, Col::Panel);
    drawRect(win, {x + pw - 1.f, rect_.top, 1.f, rect_.height}, Col::Border);

    float y = rect_.top + pad * 2.f;

    // Model selector + LoRA button
    {
        const std::string displayName = availableModels.empty()
            ? "(no models found)"
            : std::filesystem::path(availableModels[static_cast<size_t>(selectedModelIdx)]).filename().string();
        const std::string label = displayName + (showModelDropdown ? "  ^" : "  v");
        drawText(win, font, "Model:", Col::Muted, x + pad, y + 4.f, 12);
        btnModelDropdown_ = {x + pad + 52.f, y, 260.f, 22.f};
        drawButton(win, btnModelDropdown_, label, Col::Panel2,
                   showModelDropdown ? Col::GoldLt : Col::Text, false, 12, font);

        const int numSel = static_cast<int>(std::count(loraSelected.begin(), loraSelected.end(), true));
        const std::string loraLabel = "LoRA (" + std::to_string(numSel) + ")";
        btnLoraPanel_ = {x + pad + 324.f, y, 110.f, 22.f};
        drawButton(win, btnLoraPanel_, loraLabel, Col::Panel2,
                   showLoraPanel ? Col::GoldLt : Col::Text, false, 12, font);
    }
    y += 30.f;

    // Positive prompt
    drawText(win, font, "Positive prompt:", Col::Muted, x + pad, y, 12);
    y += 18.f;
    constexpr float fieldH    = 86.f;
    constexpr float fieldH_sm = 68.f;
    positiveArea.setRect({x + pad, y, fw, fieldH});
    positiveArea.render(win, font);
    y += fieldH + 4.f;

    // ── Token chips (Phase 8 — read-only DSL visualisation) ──────────────────
    if (currentDsl.subject || !currentDsl.positive.empty()) {
        constexpr float chipH    = 20.f;
        constexpr float chipPadX =  6.f;
        constexpr float chipGap  =  4.f;
        const float     startY   = y;
        float cx = x + pad;
        float cy = startY;
        int   rows = 1;

        sf::Text measure;
        measure.setFont(font);
        measure.setCharacterSize(11);

        struct Chip { std::string label; sf::Color border; sf::Color text; };
        std::vector<Chip> chips;

        if (currentDsl.subject) {
            const auto& subj = *currentDsl.subject;
            std::string lbl  = subj.value;
            if (std::abs(subj.weight - 1.0f) > 0.01f) {
                char buf[16];
                std::snprintf(buf, sizeof(buf), " %.2g\xc3\x97", subj.weight);
                lbl += buf;
            }
            chips.push_back({lbl, Col::Gold, Col::GoldLt});
        }

        for (const auto& tok : currentDsl.positive) {
            std::string lbl = tok.value;
            if (std::abs(tok.weight - 1.0f) > 0.01f) {
                char buf[16];
                std::snprintf(buf, sizeof(buf), " %.2g\xc3\x97", tok.weight); // × (UTF-8)
                lbl += buf;
            }
            const sf::Color border = (tok.weight > 1.f) ? Col::BlueLt
                                   : (tok.weight < 1.f) ? Col::Muted
                                                         : Col::Border;
            chips.push_back({lbl, border, Col::Text});
        }

        for (const auto& chip : chips) {
            measure.setString(chip.label);
            const float cw = measure.getLocalBounds().width + chipPadX * 2.f;
            if (cx + cw > x + pad + fw && cx > x + pad) {
                cy += chipH + 2.f;
                cx  = x + pad;
                if (++rows > 2) break;
            }
            drawRect(win, {cx, cy, cw, chipH}, Col::Panel2, chip.border, 1.f);
            drawText(win, font, chip.label, chip.text, cx + chipPadX, cy + 4.f, 11);
            cx += cw + chipGap;
        }
        y = startY + static_cast<float>(rows) * (chipH + 2.f) + 4.f;
    } else {
        y += pad * 2.f;
    }

    // Negative prompt
    drawText(win, font, "Negative prompt:", Col::Muted, x + pad, y, 12);
    y += 18.f;
    negativeArea.setRect({x + pad, y, fw, fieldH_sm});
    negativeArea.setTextColor(Col::Muted);
    negativeArea.render(win, font);
    y += fieldH_sm + 4.f;

    // ── Compiled preview (Phase 6 — SD1.5 only, shows model-adjusted output) ─
    if (!compiledPreview.empty()) {
        sf::Text measure;
        measure.setFont(font);
        measure.setCharacterSize(10);
        const float maxW = fw - 4.f;
        std::string txt = compiledPreview;
        measure.setString(txt);
        while (!txt.empty() && measure.getLocalBounds().width > maxW) {
            txt.pop_back();
            measure.setString(txt);
        }
        if (txt.size() < compiledPreview.size()) txt += "\xe2\x80\xa6"; // … (UTF-8)
        drawText(win, font, "\xe2\x86\x92 " + txt, Col::Muted, x + pad, y, 10); // → prefix
        y += 16.f;
    }

    y += pad * 2.f;

    // ── Sliders (always visible) ──────────────────────────────────────────────
    const float sliderW = fw;
    const float sliderH = 18.f;

    // Steps slider (range 5–50)
    stepsSliderTrack_ = {x + pad, y + 14.f, sliderW, sliderH};
    const float stepsNorm = (generationParams.numSteps - 5.f) / 45.f;
    drawSlider(win, font, stepsSliderTrack_, std::clamp(stepsNorm, 0.f, 1.f),
               "Steps", std::to_string(generationParams.numSteps));
    y += 34.f;

    // CFG Scale slider (range 1.0–20.0)
    cfgSliderTrack_ = {x + pad, y + 14.f, sliderW, sliderH};
    const float cfgNorm = (generationParams.guidanceScale - 1.0f) / 19.0f;
    char cfgBuf[16];
    std::snprintf(cfgBuf, sizeof(cfgBuf), "%.1f", generationParams.guidanceScale);
    drawSlider(win, font, cfgSliderTrack_, std::clamp(cfgNorm, 0.f, 1.f), "CFG Scale", cfgBuf);
    y += 34.f;

    // Images slider (range 1–10)
    imagesSliderTrack_ = {x + pad, y + 14.f, sliderW, sliderH};
    const float imagesNorm = (generationParams.numImages - 1.f) / 9.f;
    drawSlider(win, font, imagesSliderTrack_, std::clamp(imagesNorm, 0.f, 1.f),
               "Images", std::to_string(generationParams.numImages));
    y += 34.f;

    // ── Img2img (only when init image is set) ────────────────────────────────
    if (!generationParams.initImagePath.empty()) {
        // Info row: filename + Clear button
        const std::string stem =
            std::filesystem::path(generationParams.initImagePath).filename().string();
        const std::string truncated = stem.size() > 28 ? stem.substr(0, 26) + "\xe2\x80\xa6" : stem;
        drawText(win, font, "Init: " + truncated, Col::BlueLt, x + pad, y + 6.f, 11);
        btnClearInit_ = {x + pad + sliderW - 38.f, y + 2.f, 38.f, 18.f};
        drawButton(win, btnClearInit_, "Clear", Col::Panel2, Col::Muted, false, 10, font);
        y += 26.f;

        // Strength slider (range 0.05–1.0 step 0.05)
        strengthSliderTrack_ = {x + pad, y + 14.f, sliderW, sliderH};
        const float strengthNorm = (generationParams.strength - 0.05f) / 0.95f;
        char strBuf[8];
        std::snprintf(strBuf, sizeof(strBuf), "%.2f", generationParams.strength);
        drawSlider(win, font, strengthSliderTrack_, std::clamp(strengthNorm, 0.f, 1.f),
                   "Strength", strBuf);
        y += 34.f;
    } else {
        btnClearInit_        = {};
        strengthSliderTrack_ = {};
        y += 10.f;
    }

    // Seed input
    drawText(win, font, "Seed:", Col::Muted, x + pad, y + 6.f, 12);
    drawText(win, font, "(empty = random)", Col::Border, x + pad + 44.f, y + 6.f, 11);
    constexpr float seedFieldW = 160.f;
    constexpr float seedFieldH = 24.f;
    seedField_ = {x + pad + sliderW - seedFieldW, y, seedFieldW, seedFieldH};
    drawSingleLineField(win, font, seedField_, seedInput, seedInputCursor, seedInputActive);
    y += 34.f;

    // ── Model dropdown list (overlay, drawn last so it renders on top) ────────
    if (showModelDropdown && !availableModels.empty()) {
        constexpr float itemH   = 22.f;
        constexpr float listPad = 2.f;
        const float listX = btnModelDropdown_.left;
        const float listW = btnModelDropdown_.width;
        const float listY = btnModelDropdown_.top + btnModelDropdown_.height + 2.f;
        const int   count = static_cast<int>(availableModels.size());
        const float listH = listPad * 2.f + itemH * static_cast<float>(count);

        drawRect(win, {listX, listY, listW, listH}, Col::Panel2, Col::BorderHi, 1.f);
        modelDropdownItems_.resize(static_cast<size_t>(count));
        for (int i = 0; i < count; ++i) {
            const float itemY2 = listY + listPad + static_cast<float>(i) * itemH;
            modelDropdownItems_[static_cast<size_t>(i)] = {listX, itemY2, listW, itemH};
            const bool selected = (i == selectedModelIdx);
            if (selected)
                drawRect(win, modelDropdownItems_[static_cast<size_t>(i)], Col::Panel);
            const std::string name =
                std::filesystem::path(availableModels[static_cast<size_t>(i)]).filename().string();
            drawText(win, font, name, selected ? Col::GoldLt : Col::Text,
                     listX + 6.f, itemY2 + 4.f, 12);
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
            drawRect(win, {panelX, panelY, panelW, 30.f}, Col::Panel2, Col::BorderHi, 1.f);
            drawText(win, font, "No .safetensors found in LoRA dir",
                     Col::Muted, panelX + 8.f, panelY + 8.f, 11);
        } else {
            const float panelH = listPad * 2.f + rowH * static_cast<float>(count);
            drawRect(win, {panelX, panelY, panelW, panelH}, Col::Panel2, Col::BorderHi, 1.f);

            loraRowToggleRects_.resize(static_cast<size_t>(count));
            loraScaleRects_.resize(static_cast<size_t>(count));

            for (int i = 0; i < count; ++i) {
                const float rowY2 = panelY + listPad + static_cast<float>(i) * rowH;
                const bool  sel   = (i < static_cast<int>(loraSelected.size())) && loraSelected[static_cast<size_t>(i)];

                if (sel)
                    drawRect(win, {panelX + 1.f, rowY2, panelW - 2.f, rowH}, Col::Panel);

                constexpr float cbSize = 10.f;
                const float cbX = panelX + 8.f;
                const float cbY = rowY2 + (rowH - cbSize) / 2.f;
                drawRect(win, {cbX, cbY, cbSize, cbSize}, sel ? Col::GoldLt : Col::Panel2, Col::Border, 1.f);

                const std::string name = std::filesystem::path(
                    availableLoras[static_cast<size_t>(i)]).stem().string();
                drawText(win, font, name, sel ? Col::GoldLt : Col::Text,
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

    // Slider tracks — initiate drag
    if (stepsSliderTrack_.contains(pos))    { draggingSlider_ = DraggingSlider::Steps;    return true; }
    if (cfgSliderTrack_.contains(pos))      { draggingSlider_ = DraggingSlider::Cfg;      return true; }
    if (imagesSliderTrack_.contains(pos))   { draggingSlider_ = DraggingSlider::Images;   return true; }
    if (strengthSliderTrack_.contains(pos)) { draggingSlider_ = DraggingSlider::Strength; return true; }

    // Clear init image
    if (btnClearInit_.contains(pos)) {
        generationParams.initImagePath.clear();
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

    // Tab: cycle focus positive → negative → positive
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
