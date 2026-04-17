#include "MenuBar.hpp"
#include "../../enum/constants.hpp"
#include "../../ui/Buttons.hpp"
#include "../../ui/Helpers.hpp"
#include <algorithm>

using namespace Helpers;

void MenuBar::setRect(const sf::FloatRect& rect) {
    rect_ = rect;
}

void MenuBar::setPresets(const std::vector<Preset>& presets, const std::string& activePresetId) {
    presets_       = presets;
    activePresetId_ = activePresetId;
}

// ── Draw helpers ──────────────────────────────────────────────────────────────

void MenuBar::drawSingleLineField(sf::RenderWindow& win, sf::Font& font,
                                   const sf::FloatRect& field,
                                   const std::string& text,
                                   int cursor, bool active) {
    drawRect(win, field, Col::Panel, active ? Col::GoldLt : Col::Border, 1.f);

    constexpr float    padX     = 6.f;
    constexpr unsigned fontSize = 13;
    const float textY  = field.top + (field.height - static_cast<float>(fontSize)) / 2.f - 1.f;
    const float maxW   = field.width - padX * 2.f;

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

void MenuBar::drawSaveModal(sf::RenderWindow& win, sf::Font& font) {
    // Dim background
    const float winW = static_cast<float>(WIN_W);
    const float winH = static_cast<float>(WIN_H);
    sf::RectangleShape overlay({winW, winH});
    overlay.setFillColor(Col::Overlay);
    win.draw(overlay);

    constexpr float boxW = 360.f, boxH = 130.f;
    const float boxX = (winW - boxW) / 2.f;
    const float boxY = (winH - boxH) / 2.f;
    drawRect(win, {boxX, boxY, boxW, boxH}, Col::Panel2, Col::BorderHi, 2.f);
    drawTextC(win, font, "Save preset as...", Col::GoldLt, winW / 2.f, boxY + 12.f, 13, true);

    constexpr float padX   = 16.f;
    constexpr float fieldH = 26.f;
    const float fieldY = boxY + 42.f;
    saveModalField_ = {boxX + padX, fieldY, boxW - padX * 2.f, fieldH};
    drawSingleLineField(win, font, saveModalField_, saveNameInput, saveNameCursor, true);

    constexpr float btnW = 80.f, btnH = 26.f;
    const float btnY = boxY + boxH - btnH - 12.f;
    saveModalCancel_ = {boxX + boxW - btnW * 2.f - padX - 4.f, btnY, btnW, btnH};
    saveModalOk_     = {boxX + boxW - btnW - padX,              btnY, btnW, btnH};
    drawButton(win, saveModalCancel_, "Cancel", Col::Panel2, Col::Muted,  false, 12, font);
    drawButton(win, saveModalOk_,     "Save",   Col::Panel,  Col::GoldLt, false, 12, font);
}

// ── Render ────────────────────────────────────────────────────────────────────

void MenuBar::render(sf::RenderWindow& win, sf::Font& font) {
    const float x  = rect_.left;
    const float y  = rect_.top;
    const float w  = rect_.width;
    const float h  = rect_.height;
    const float cx = x + w / 2.f;

    // Background + bottom border
    drawRect(win, rect_, Col::Panel2);
    drawRect(win, {x, y + h - 1.f, w, 1.f}, Col::Border);

    constexpr float pad = static_cast<float>(PAD);

    // Back button
    btnBack_ = {x + pad, y + (h - 26.f) / 2.f, 70.f, 26.f};
    drawButton(win, btnBack_, "< Back", Col::Panel, Col::Muted, false, 12, font);

    // Title
    drawTextC(win, font, "Image Generator", Col::GoldLt, cx, y + (h - 18.f) / 2.f, 16, true);

    // Presets button (right after Back)
    const std::string presetsLabel = "Presets " + std::string(showPresetDropdown ? "^" : "v");
    btnPresets_ = {btnBack_.left + btnBack_.width + 6.f, y + (h - 26.f) / 2.f, 110.f, 26.f};
    drawButton(win, btnPresets_, presetsLabel, Col::Panel,
               showPresetDropdown ? Col::GoldLt : Col::Muted, false, 12, font);

    // Settings button
    btnSettings_ = {x + w - pad - 80.f, y + (h - 26.f) / 2.f, 80.f, 26.f};
    drawButton(win, btnSettings_, "Settings", Col::Panel, Col::Muted, false, 12, font);

    // ── Preset dropdown (overlay) ─────────────────────────────────────────────
    if (showPresetDropdown) {
        constexpr float itemH   = 24.f;
        constexpr float listPad = 2.f;
        const float listX = btnPresets_.left;
        const float listY = btnPresets_.top + btnPresets_.height + 2.f;
        const float listW = 220.f;

        const int   count  = static_cast<int>(presets_.size());
        const bool  hasActive = !activePresetId_.empty();
        // 2 action items + separator + preset list
        const float listH = listPad * 2.f + itemH * 2.f + 5.f + itemH * static_cast<float>(count);
        drawRect(win, {listX, listY, listW, listH}, Col::Panel2, Col::BorderHi, 1.f);

        float iy = listY + listPad;

        // "Save" — overwrite active preset (greyed when none active)
        dropdownSaveItem_ = {listX, iy, listW, itemH};
        drawText(win, font, "Save",
                 hasActive ? Col::Text : Col::Muted, listX + 8.f, iy + 5.f, 12);
        iy += itemH;

        // "Save As..."
        dropdownSaveAsItem_ = {listX, iy, listW, itemH};
        drawText(win, font, "Save As...", Col::Text, listX + 8.f, iy + 5.f, 12);
        iy += itemH;

        // Separator
        drawRect(win, {listX + 4.f, iy + 2.f, listW - 8.f, 1.f}, Col::Border);
        iy += 5.f;

        // Preset list (Load)
        presetDropdownItems_.resize(static_cast<size_t>(count));
        for (int i = 0; i < count; ++i) {
            presetDropdownItems_[static_cast<size_t>(i)] = {listX, iy, listW, itemH};
            const bool active = (presets_[static_cast<size_t>(i)].id == activePresetId_);
            if (active) drawRect(win, presetDropdownItems_[static_cast<size_t>(i)], Col::Panel);
            const std::string prefix = active ? "\u2713 " : "  ";
            drawText(win, font, prefix + presets_[static_cast<size_t>(i)].name,
                     active ? Col::GoldLt : Col::Text, listX + 8.f, iy + 5.f, 12);
            iy += itemH;
        }
    }

    // Save modal drawn on top of everything
    if (showSaveModal)
        drawSaveModal(win, font);
}

// ── Event handling ────────────────────────────────────────────────────────────

bool MenuBar::handleEvent(const sf::Event& e) {
    // Save modal intercepts all input while open
    if (showSaveModal) {
        if (e.type == sf::Event::KeyPressed) {
            switch (e.key.code) {
            case sf::Keyboard::Enter:
                if (!saveNameInput.empty()) saveConfirmed = true;
                showSaveModal = false;
                break;
            case sf::Keyboard::Escape:
                showSaveModal = false;
                saveNameInput.clear();
                saveNameCursor = 0;
                break;
            case sf::Keyboard::BackSpace:
                if (saveNameCursor > 0) {
                    saveNameInput.erase(static_cast<size_t>(--saveNameCursor), 1);
                }
                break;
            case sf::Keyboard::Delete:
                if (saveNameCursor < static_cast<int>(saveNameInput.size()))
                    saveNameInput.erase(static_cast<size_t>(saveNameCursor), 1);
                break;
            case sf::Keyboard::Left:
                if (saveNameCursor > 0) --saveNameCursor;
                break;
            case sf::Keyboard::Right:
                if (saveNameCursor < static_cast<int>(saveNameInput.size())) ++saveNameCursor;
                break;
            case sf::Keyboard::Home:
                saveNameCursor = 0;
                break;
            case sf::Keyboard::End:
                saveNameCursor = static_cast<int>(saveNameInput.size());
                break;
            default:
                break;
            }
            return true;
        }
        if (e.type == sf::Event::TextEntered) {
            const auto c = e.text.unicode;
            if (c >= 32 && c < 127 && saveNameInput.size() < 64) {
                saveNameInput.insert(static_cast<size_t>(saveNameCursor), 1, static_cast<char>(c));
                ++saveNameCursor;
            }
            return true;
        }
        if (e.type == sf::Event::MouseButtonPressed && e.mouseButton.button == sf::Mouse::Left) {
            const sf::Vector2f pos{static_cast<float>(e.mouseButton.x),
                                   static_cast<float>(e.mouseButton.y)};
            if (saveModalOk_.contains(pos)) {
                if (!saveNameInput.empty()) saveConfirmed = true;
                showSaveModal = false;
            } else if (saveModalCancel_.contains(pos)) {
                showSaveModal = false;
                saveNameInput.clear();
                saveNameCursor = 0;
            }
            return true;
        }
        return false;
    }

    if (e.type == sf::Event::MouseButtonPressed && e.mouseButton.button == sf::Mouse::Left) {
        const sf::Vector2f pos{static_cast<float>(e.mouseButton.x),
                               static_cast<float>(e.mouseButton.y)};

        // Dropdown item clicks
        if (showPresetDropdown) {
            if (dropdownSaveItem_.contains(pos)) {
                showPresetDropdown = false;
                if (!activePresetId_.empty())
                    saveCurrentRequested = true;
                return true;
            }
            if (dropdownSaveAsItem_.contains(pos)) {
                showPresetDropdown = false;
                showSaveModal      = true;
                saveNameInput.clear();
                saveNameCursor     = 0;
                return true;
            }
            for (int i = 0; i < static_cast<int>(presetDropdownItems_.size()); ++i) {
                if (presetDropdownItems_[static_cast<size_t>(i)].contains(pos)) {
                    selectedPresetId   = presets_[static_cast<size_t>(i)].id;
                    showPresetDropdown = false;
                    return true;
                }
            }
            showPresetDropdown = false;
            // Fall through — click may have been on a bar button
        }

        if (btnBack_.contains(pos))     { backRequested     = true; return true; }
        if (btnSettings_.contains(pos)) { settingsRequested = true; return true; }
        if (btnPresets_.contains(pos))  { showPresetDropdown = !showPresetDropdown; return true; }

        // Only consume clicks within the bar rect
        return rect_.contains(pos);
    }

    return false;
}
