#include "SettingsModal.hpp"
#include "../../enum/constants.hpp"
#include "../../ui/Buttons.hpp"
#include "../../ui/Helpers.hpp"

using namespace Helpers;

void SettingsModal::drawSingleLineField(sf::RenderWindow& win, sf::Font& font,
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

void SettingsModal::render(sf::RenderWindow& win, sf::Font& font) {
    const float WIN_W_F = static_cast<float>(win.getSize().x);
    const float WIN_H_F = static_cast<float>(win.getSize().y);

    // Dim background
    sf::RectangleShape overlay({WIN_W_F, WIN_H_F});
    overlay.setFillColor(Col::Overlay);
    win.draw(overlay);

    // Modal panel
    constexpr float boxW = 560.f, boxH = 330.f;
    const float boxX = (WIN_W_F - boxW) / 2.f;
    const float boxY = (WIN_H_F - boxH) / 2.f;
    drawRect(win, {boxX, boxY, boxW, boxH}, Col::Panel2, Col::BorderHi, 2.f);
    drawTextC(win, font, "Settings", Col::GoldLt, WIN_W_F / 2.f, boxY + 14.f, 15, true);

    constexpr float padX      = 20.f;
    constexpr float labelW    = 140.f;
    constexpr float browseW   = 38.f;
    constexpr float browseGap = 4.f;
    constexpr float fieldW    = 318.f;
    constexpr float fieldH    = 26.f;
    const float     fieldX    = boxX + padX + labelW;

    // Row 1: model directory
    float rowY = boxY + 52.f;
    drawText(win, font, "Model directory:", Col::Muted, boxX + padX, rowY + 6.f, 12);
    settingsModelDirField = {fieldX, rowY, fieldW, fieldH};
    drawSingleLineField(win, font, settingsModelDirField, settingsModelDir,
                        settingsModelDirCursor, settingsModelDirActive);
    settingsBtnBrowseModel = {fieldX + fieldW + browseGap, rowY, browseW, fieldH};
    drawButton(win, settingsBtnBrowseModel, "...", Col::Panel2, Col::Muted, false, 12, font);

    // Row 2: output directory
    rowY += 48.f;
    drawText(win, font, "Output directory:", Col::Muted, boxX + padX, rowY + 6.f, 12);
    settingsOutputDirField = {fieldX, rowY, fieldW, fieldH};
    drawSingleLineField(win, font, settingsOutputDirField, settingsOutputDir,
                        settingsOutputDirCursor, settingsOutputDirActive);
    settingsBtnBrowseOutput = {fieldX + fieldW + browseGap, rowY, browseW, fieldH};
    drawButton(win, settingsBtnBrowseOutput, "...", Col::Panel2, Col::Muted, false, 12, font);

    // Row 3: LLM model directory
    rowY += 48.f;
    drawText(win, font, "LLM model dir:", Col::Muted, boxX + padX, rowY + 6.f, 12);
    settingsLlmModelDirField = {fieldX, rowY, fieldW, fieldH};
    drawSingleLineField(win, font, settingsLlmModelDirField, settingsLlmModelDir,
                        settingsLlmModelDirCursor, settingsLlmModelDirActive);
    settingsBtnBrowseLlm = {fieldX + fieldW + browseGap, rowY, browseW, fieldH};
    drawButton(win, settingsBtnBrowseLlm, "...", Col::Panel2, Col::Muted, false, 12, font);
    if (llmLoading)
        drawText(win, font, "Loading...", Col::Muted, fieldX, rowY + fieldH + 2.f, 10);

    // Row 4: LoRA directory
    rowY += 48.f;
    drawText(win, font, "LoRA directory:", Col::Muted, boxX + padX, rowY + 6.f, 12);
    settingsLoraDirField = {fieldX, rowY, fieldW, fieldH};
    drawSingleLineField(win, font, settingsLoraDirField, settingsLoraDir,
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

bool SettingsModal::handleEvent(const sf::Event& e) {
    // Determine which field/cursor pair is active
    std::string& activeField = settingsLoraDirActive     ? settingsLoraDir
                             : settingsLlmModelDirActive ? settingsLlmModelDir
                             : settingsModelDirActive    ? settingsModelDir
                                                         : settingsOutputDir;
    int& cursor = settingsLoraDirActive     ? settingsLoraDirCursor
                : settingsLlmModelDirActive ? settingsLlmModelDirCursor
                : settingsModelDirActive    ? settingsModelDirCursor
                                            : settingsOutputDirCursor;

    if (e.type == sf::Event::KeyPressed) {
        switch (e.key.code) {
        case sf::Keyboard::Tab:
            // Cycle: ModelDir → OutputDir → LlmDir → LoraDir → ModelDir
            if (settingsModelDirActive) {
                settingsModelDirActive = false; settingsOutputDirActive = true;
            } else if (settingsOutputDirActive) {
                settingsOutputDirActive = false; settingsLlmModelDirActive = true;
            } else if (settingsLlmModelDirActive) {
                settingsLlmModelDirActive = false; settingsLoraDirActive = true;
            } else {
                settingsLoraDirActive = false; settingsModelDirActive = true;
            }
            break;
        case sf::Keyboard::Enter:
            saveRequested = true;
            break;
        case sf::Keyboard::Escape:
            cancelRequested = true;
            break;
        case sf::Keyboard::Left:
            if (cursor > 0) --cursor;
            break;
        case sf::Keyboard::Right:
            if (cursor < static_cast<int>(activeField.size())) ++cursor;
            break;
        case sf::Keyboard::Home:
            cursor = 0;
            break;
        case sf::Keyboard::End:
            cursor = static_cast<int>(activeField.size());
            break;
        case sf::Keyboard::BackSpace:
            if (cursor > 0) { activeField.erase(static_cast<size_t>(--cursor), 1); }
            break;
        case sf::Keyboard::Delete:
            if (cursor < static_cast<int>(activeField.size()))
                activeField.erase(static_cast<size_t>(cursor), 1);
            break;
        default:
            break;
        }
        return true;
    }

    if (e.type == sf::Event::TextEntered) {
        const auto c = e.text.unicode;
        if (c >= 32 && c < 127) {
            activeField.insert(static_cast<size_t>(cursor), 1, static_cast<char>(c));
            ++cursor;
        }
        return true;
    }

    if (e.type == sf::Event::MouseButtonPressed && e.mouseButton.button == sf::Mouse::Left) {
        const sf::Vector2f pos{static_cast<float>(e.mouseButton.x),
                               static_cast<float>(e.mouseButton.y)};

        if (settingsBtnSave.contains(pos))   { saveRequested = true;   return true; }
        if (settingsBtnCancel.contains(pos)) { cancelRequested = true; return true; }

        if (settingsBtnBrowseModel.contains(pos))  { browseTarget = BrowseTarget::ModelDir;  return true; }
        if (settingsBtnBrowseOutput.contains(pos)) { browseTarget = BrowseTarget::OutputDir; return true; }
        if (settingsBtnBrowseLlm.contains(pos))    { browseTarget = BrowseTarget::LlmDir;    return true; }
        if (settingsBtnBrowseLora.contains(pos))   { browseTarget = BrowseTarget::LoraDir;   return true; }

        if (settingsModelDirField.contains(pos)) {
            settingsModelDirActive = true; settingsOutputDirActive = false;
            settingsLlmModelDirActive = false; settingsLoraDirActive = false;
            return true;
        }
        if (settingsOutputDirField.contains(pos)) {
            settingsModelDirActive = false; settingsOutputDirActive = true;
            settingsLlmModelDirActive = false; settingsLoraDirActive = false;
            return true;
        }
        if (settingsLlmModelDirField.contains(pos)) {
            settingsModelDirActive = false; settingsOutputDirActive = false;
            settingsLlmModelDirActive = true; settingsLoraDirActive = false;
            return true;
        }
        if (settingsLoraDirField.contains(pos)) {
            settingsModelDirActive = false; settingsOutputDirActive = false;
            settingsLlmModelDirActive = false; settingsLoraDirActive = true;
            return true;
        }
        return true; // absorb all clicks while modal is open
    }

    return false;
}
