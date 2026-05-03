#pragma once
#include <string>
#include <SFML/Graphics.hpp>

// Settings modal overlay.
// Owns its own state, rendering, and keyboard/click handling.
// The controller opens it via open(), reads action flags, and calls save logic.
class SettingsModal {
public:
    // ── Editable fields (populated by open()) ────────────────────────────────
    std::string settingsOutputDir;
    std::string settingsLlmModelDir;
    std::string settingsLoraDir;

    int  settingsOutputDirCursor    = 0;
    int  settingsLlmModelDirCursor  = 0;
    int  settingsLoraDirCursor      = 0;

    bool settingsOutputDirActive    = true;
    bool settingsLlmModelDirActive  = false;
    bool settingsLoraDirActive      = false;

    // ── Action flags (read and cleared by controller) ─────────────────────────
    bool saveRequested   = false;
    bool cancelRequested = false;

    enum class BrowseTarget { None, OutputDir, LlmDir, LoraDir };
    BrowseTarget browseTarget = BrowseTarget::None;

    // llmLoading is displayed inside the modal when true
    bool llmLoading = false;

    // ── Interface ─────────────────────────────────────────────────────────────
    void render(sf::RenderWindow& win, sf::Font& font);

    // Returns true if the event was consumed by the modal.
    bool handleEvent(const sf::Event& e);

private:
    // Hit rects (written during render, read by handleEvent)
    sf::FloatRect settingsOutputDirField;
    sf::FloatRect settingsLlmModelDirField;
    sf::FloatRect settingsLoraDirField;

    sf::FloatRect settingsBtnBrowseOutput;
    sf::FloatRect settingsBtnBrowseLlm;
    sf::FloatRect settingsBtnBrowseLora;

    sf::FloatRect settingsBtnSave;
    sf::FloatRect settingsBtnCancel;

    void drawSingleLineField(sf::RenderWindow& win, sf::Font& font,
                             const sf::FloatRect& field,
                             const std::string& text,
                             int cursor, bool active);
};
