#pragma once
#include <string>
#include <vector>
#include <SFML/Graphics.hpp>
#include "../../presets/Preset.hpp"

// Top menu bar: Back button, title, Presets dropdown (with save modal), Settings button.
// Owns its own rendering and input handling for the bar region.
class MenuBar {
public:
    // ── Preset dropdown state ─────────────────────────────────────────────────
    bool showPresetDropdown = false;

    // ── Preset save modal state ───────────────────────────────────────────────
    bool        showSaveModal  = false;
    std::string saveNameInput;
    int         saveNameCursor = 0;
    bool        saveConfirmed  = false; // controller reads, clears, calls createFromGeneration

    // ── Action flags (set by handleEvent, cleared by controller) ──────────────
    bool        backRequested        = false;
    bool        settingsRequested    = false;
    bool        saveCurrentRequested = false; // "Save" clicked — overwrite active preset
    std::string selectedPresetId;             // non-empty when user picks a preset to load

    // ── Interface ─────────────────────────────────────────────────────────────
    void setRect(const sf::FloatRect& rect);
    void render(sf::RenderWindow& win, sf::Font& font);

    // Returns true if the event was consumed.
    bool handleEvent(const sf::Event& e);

    // Call whenever the preset list changes or active preset changes.
    void setPresets(const std::vector<Preset>& presets, const std::string& activePresetId);

private:
    sf::FloatRect rect_;

    // Preset list copy for rendering
    std::vector<Preset> presets_;
    std::string         activePresetId_;

    // Hit rects
    sf::FloatRect btnBack_;
    sf::FloatRect btnQuickSave_;
    sf::FloatRect btnSettings_;
    sf::FloatRect btnPresets_;
    sf::FloatRect              dropdownSaveItem_;     // "Save" (overwrite)
    sf::FloatRect              dropdownSaveAsItem_;   // "Save As..."
    std::vector<sf::FloatRect> presetDropdownItems_;  // preset list (load)

    // Save modal hit rects
    sf::FloatRect saveModalField_;
    sf::FloatRect saveModalOk_;
    sf::FloatRect saveModalCancel_;

    void drawSaveModal(sf::RenderWindow& win, sf::Font& font);
    void drawSingleLineField(sf::RenderWindow& win, sf::Font& font,
                             const sf::FloatRect& field,
                             const std::string& text,
                             int cursor, bool active);
};
