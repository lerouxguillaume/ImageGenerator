#pragma once
#include <atomic>
#include <string>
#include <SFML/Graphics.hpp>
#include "MultiLineTextArea.hpp"

// Bottom LLM bar (visible only when LLM is available or loading).
// Collapsed: 44px toggle strip. Expanded: bar grows by LLM_EXPANDED_H, showing
// the instruction textarea below the toggle row.
class LlmBar {
public:
    // ── State (written by controller each update()) ───────────────────────────
    bool promptEnhancerAvailable = false;
    bool llmLoading              = false;

    // ── Instruction field ─────────────────────────────────────────────────────
    MultiLineTextArea instructionArea{500, 2};
    bool expanded  = false;
    bool enhancing = false;

    // ── Enhancement result (snapshot at start, result owned by controller future) ──
    std::string originalPositive;
    std::string originalNegative;

    // ── Action flag (cleared by controller after launching enhancement) ───────
    bool enhanceRequested = false;

    // ── Interface ─────────────────────────────────────────────────────────────
    void setRect(const sf::FloatRect& rect); // collapsed bar rect
    void render(sf::RenderWindow& win, sf::Font& font);

    // Returns true if the event was consumed.
    bool handleEvent(const sf::Event& e);

private:
    sf::FloatRect rect_;
    sf::FloatRect btnToggle_;
    sf::FloatRect btnEnhance_;
};
