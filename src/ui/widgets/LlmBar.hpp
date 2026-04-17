#pragma once
#include <atomic>
#include <string>
#include <SFML/Graphics.hpp>
#include "MultiLineTextArea.hpp"

// Bottom LLM bar (visible only when LLM is available or loading).
// Collapsed: 44px toggle strip. Expanded: overlay panel above the bar with
// the instruction textarea and Enhance button.
class LlmBar {
public:
    // ── State (written by controller each update()) ───────────────────────────
    bool promptEnhancerAvailable = false;
    bool llmLoading              = false;

    // ── Instruction field ─────────────────────────────────────────────────────
    MultiLineTextArea instructionArea{500, 2};
    bool expanded  = false;
    bool enhancing = false;

    // ── Enhancement result (written by background thread) ────────────────────
    std::atomic<bool> enhanceDone{false};
    std::string       enhancedPositive;
    std::string       enhancedNegative;

    // ── Action flag (cleared by controller after launching enhancement) ───────
    bool enhanceRequested = false;

    // ── Interface ─────────────────────────────────────────────────────────────
    void setRect(const sf::FloatRect& rect); // collapsed bar rect
    void render(sf::RenderWindow& win, sf::Font& font);

    // Returns true if the event was consumed.
    bool handleEvent(const sf::Event& e);

private:
    sf::FloatRect rect_;        // collapsed bar
    sf::FloatRect expandedRect_; // overlay panel (computed each render)

    sf::FloatRect btnToggle_;
    sf::FloatRect btnEnhance_;
};
