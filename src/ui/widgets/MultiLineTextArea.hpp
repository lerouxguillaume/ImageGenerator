#pragma once
#include "base/Widget.h"
#include <SFML/Graphics/Color.hpp>
#include <SFML/System/Vector2.hpp>
#include <SFML/Window/Event.hpp>
#include <string>
#include <vector>

// Self-contained multi-line editable text area.
// Owns its text, cursor, word-wrap layout, scroll position, and focus state.
// The host controller calls handleEvent() / handleScroll() / handleClick() to
// forward input, and reads getText() / isActive() to query state.
class MultiLineTextArea : public Widget {
public:
    // charLimit: max characters allowed.
    // visibleLines: how many wrapped lines to show before scrolling (default 4).
    explicit MultiLineTextArea(int charLimit = 2000, int visibleLines = 4);

    // ── Widget interface ──────────────────────────────────────────────────────
    void          render(sf::RenderWindow& win, sf::Font& font) override;
    void          setRect(const sf::FloatRect& rect) override;
    sf::FloatRect getRect() const;

    // ── Input ─────────────────────────────────────────────────────────────────
    // Returns true if the event was consumed (so the caller can skip further routing).
    bool handleEvent(const sf::Event& e);
    // delta > 0 → scroll down, delta < 0 → scroll up (by |delta| lines).
    void handleScroll(float delta);
    // Sets the field active when pos is inside the rect; no-op otherwise.
    void handleClick(sf::Vector2f pos);

    // ── State ─────────────────────────────────────────────────────────────────
    void               setText(const std::string& t);
    const std::string& getText() const;
    bool               isActive() const;
    void               setActive(bool active);
    void               setTextColor(sf::Color c);

private:
    struct VisualLine { int start, end; }; // [start, end) byte range in text_

    // Compute pixel-accurate word-wrapped line layout for the current text.
    // Requires a font reference for glyph metrics.
    std::vector<VisualLine> computeLines(sf::Font& font) const;

    std::string             text_;
    int                     cursor_      = 0;
    int                     prevCursor_  = 0;
    bool                    active_      = false;
    bool                    allSelected_ = false;
    std::vector<VisualLine> lines_;       // rebuilt each render() call
    int                     scrollLine_  = 0;
    sf::FloatRect           rect_;
    sf::Color               textColor_   = sf::Color(220, 210, 185); // Col::Text default
    int                     charLimit_;
    int                     visibleLines_;
    sf::Font*               cachedFont_  = nullptr; // set each render(); used by handleClick
};
