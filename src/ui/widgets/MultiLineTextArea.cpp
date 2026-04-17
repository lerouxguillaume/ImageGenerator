#include "MultiLineTextArea.hpp"
#include "../../enum/constants.hpp"
#include "../Helpers.hpp"
#include <SFML/Window/Clipboard.hpp>
#include <algorithm>

using namespace Helpers;

// ── Layout constants ──────────────────────────────────────────────────────────
static constexpr float    FIELD_PAD_X  = 8.f;
static constexpr float    FIELD_PAD_Y  = 6.f;
static constexpr float    FIELD_LINE_H = 17.f;  // font size 13 + 4px leading
static constexpr unsigned FIELD_FONT   = 13;

// ── Constructor ───────────────────────────────────────────────────────────────

MultiLineTextArea::MultiLineTextArea(int charLimit, int visibleLines)
    : charLimit_(charLimit), visibleLines_(visibleLines)
{}

// ── Widget interface ──────────────────────────────────────────────────────────

void MultiLineTextArea::setRect(const sf::FloatRect& rect) {
    rect_ = rect;
}

sf::FloatRect MultiLineTextArea::getRect() const {
    return rect_;
}

// ── Word-wrap layout ──────────────────────────────────────────────────────────

std::vector<MultiLineTextArea::VisualLine>
MultiLineTextArea::computeLines(sf::Font& font) const {
    std::vector<VisualLine> lines;
    if (text_.empty()) { lines.push_back({0, 0}); return lines; }

    const float maxW = rect_.width - FIELD_PAD_X * 2.f - 6.f; // 6px for scrollbar

    sf::Text t;
    t.setFont(font);
    t.setCharacterSize(FIELD_FONT);

    struct Word { int start, end; std::string text; };
    std::vector<Word> words;
    int i = 0;
    const int n = static_cast<int>(text_.size());
    while (i < n) {
        while (i < n && text_[i] == ' ') ++i;
        if (i >= n) break;
        const int ws = i;
        while (i < n && text_[i] != ' ') ++i;
        words.push_back({ws, i, text_.substr(static_cast<size_t>(ws),
                                              static_cast<size_t>(i - ws))});
    }

    if (words.empty()) { lines.push_back({0, 0}); return lines; }

    int lineStart      = words[0].start;
    int lineEnd        = words[0].end;
    std::string lineText = words[0].text;

    for (int wi = 1; wi < static_cast<int>(words.size()); ++wi) {
        const std::string test = lineText + " " + words[wi].text;
        t.setString(test);
        if (t.getLocalBounds().width > maxW) {
            lines.push_back({lineStart, lineEnd});
            lineStart = words[wi].start;
            lineEnd   = words[wi].end;
            lineText  = words[wi].text;
        } else {
            lineEnd  = words[wi].end;
            lineText = test;
        }
    }
    lines.push_back({lineStart, lineEnd});
    return lines;
}

// ── Rendering ─────────────────────────────────────────────────────────────────

void MultiLineTextArea::render(sf::RenderWindow& win, sf::Font& font) {
    cachedFont_ = &font;
    drawRect(win, rect_, Col::Panel2, active_ ? Col::BorderHi : Col::Border, 1.f);

    // Selection highlight (drawn before text so text renders on top)
    if (active_ && allSelected_) {
        sf::RectangleShape sel({rect_.width - 2.f, rect_.height - 2.f});
        sel.setPosition(rect_.left + 1.f, rect_.top + 1.f);
        sel.setFillColor(sf::Color(42, 85, 138, 110));
        win.draw(sel);
    }

    lines_ = computeLines(font);

    // Find cursor's visual line
    int cursorLine = static_cast<int>(lines_.size()) - 1;
    for (int l = 0; l + 1 < static_cast<int>(lines_.size()); ++l) {
        if (cursor_ < lines_[l + 1].start) { cursorLine = l; break; }
    }

    // Auto-scroll so cursor stays visible — only when cursor actually moved,
    // so manual scroll isn't overridden every frame.
    if (active_ && cursor_ != prevCursor_) {
        if (cursorLine < scrollLine_) scrollLine_ = cursorLine;
        if (cursorLine >= scrollLine_ + visibleLines_)
            scrollLine_ = cursorLine - visibleLines_ + 1;
    }
    prevCursor_ = cursor_;
    scrollLine_ = std::clamp(scrollLine_, 0,
                             std::max(0, static_cast<int>(lines_.size()) - visibleLines_));

    // Draw visible lines
    const int last = std::min(static_cast<int>(lines_.size()), scrollLine_ + visibleLines_);
    for (int l = scrollLine_; l < last; ++l) {
        const auto [lStart, lEnd] = lines_[static_cast<size_t>(l)];
        std::string lineText = text_.substr(static_cast<size_t>(lStart),
                                            static_cast<size_t>(lEnd - lStart));
        if (active_ && !allSelected_ && l == cursorLine) {
            const int col = std::clamp(cursor_ - lStart, 0, static_cast<int>(lineText.size()));
            lineText.insert(static_cast<size_t>(col), "|");
        }
        const float ly = rect_.top + FIELD_PAD_Y
                        + static_cast<float>(l - scrollLine_) * FIELD_LINE_H;
        drawText(win, font, lineText, textColor_, rect_.left + FIELD_PAD_X, ly, FIELD_FONT);
    }

    // Scrollbar (only when content overflows)
    if (static_cast<int>(lines_.size()) > visibleLines_) {
        const float trackH = rect_.height - FIELD_PAD_Y * 2.f;
        const float ratio  = static_cast<float>(visibleLines_)
                           / static_cast<float>(lines_.size());
        const float thumbH = std::max(8.f, trackH * ratio);
        const float thumbY = rect_.top + FIELD_PAD_Y
                           + trackH * static_cast<float>(scrollLine_)
                                     / static_cast<float>(lines_.size());
        const float trackX = rect_.left + rect_.width - FIELD_PAD_X + 1.f;
        drawRect(win, {trackX, rect_.top + FIELD_PAD_Y, 4.f, trackH}, Col::Panel);
        drawRect(win, {trackX, thumbY, 4.f, thumbH}, Col::Border);
    }
}

// ── Input ─────────────────────────────────────────────────────────────────────

bool MultiLineTextArea::handleEvent(const sf::Event& e) {
    if (!active_) return false;

    // Helper: find which visual line the cursor is on
    auto findLine = [this]() {
        int l = static_cast<int>(lines_.size()) - 1;
        for (int i = 0; i + 1 < static_cast<int>(lines_.size()); ++i) {
            if (cursor_ < lines_[i + 1].start) { l = i; break; }
        }
        return l;
    };

    if (e.type == sf::Event::KeyPressed) {
        switch (e.key.code) {
        case sf::Keyboard::Left:
            allSelected_ = false;
            if (cursor_ > 0) --cursor_;
            return true;

        case sf::Keyboard::Right:
            allSelected_ = false;
            if (cursor_ < static_cast<int>(text_.size())) ++cursor_;
            return true;

        case sf::Keyboard::Up:
            allSelected_ = false;
            if (!lines_.empty()) {
                const int l = findLine();
                if (l > 0) {
                    const int col     = cursor_ - lines_[static_cast<size_t>(l)].start;
                    const int prevLen = lines_[static_cast<size_t>(l - 1)].end
                                      - lines_[static_cast<size_t>(l - 1)].start;
                    cursor_ = lines_[static_cast<size_t>(l - 1)].start
                             + std::min(col, prevLen);
                }
            }
            return true;

        case sf::Keyboard::Down:
            allSelected_ = false;
            if (!lines_.empty()) {
                const int l = findLine();
                if (l + 1 < static_cast<int>(lines_.size())) {
                    const int col     = cursor_ - lines_[static_cast<size_t>(l)].start;
                    const int nextLen = lines_[static_cast<size_t>(l + 1)].end
                                      - lines_[static_cast<size_t>(l + 1)].start;
                    cursor_ = lines_[static_cast<size_t>(l + 1)].start
                             + std::min(col, nextLen);
                }
            }
            return true;

        case sf::Keyboard::Home:
            allSelected_ = false;
            cursor_ = 0;
            return true;

        case sf::Keyboard::End:
            allSelected_ = false;
            cursor_ = static_cast<int>(text_.size());
            return true;

        case sf::Keyboard::BackSpace:
            if (allSelected_) {
                text_.clear(); cursor_ = 0; allSelected_ = false;
            } else if (!text_.empty() && cursor_ > 0) {
                --cursor_;
                text_.erase(static_cast<size_t>(cursor_), 1);
            }
            return true;

        case sf::Keyboard::Delete:
            if (allSelected_) {
                text_.clear(); cursor_ = 0; allSelected_ = false;
            } else if (cursor_ < static_cast<int>(text_.size())) {
                text_.erase(static_cast<size_t>(cursor_), 1);
            }
            return true;

        default:
            if (e.key.control) {
                if (e.key.code == sf::Keyboard::A) {
                    allSelected_ = true;
                    return true;
                }
                if (e.key.code == sf::Keyboard::C) {
                    sf::Clipboard::setString(text_);
                    return true;
                }
                if (e.key.code == sf::Keyboard::V) {
                    if (allSelected_) { text_.clear(); cursor_ = 0; allSelected_ = false; }
                    const std::string clip = sf::Clipboard::getString().toAnsiString();
                    for (char c : clip) {
                        if (static_cast<unsigned char>(c) >= 32
                            && static_cast<int>(text_.size()) < charLimit_) {
                            text_.insert(static_cast<size_t>(cursor_), 1, c);
                            ++cursor_;
                        }
                    }
                    return true;
                }
            }
            break;
        }
    }

    if (e.type == sf::Event::TextEntered) {
        const auto ch = e.text.unicode;
        if (ch >= 32 && ch < 127 && static_cast<int>(text_.size()) < charLimit_) {
            if (allSelected_) { text_.clear(); cursor_ = 0; allSelected_ = false; }
            text_.insert(static_cast<size_t>(cursor_), 1, static_cast<char>(ch));
            ++cursor_;
            return true;
        }
    }

    return false;
}

void MultiLineTextArea::handleScroll(float delta) {
    const int lines = static_cast<int>(delta > 0 ? 1 : -1);
    scrollLine_ = std::max(0, scrollLine_ + lines);
}

void MultiLineTextArea::handleClick(sf::Vector2f pos) {
    if (!rect_.contains(pos)) return;
    active_      = true;
    allSelected_ = false;

    if (!cachedFont_ || lines_.empty()) return;

    // Which visual (scrolled) line was clicked?
    const float relY    = pos.y - rect_.top - FIELD_PAD_Y;
    const int   visLine = std::clamp(static_cast<int>(relY / FIELD_LINE_H), 0, visibleLines_ - 1);
    const int   lineIdx = std::clamp(scrollLine_ + visLine, 0, static_cast<int>(lines_.size()) - 1);

    const auto& vl       = lines_[static_cast<size_t>(lineIdx)];
    const std::string lineText = text_.substr(static_cast<size_t>(vl.start),
                                              static_cast<size_t>(vl.end - vl.start));
    const float relX = pos.x - rect_.left - FIELD_PAD_X;

    // Find the character column closest to relX
    sf::Text t;
    t.setFont(*cachedFont_);
    t.setCharacterSize(FIELD_FONT);

    int bestCol = 0;
    for (int c = 1; c <= static_cast<int>(lineText.size()); ++c) {
        t.setString(lineText.substr(0, static_cast<size_t>(c)));
        const float wAt = t.getLocalBounds().width;
        if (wAt > relX) {
            t.setString(lineText.substr(0, static_cast<size_t>(c - 1)));
            bestCol = (relX - t.getLocalBounds().width < wAt - relX) ? c - 1 : c;
            break;
        }
        bestCol = c;
    }

    cursor_ = vl.start + bestCol;
}

// ── State accessors ───────────────────────────────────────────────────────────

void MultiLineTextArea::setText(const std::string& t) {
    text_        = t;
    cursor_      = static_cast<int>(t.size());
    allSelected_ = false;
    scrollLine_  = 0;
}

const std::string& MultiLineTextArea::getText() const {
    return text_;
}

bool MultiLineTextArea::isActive() const {
    return active_;
}

void MultiLineTextArea::setActive(bool active) {
    active_ = active;
    if (active_) {
        // Move cursor to end when activating programmatically
        cursor_ = static_cast<int>(text_.size());
        allSelected_ = false;
    }
}

void MultiLineTextArea::setTextColor(sf::Color c) {
    textColor_ = c;
}
