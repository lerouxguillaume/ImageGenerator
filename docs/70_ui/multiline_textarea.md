# MultiLineTextArea Widget

`src/ui/widgets/MultiLineTextArea.hpp/.cpp` — self-contained text area widget. The view and controller do not manipulate cursor or scroll state directly.

## Constructor

```cpp
explicit MultiLineTextArea(int charLimit = 2000, int visibleLines = 4);
```

`visibleLines` drives rendering height and scroll bounds. Positive/negative prompt areas use 4 lines; the instruction area uses 2.

## View instances

```cpp
MultiLineTextArea positiveArea{2000};       // 4 visible lines, 2000 char limit
MultiLineTextArea negativeArea{2000};       // 4 visible lines, 2000 char limit
MultiLineTextArea instructionArea{500, 2};  // 2 visible lines, 500 char limit
```

## Ownership

The widget owns all text state internally:
- Text content and cursor byte offset
- All-selected flag (Ctrl+A)
- Word-wrap layout (`VisualLine = {start, end}` byte ranges) — rebuilt each `render()` call
- Vertical scroll position
- Active/focus state

## Public API

```cpp
void render(sf::RenderWindow& win, sf::Font& font);  // call setRect() first
void setRect(const sf::FloatRect& rect);
bool handleEvent(const sf::Event& e);   // returns true if consumed
void handleScroll(float delta);         // +1 = scroll down, -1 = up
void handleClick(sf::Vector2f pos);     // activates self if pos is inside rect
void setText(const std::string& t);
const std::string& getText() const;
bool isActive() const;
void setActive(bool active);            // true → cursor moves to end of text
void setTextColor(sf::Color c);
```

`setRect()` is called immediately before each `render()` call. Layout constants: `FIELD_W=700`, `FIELD_H=86` (4-line fields), `FIELD_H_SM=46` (2-line instruction field).

## Keyboard handling

| Key | Action |
|---|---|
| Left/Right | Move cursor by character |
| Up/Down | Move cursor using word-wrap line layout |
| Home/End | Jump to start/end of current visual line |
| Backspace/Delete | Remove character; clears all-selected text first |
| Ctrl+A | Select all; next keystroke replaces entire content |
| Ctrl+C | Copy full field text to clipboard |
| Ctrl+V | Paste from clipboard, filtered to ASCII ≥ 32, capped at `charLimit_` |
| TextEntered | Filtered to ASCII ≥ 32, enforces `charLimit_` |

## Focus / Tab cycle

Tab cycles: positive → negative → instruction (when LLM available) → positive. Mutual exclusion is enforced by calling `setActive(false)` on all three before activating the next.

The instruction area is only rendered and only receives clicks when `view.promptEnhancerAvailable` (or `llmLoading`) is true.

## What NOT to do

- Do not call `computeLines()` or `drawPromptField()` — those functions have been removed.
- Do not manipulate cursor/scroll fields directly from the view or controller — use the public API.
- Do not render or route clicks to `instructionArea` when `promptEnhancerAvailable` is false.
