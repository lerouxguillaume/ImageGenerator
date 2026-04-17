# UI Architecture

SFML-based retained UI system with a component panel architecture.

---

# Layout (1280 × 800)

```
┌─────────────────────────────────────────────────────────────┐  y=0
│  MenuBar                                                    │  h=40
├──────────────────────────┬──────────────────────────────────┤  y=40
│  SettingsPanel           │  ResultPanel                     │
│  (460px wide)            │  (820px wide)                    │
│                          │                                  │
│  model / lora            │  generated image (native size,   │
│  positive prompt         │   downscaled only if too large)  │
│  negative prompt         │                                  │
│  steps / cfg / images    │  [Generate]                      │
│  seed                    │  progress bar / cancel           │
│                          │  error banner                    │
├──────────────────────────┴──────────────────────────────────┤  y varies
│  LlmBar collapsed (visible when LLM available or loading)   │  h=44
│  LlmBar expanded (instruction area below toggle row)        │  h=124
└─────────────────────────────────────────────────────────────┘
```

Layout constants (`src/enum/constants.hpp`):
- `MENU_BAR_H = 40` — top bar height
- `LEFT_PANEL_W = 460` — SettingsPanel width
- `LLM_BAR_H = 44` — bottom bar height (collapsed)
- `LLM_EXPANDED_H = 80` — extra height added when LLM bar is expanded
- `BODY_H_FULL = 760` — body height without LLM bar (compile-time constant)

Body height is computed dynamically in `ImageGeneratorView::render`:
- No LLM bar: `WIN_H - MENU_BAR_H`
- LLM bar collapsed: `WIN_H - MENU_BAR_H - LLM_BAR_H`
- LLM bar expanded: `WIN_H - MENU_BAR_H - LLM_BAR_H - LLM_EXPANDED_H`

---

# Panel classes

Each panel (`src/ui/widgets/`):
- Owns its visual state (hit rects, show/hide flags, text fields, atomics)
- Exposes `setRect(FloatRect)`, `render(win, font)`, `handleEvent(event) → bool`
- Exposes **action flags** (`bool`/`std::string`) the controller checks after `handleEvent`
- Does **not** call into the controller; all cross-panel coordination is in the controller

## `MenuBar`

State: preset dropdown, save-as modal  
Action flags: `backRequested`, `settingsRequested`, `saveConfirmed`, `selectedPresetId`  
`setPresets(presets, activeId)` — called by controller after preset list changes

## `SettingsPanel`

State: `positiveArea`, `negativeArea`, `generationParams`, model selection, LoRA list, seed, slider drag  
All params always visible (no Advanced toggle)  
Tab cycles focus: positive → negative → positive  
`positiveArea` shows 4 visible lines (fieldH=86); `negativeArea` shows 3 visible lines (fieldH=68).  
Focus is mutually exclusive with `LlmBar::instructionArea` — controller enforces this after each handleEvent.

## `ResultPanel`

State: `resultTexture`, `generating`, progress atomics (`generationStep`, `cancelToken`, …)  
Action flags: `generateRequested`, `cancelToken` (set by Cancel button, read+cleared by controller)  
Generate button at bottom of panel; progress overlay covers the image during generation.  
Generated image displayed at native pixel size; downscaled only when larger than the panel.

## `LlmBar`

State: `instructionArea`, `expanded` toggle, `enhanceDone` atomic  
Action flag: `enhanceRequested`  
Collapsed: 44px strip. Expanded: bar grows by `LLM_EXPANDED_H` (80px); instruction textarea
appears below the toggle row (not as a floating overlay).  
Only rendered when `promptEnhancerAvailable || llmLoading`

## `SettingsModal`

State: 4 directory strings + cursors + active flags  
Action flags: `saveRequested`, `cancelRequested`, `browseTarget`  
Rendered as a full-screen overlay when `view.showSettings == true`

---

# Widget base class

`src/ui/widgets/base/Widget.h`:
```cpp
class Widget {
public:
    virtual void render(sf::RenderWindow& win, sf::Font& font) = 0;
    virtual void setRect(const sf::FloatRect& rect) = 0;
};
```

Panels are not required to inherit Widget — they follow the same interface pattern but are concrete classes.

`MultiLineTextArea` (inherits Widget) is the only shared input widget, used by `SettingsPanel` and `LlmBar`.

---

# Event routing

```cpp
// Controller::handleEvent() priority order:
1. Window close / Escape chain (save modal > settings modal > preset dropdown > navigate)
2. settingsModal.handleEvent()   — when showSettings == true (blocks all else)
3. menuBar.handleEvent()         → check action flags (back, settings, preset select, save)
4. settingsPanel.handleEvent()
5. resultPanel.handleEvent()     → check generateRequested
6. llmBar.handleEvent()          → check enhanceRequested
```

Each panel's `handleEvent` returns `true` if the event was consumed. The controller stops routing on the first consumer.

---

# Action flag pattern

Panels set flags; the controller acts on them after the delegation chain:

```cpp
if (view.resultPanel.handleEvent(e)) {
    if (view.resultPanel.generateRequested) {
        view.resultPanel.generateRequested = false;
        launchGeneration(view);   // controller launches the jthread
    }
    if (view.resultPanel.cancelToken.exchange(false))
        generationThread_.request_stop();  // bridges Cancel button → jthread stop_token
    return;
}
```

No callbacks or virtual dispatch — plain public bools and strings.

---

# Drawing helpers

`src/ui/Helpers.hpp` — `drawRect`, `drawText`, `drawTextC`, `drawTextR`, `drawWrapped`, `drawBar`  
`src/ui/Buttons.hpp` — `drawButton`  
`src/ui/Theme.cpp` — font loading singleton  
`src/enum/constants.hpp` — `Col::*` colour palette, layout constants  

All panels use these helpers directly; no per-panel theming.

---

# Font access

`ImageGeneratorView` inherits `sf::Font font` from `Screen`. The font is passed to each panel's `render(win, font)` call. Panels do not load fonts themselves.
