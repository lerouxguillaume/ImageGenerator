# UI Architecture

SFML-based retained UI system with a component panel architecture.

---

# Window

The window is **resizable** (`sf::Style::Close | sf::Style::Resize`).

- Default size: 1280 × 800
- Minimum enforced size: **700 × 550** (smaller resize attempts are snapped back)
- On `sf::Event::Resized`: the SFML view is updated so coordinates map directly to pixels
- `SettingsPanel` stays at fixed width (`LEFT_PANEL_W = 460`); `ResultPanel` takes remaining width
- All layout values read from `win.getSize()` dynamically every frame — no hard-coded WIN_W/WIN_H in panel code

---

# Layout (default 1280 × 800, scales with window)

```
┌─────────────────────────────────────────────────────────────┐  y=0
│  MenuBar                                                    │  h=40
├──────────────────────────┬──────────────────────────────────┤  y=40
│  SettingsPanel           │  ResultPanel                     │
│  (460px wide, fixed)     │  (win.width − 460, grows)        │
│                          │                                  │
│  model / lora            │  generated image (native size,   │
│  positive prompt         │   downscaled only if too large)  │
│  token chips             │                                  │
│  negative prompt         │  [Generate]                      │
│  compiled preview*       │  progress bar / cancel           │
│  steps / cfg / images    │  error banner                    │
│  seed                    │                                  │
├──────────────────────────┴──────────────────────────────────┤  y varies
│  LlmBar collapsed (visible when LLM available or loading)   │  h=44
│  LlmBar expanded (instruction area below toggle row)        │  h=124
└─────────────────────────────────────────────────────────────┘
```

*compiled preview visible only when SD1.5 model is selected

Layout constants (`src/enum/constants.hpp`):
- `MENU_BAR_H = 40` — top bar height
- `LEFT_PANEL_W = 460` — SettingsPanel width (fixed)
- `LLM_BAR_H = 44` — bottom bar height (collapsed)
- `LLM_EXPANDED_H = 80` — extra height added when LLM bar is expanded

Body height is computed dynamically in `ImageGeneratorView::render` from `win.getSize()`:
- No LLM bar: `winH - MENU_BAR_H`
- LLM bar collapsed: `winH - MENU_BAR_H - LLM_BAR_H`
- LLM bar expanded: `winH - MENU_BAR_H - LLM_BAR_H - LLM_EXPANDED_H`

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
Save modal and overlay dim use `win.getSize()` for centering.

## `SettingsPanel`

State: `positiveArea`, `negativeArea`, `generationParams`, model selection, LoRA list, seed, slider drag  
DSL display state (set by controller each frame): `currentDsl`, `compiledPreview`  
All params always visible (no Advanced toggle)  
Tab cycles focus: positive → negative → positive  
`positiveArea` shows 4 visible lines (fieldH=86); `negativeArea` shows 3 visible lines (fieldH=68).  
Focus is mutually exclusive with `LlmBar::instructionArea` — controller enforces this after each handleEvent.

**Img2img section** (conditional — only shown when `generationParams.initImagePath` is non-empty):
- Info row: truncated filename + `[Clear]` button — clicking Clear sets `initImagePath = ""`
- Strength slider: range 0.05–1.0 in 0.05 steps; stored in `generationParams.strength`

### Token chip row (Phase 8)

Rendered between positive area and negative label. Read-only — visualises the parsed DSL:
- Subject chip: gold border + gold text; weight suffix shown when non-default
- Positive token chips: neutral border; blue if `weight > 1`, muted if `weight < 1`
- Weight shown as `label 2.0×` when non-default
- Wraps to a second row (max 2 rows)
- Height is dynamic (20px per row + gaps)

### Compiled preview strip (Phase 6)

Single muted line `→ <compiled string>` rendered below the negative area.
Visible **only when SD1.5 is selected** — hidden for SDXL since output matches input.
Shows the full compiled positive including any quality boosters injected from `ModelDefaults.qualityBoosters`.

## `ResultPanel`

State: `resultTexture`, `generating`, progress atomics (`generationStep`, `cancelToken`, …)  
Action flags: `generateRequested`, `useAsInitRequested`, `cancelToken`  
Generate button at bottom of panel; progress overlay covers the image during generation.  
Generated image displayed at native pixel size; downscaled only when larger than the panel.

When `resultLoaded`:
- `[Use as init]` button appears to the left of `[Generate]`
- Clicking it sets `useAsInitRequested = true`
- Controller responds by copying `lastImagePath` → `settingsPanel.generationParams.initImagePath`

## `LlmBar`

State: `instructionArea`, `expanded` toggle  
Capture fields: `originalPositive`, `originalNegative` (snapshot before enhancement for merge base)  
Action flag: `enhanceRequested`  
Collapsed: 44px strip. Expanded: bar grows by `LLM_EXPANDED_H` (80px); instruction textarea
appears below the toggle row (not as a floating overlay).  
Only rendered when `promptEnhancerAvailable || llmLoading`

## `SettingsModal`

State: 4 directory strings + cursors + active flags  
Action flags: `saveRequested`, `cancelRequested`, `browseTarget`  
Rendered as a full-screen overlay when `view.showSettings == true`  
Overlay dim and box centering use `win.getSize()` dynamically.

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
1. sf::Event::Resized — update view + enforce min size (handled in App::run before controllers)
2. Window close / Escape chain (save modal > settings modal > preset dropdown > navigate)
3. settingsModal.handleEvent()   — when showSettings == true (blocks all else)
4. menuBar.handleEvent()         → check action flags (back, settings, preset select, save)
5. settingsPanel.handleEvent()
6. resultPanel.handleEvent()     → check generateRequested
7. llmBar.handleEvent()          → check enhanceRequested
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
    if (view.resultPanel.useAsInitRequested) {
        view.resultPanel.useAsInitRequested = false;
        view.settingsPanel.generationParams.initImagePath = view.resultPanel.lastImagePath;
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
