# UI Architecture

SFML-based retained UI system with a component panel architecture.

The UI styling lives entirely in the centralized theme layer `src/ui/Theme.h`,
accessed via `Theme::instance().colors()` / `.metrics()` / `.typography()`:
- `UiColors` — semantic colors for backgrounds, panels, accents, status states, and overlays
- `UiMetrics` — spacing tokens, control heights, layout widths, and shared sizing
- `UiTypography` — shared text scale for titles, section labels, body copy, and helper text

There is no longer a `constants.hpp` compatibility layer — it has been removed and all code reads Theme directly.

---

# Screen model

The application has two top-level screens:

- `MenuView` — launcher with `Generate Images` and `Import Model`
- `ImageGeneratorView` — the single generate/edit workspace

There is one `ImageGeneratorView` + `ImageGeneratorController`. It is **txt2img by default**; attaching an optional input image (via the **Edit** button on a result) turns the same screen into **img2img** — a strength slider and an "Editing: `<file>`" banner appear, and the primary button relabels to **Edit Image**. Clearing the input image (the `×` on the banner) returns to plain generation.

The current visual language is a dark desktop-tool style with framed surfaces, centralized spacing/typography tokens, and cooler accent colors for active and primary actions.

---

# Window

The window is **resizable** (`sf::Style::Close | sf::Style::Resize`).

- **Opens maximized** — `App` constructor calls `maximizeWindow(win)` (native `ShowWindow(SW_MAXIMIZE)` on Windows; sizes to `getDesktopMode()` elsewhere). The title bar and window controls are kept (this is *not* `sf::Style::Fullscreen`).
- Default size: 1280 × 800 — this is the **restore** size when the user un-maximizes
- Minimum enforced size: **700 × 550** (smaller resize attempts are snapped back)
- On `sf::Event::Resized`: the SFML view is updated so coordinates map directly to pixels
- `SettingsPanel` stays at fixed width (`metrics.generatorLeftPanelWidth = 460`); `ResultPanel` takes remaining width
- All layout values read from `win.getSize()` dynamically every frame — no hard-coded window dimensions in panel code

---

# Layout (default 1280 × 800, scales with window)

## Generate / edit screen

```
┌─────────────────────────────────────────────────────────────┐  y=0
│  MenuBar                                                    │  h=40
├──────────────────────────┬──────────────────────────────────┤  y=40
│  SettingsPanel (cards)   │  ResultPanel                     │
│  (460px wide, fixed)     │  (win.width − 460, grows)        │
│                          │                                  │
│  MODEL & LORA            │  preview image [×deselect]       │
│  PROMPT (pos/chips/neg)  │   (native size, downscaled)      │
│  (EDITING SOURCE)**      │  caption: file · WxH · N of M    │
│  SAMPLING (steps/cfg/im) │  progress bar / cancel           │
│  HIRES FIX               │  error banner                    │
│  SEED                    │  gallery strip / grid            │
│  [ Generate / Edit ]     │  action bar [Edit] [Delete]      │
├──────────────────────────┴──────────────────────────────────┤  y varies
│  LlmBar collapsed (visible when LLM available or loading)   │  h=44
│  LlmBar expanded (instruction area below toggle row)        │  h=124
└─────────────────────────────────────────────────────────────┘
```

*compiled preview visible only when SD1.5 model is selected
**the `EDITING SOURCE` card (strength presets/slider) appears only when an input image is attached; the rail's primary button then reads **Edit Image**

Layout metrics (`Theme::metrics()` / `UiMetrics`):
- `menuBarHeight = 40` — top bar height
- `generatorLeftPanelWidth = 460` — SettingsPanel width (fixed)
- `llmBarHeight = 44` — bottom bar height (collapsed)
- `llmExpandedExtraHeight = 80` — extra height added when LLM bar is expanded

Body height is computed dynamically in `ImageGeneratorView::render` from `win.getSize()`:
- No LLM bar: `winH - menuBarHeight`
- LLM bar collapsed: `winH - menuBarHeight - llmBarHeight`
- LLM bar expanded: `winH - menuBarHeight - llmBarHeight - llmExpandedExtraHeight`

The LLM bar is shown whenever a prompt enhancer is available or loading, regardless of whether an input image is attached.

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
`showPresetControls` is always enabled on this screen.  
Save modal and overlay dim use `win.getSize()` for centering.
The menu bar now renders as a themed header surface rather than a flat strip.

## `SettingsPanel`

State: `positiveArea`, `negativeArea`, `generationParams`, model selection, LoRA list, seed, slider drag  
DSL display state (set by controller each frame): `currentDsl`, `compiledPreview`  
All params always visible (no Advanced toggle)  
Renders as a stack of **framed cards** inside the left rail, each with an uppercase micro-label header: `MODEL & LORA`, `PROMPT`, (`EDITING SOURCE`), `SAMPLING`, `HIRES FIX`, `SEED`. `render()` uses a local `drawCard(top,h,title)` helper (draws the framed box + header, returns the content-start y) and a `drawSwitch(box,on,enabled)` helper. Card heights are computed per-frame from their content (chip rows, hires on/off, img2img on/off) so a card's box exactly wraps its controls.
- Model card: **full-width** dropdown styled as an input with an inline `SDXL`/`SD1.5` badge; LoRA button below (or a "No LoRA for this model" note)
- Sampling card: `Steps` / `CFG scale` / `Images` on one unified slider style (uppercase label left, mono value right, teal track/thumb)
- Hires card: an on/off **toggle switch** (`drawSwitch`, reusing `btnHiresToggle_` as the hit rect) that reveals the scale/strength sliders + Pixel/Latent segmented control only when enabled
- The **Generate / Edit Image** primary button is pinned full-width to the **bottom of this rail** (`btnGenerate_` → `generateRequested`, launched by the controller). Its label is derived locally from `generationParams.initImagePath` (empty → `Generate`, else `Edit Image`). It is *not* in the ResultPanel.
- The rail is fixed-height with no scroll; the always-visible cards + the opt-in expansions (hires-on, img2img) are tuned to fit the maximized window. A vertical scroll is a possible future addition if content overflows at small sizes.
Tab cycles focus: positive → negative → positive.  
`positiveArea` shows 4 visible lines (fieldH=86); `negativeArea` shows 3 visible lines (fieldH=68).  
Focus is mutually exclusive with `LlmBar::instructionArea` — controller enforces this after each handleEvent.

**Always shown**:
- positive prompt field
- token chip visualisation
- negative prompt field
- compiled preview (SD1.5 only)

**Img2img controls** — an `EDITING SOURCE` card (shown only when `generationParams.initImagePath` is set):
- The card header shows the source `<file>` name + a `×` button (`btnClearInit_`) — clicking `×` sets `initImagePath = ""`, returning to txt2img
- Strength presets: `[Subtle]`, `[Medium]`, `[Strong]` map to coarse strength defaults before fine-tuning
- Strength slider: range 0.05–1.0 in 0.05 steps; stored in `generationParams.strength`
- The normal positive/negative prompt drives the edit — there is no separate edit-instruction field
- If the selected model has no VAE encoder, a "img2img not supported by this model" note replaces the strength controls

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

State: `resultTexture`, `generating`, progress atomics (`generationStep`, `generationStage`, `generationImageNum`, `generationTotalImages`, `cancelToken`, …)  
The generating overlay reads `generationStage` (a `atomic<GenerationStage>`) to show a typed label — e.g. "Loading model...", "Encoding prompt...", "Encoding image..." (img2img), "Step N / M" during denoising, "Decoding image...". The step counter drives the progress bar fill only during `Denoising`; all other stages leave the bar at its current fill.  
Action flags: `generateRequested`, `improveRequested`, `deleteRequested`, `cancelToken`  
Path fields: `lastImagePath` (base output path, set at generation start), `displayedImagePath` (path of the image currently shown).

**Gallery** (`gallery: vector<GalleryItem>`, `selectedIndex: int`):  
- Populated by `ImageGeneratorController::refreshGallery()` on first open, after each generation, and when the output dir changes in settings
- Each `GalleryItem` holds `path`, `filename`, `score`, and a `shared_ptr<sf::Texture>` thumbnail (null until async load completes). *There are no per-thumbnail rating badges* — the old `recommended`/`usable`/`near` fields and their `BEST`/`OK`/`NEAR` badges were removed.
- Thumbnails are loaded and resized to ≤92 px on background threads; `flushPendingThumbs()` promotes ready `sf::Image` results to `sf::Texture` each frame in `update()`
- Sorted newest-first by file modification time

**Single interaction model** (`renderGallery()`): the gallery has one selection gesture — **click a thumbnail** — plus **mouse-wheel scroll** over the gallery region. The old duplicate `< >` arrow pairs (one flanking the preview, one paging the strip) were removed. Two view modes share the same renderer, toggled by the header's **Grid view / Collapse** button (`btnGalleryExpand_`):
- **Strip** (default, `galleryExpanded_ = false`): a single horizontal row pinned above the action bar; wheel scrolls horizontally. The preview image + a `filename · WxH · N of M` caption sit above it.
- **Grid** (`galleryExpanded_ = true`): a multi-row wall that takes over the panel body (preview hidden); wheel scrolls vertically.
- Drawing is clipped to the scroll region via a temporary `sf::View` (viewport = the content rect) so scrolled thumbnails don't overdraw. `galleryScroll_`/`galleryScrollMax_` track the offset; the selection is scrolled into view once when it changes (`lastSelectedIndex_`). A thin scrollbar hint is drawn when content overflows.
- The selected thumbnail gets a full teal frame + a corner tick badge (not a subtle 1px/2px border difference).
- Wheel events reach the panel because `SettingsPanel::handleEvent` returns false for wheel not over its text areas, so the controller's routing chain falls through to `resultPanel.handleEvent`.

**Deselect** (`deselectRequested`): two gestures unselect the current image and drop out of img2img edit mode — the `×` button on the preview's top-right corner (`btnDeselect_`), or clicking the already-selected thumbnail again. The controller responds by calling `selectGalleryImage(view, -1)` and clearing `generationParams.initImagePath`.

**Button layout when `resultLoaded`** (bottom-left of panel):  
- `[Edit]` · `[Delete]` — both act on the selected result. The action bar is hidden when no image is selected. Generate lives at the bottom of the settings rail (see SettingsPanel).

- `[Edit]`: sets `improveRequested`; the controller attaches `displayedImagePath` as the input image **in place** (setting `generationParams.initImagePath`, defaulting strength to 0.5) — no screen switch. The screen is now in img2img mode.
- `[Delete]`: sets `deleteRequested`; controller canonicalizes the path, verifies it is inside `config.outputDir`, removes the file, then calls `refreshGallery()`
- The primary button label reads `Generate` in txt2img and `Edit Image` once an input image is attached

## `LlmBar`

State: `instructionArea`, `expanded` toggle  
Capture fields: `originalPositive`, `originalNegative` (snapshot before enhancement for merge base)  
Action flag: `enhanceRequested`  
Collapsed: 44px strip. Expanded: bar grows by `metrics.llmExpandedExtraHeight` (80px); instruction textarea
appears below the toggle row (not as a floating overlay).  
Only rendered when `promptEnhancerAvailable || llmLoading`

## `SettingsModal`

State: 4 directory strings + cursors + active flags  
Action flags: `saveRequested`, `cancelRequested`, `browseTarget`  
Rendered as a full-screen overlay when `view.showSettings == true`  
Overlay dim and box centering use `win.getSize()` dynamically.
The modal uses the shared theme palette and control styling so directory selection is visually consistent.

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
6. resultPanel.handleEvent()     → check generateRequested / improveRequested / deleteRequested
7. llmBar.handleEvent()          → check enhanceRequested
```

Each panel's `handleEvent` returns `true` if the event was consumed. The controller stops routing on the first consumer.

---

# Action flag pattern

Panels set flags; the controller acts on them after the delegation chain:

```cpp
if (view.resultPanel.handleEvent(e)) {
    // Gallery thumbnail click — load selected image
    const std::string selectedPath = view.resultPanel.getSelectedImagePath();
    if (!selectedPath.empty() && selectedPath != view.resultPanel.displayedImagePath)
        selectGalleryImage(view, view.resultPanel.selectedIndex);
    if (view.resultPanel.generateRequested) {
        view.resultPanel.generateRequested = false;
        launchGeneration(view);   // controller launches the jthread
    }
    if (view.resultPanel.improveRequested) {
        view.resultPanel.improveRequested   = false;
        // attach displayedImagePath as the img2img input, in place
        view.settingsPanel.generationParams.initImagePath = view.resultPanel.displayedImagePath;
    }
    if (view.resultPanel.deleteRequested) {
        view.resultPanel.deleteRequested = false;
        // canonicalize + verify path is inside outputDir before removing
        std::filesystem::remove(canonicalSelected, ec);
        refreshGallery(view);
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
`src/ui/Theme.h/.cpp` — font loading plus `UiColors`, `UiMetrics`, `UiTypography`

Panels and shared helpers draw from the centralized theme tokens (`Theme::instance().colors()` / `.metrics()`) rather than ad hoc per-screen styling.

---

# Font access

`ImageGeneratorView` inherits `sf::Font font` from `Screen`. The font is passed to each panel's `render(win, font)` call. Panels do not load fonts themselves.
