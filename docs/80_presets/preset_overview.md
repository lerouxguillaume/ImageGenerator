# Preset System

Allows users to save, apply, update, and duplicate named generation configurations.

---

# Data model

## `Preset` (`src/presets/Preset.hpp`)

Plain data struct persisted to `presets.json`:

| Field | Type | Notes |
|---|---|---|
| `id` | `std::string` | Unique identifier (`"preset_<ms>"`) |
| `name` | `std::string` | User-facing label |
| `dsl` | `Prompt` | Full prompt DSL — source of truth for prompts |
| `steps` | `int` | Denoising steps |
| `cfg` | `float` | CFG / guidance scale |
| `modelId` | `std::string` | Model folder name (matches `availableModels` entries) |
| `width` | `int` | Output width in pixels; 0 = model default |
| `height` | `int` | Output height in pixels; 0 = model default |
| `createdAt` | `uint64_t` | Unix seconds at creation |

Raw `basePrompt`/`negativePrompt` strings are gone. The `Prompt` DSL (`src/prompt/Prompt.hpp`)
is the single source of truth. See `docs/85_prompt/prompt_dsl.md`.

## `GenerationSettings` (`src/presets/Preset.hpp`)

Lightweight snapshot of current UI state; used as input to `PresetManager` methods.
Intentionally decoupled from `SettingsPanel` to avoid circular includes.

| Field | Type | Notes |
|---|---|---|
| `dsl` | `Prompt` | Parsed DSL from current text areas |
| `modelId` | `std::string` | Model folder basename |
| `steps` | `int` | |
| `cfg` | `float` | |
| `width` | `int` | |
| `height` | `int` | |
| `presetId` | `std::string` | Empty if no preset is active |

---

# PresetManager (`src/presets/PresetManager.hpp/cpp`)

Owned by `ImageGeneratorController` as a value member (`presetManager`).
Loads `presets.json` on construction; saves after every mutation.

## Core methods

```cpp
// Create a new preset from a GenerationSettings snapshot.
Preset createFromGeneration(const GenerationSettings& gen, const std::string& name);

// Overwrite an existing preset's data fields (preserves id, name, createdAt).
void updateFromGeneration(const std::string& presetId, const GenerationSettings& gen);

// Copy a preset under a new name with a fresh id and createdAt.
Preset duplicatePreset(const std::string& presetId, const std::string& newName);

std::optional<Preset>      getPreset(const std::string& id) const;
const std::vector<Preset>& getAllPresets() const;
```

## Behaviour rules

- Applying a preset **replaces** all settings (no merge)
- Editing settings after applying a preset does **not** modify the preset
- Updating a preset is always **explicit** — no implicit mutation

---

# Storage

File: `presets.json` (working directory)  
Format: JSON array of preset objects, pretty-printed with 4-space indent.

Missing file → start empty (logged at INFO).  
Corrupt/invalid JSON → start empty (logged at ERROR).

## JSON format

```json
[
  {
    "id": "preset_1713432000000",
    "name": "Cinematic portrait",
    "dsl": {
      "subject": "girl",
      "styles": [],
      "positive": [
        {"value": "soft lighting", "weight": 1.0},
        {"value": "85mm",          "weight": 1.3}
      ],
      "negative": [
        {"value": "blurry", "weight": 1.0}
      ]
    },
    "steps": 30,
    "cfg": 7.0,
    "modelId": "sdxl-base",
    "width": 1024,
    "height": 1024,
    "createdAt": 1713432000
  }
]
```

**Note:** old `presets.json` files with `basePrompt`/`negativePrompt` string fields
will load with an empty DSL (those fields are silently ignored). Resave to migrate.

---

# Integration

## Applying a preset to the settings panel

```cpp
void applyPresetToSettings(const Preset& preset, SettingsPanel& panel);
```

Free function declared in `PresetManager.hpp`, defined in `PresetManager.cpp`.

Sets:
- `panel.positiveArea` via `PromptCompiler::compile(preset.dsl, ModelType::SDXL)`
- `panel.negativeArea` via `PromptCompiler::compileNegative(preset.dsl, ModelType::SDXL)`
- `panel.generationParams.numSteps`, `guidanceScale`, `width`, `height`
- `panel.selectedModelIdx` (linear scan of `panel.availableModels` by folder name)
- `panel.activePresetId`

SDXL (neutral) form is used for display. The correct model-specific compilation is
applied fresh at generation time via `launchGeneration`.

If `modelId` is not found in `availableModels`, `selectedModelIdx` is left unchanged and
a warning is logged.

## Building a GenerationSettings snapshot from the view

The controller's `buildGenerationSettings(view)` helper captures current state:

```cpp
GenerationSettings gs;
gs.dsl     = PromptParser::parse(sp.positiveArea.getText(), sp.negativeArea.getText());
gs.modelId = filesystem::path(sp.getSelectedModelDir()).filename().string();
gs.steps   = sp.generationParams.numSteps;
gs.cfg     = sp.generationParams.guidanceScale;
gs.width   = sp.generationParams.width;
gs.height  = sp.generationParams.height;
gs.presetId = sp.activePresetId;
```

## Tracking active preset

`SettingsPanel::activePresetId` (empty string) indicates which preset is currently applied.
Set by `applyPresetToSettings`; the controller also sets it after `createFromGeneration`.
`MenuBar` displays a ✓ next to the active preset in the dropdown.

---

# ID generation

IDs are millisecond-precision Unix timestamps prefixed with `"preset_"`.
Collision is not possible in normal interactive use.
