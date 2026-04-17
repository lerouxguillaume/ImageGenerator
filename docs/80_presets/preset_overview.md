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
| `basePrompt` | `std::string` | Positive prompt |
| `negativePrompt` | `std::string` | Negative prompt |
| `steps` | `int` | Denoising steps |
| `cfg` | `float` | CFG / guidance scale |
| `modelId` | `std::string` | Model folder name (matches `availableModels` entries) |
| `width` | `int` | Output width in pixels; 0 = model default |
| `height` | `int` | Output height in pixels; 0 = model default |
| `createdAt` | `uint64_t` | Unix seconds at creation |

## `Generation` (`src/presets/Preset.hpp`)

Lightweight snapshot of current UI state; used as input to `PresetManager` methods.
Intentionally decoupled from `ImageGeneratorView` to avoid circular includes.

Fields mirror `Preset` plus `presetId` (empty string if no preset is active).

---

# PresetManager (`src/presets/PresetManager.hpp/cpp`)

Owned by `ImageGeneratorController` as a value member (`presetManager`).
Loads `presets.json` on construction; saves after every mutation.

## Core methods

```cpp
// Create a new preset from a Generation snapshot.
Preset createFromGeneration(const Generation& gen, const std::string& name);

// Overwrite an existing preset's data fields (preserves id, name, createdAt).
void updateFromGeneration(const std::string& presetId, const Generation& gen);

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

---

# Integration

## Applying a preset to the view

```cpp
void applyPresetToSettings(const Preset& preset, ImageGeneratorView& view);
```

Free function declared in `PresetManager.hpp`, defined in `PresetManager.cpp`.

Sets:
- `view.positiveArea` / `view.negativeArea` via `setText()`
- `view.generationParams.numSteps`, `guidanceScale`, `width`, `height`
- `view.selectedModelIdx` (linear scan of `view.availableModels` by folder name)
- `view.activePresetId`

If `modelId` is not found in `availableModels`, `selectedModelIdx` is left unchanged and a warning is logged.

## Building a Generation snapshot from the view

```cpp
Generation gen;
gen.prompt         = view.positiveArea.getText();
gen.negativePrompt = view.negativeArea.getText();
gen.modelId        = view.availableModels[view.selectedModelIdx];
gen.steps          = view.generationParams.numSteps;
gen.cfg            = view.generationParams.guidanceScale;
gen.width          = view.generationParams.width;
gen.height         = view.generationParams.height;
gen.presetId       = view.activePresetId; // empty if none
```

## Tracking active preset

`ImageGeneratorView::activePresetId` (empty string) indicates which preset is currently applied.
Set by `applyPresetToSettings`; clear it when the user manually edits a field if the UI should indicate "unsaved changes".

---

# ID generation

IDs are millisecond-precision Unix timestamps prefixed with `"preset_"`.
Collision is not possible in normal interactive use.
