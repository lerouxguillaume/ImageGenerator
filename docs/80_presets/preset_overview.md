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

## `GenerationSettings` (`src/presets/Preset.hpp`)

Lightweight snapshot of current UI state; used as input to `PresetManager` methods.
Intentionally decoupled from `SettingsPanel` to avoid circular includes.

Fields mirror `Preset` plus `presetId` (empty string if no preset is active).

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

---

# Integration

## Applying a preset to the settings panel

```cpp
void applyPresetToSettings(const Preset& preset, SettingsPanel& panel);
```

Free function declared in `PresetManager.hpp`, defined in `PresetManager.cpp`.

Sets:
- `panel.positiveArea` / `panel.negativeArea` via `setText()`
- `panel.generationParams.numSteps`, `guidanceScale`, `width`, `height`
- `panel.selectedModelIdx` (linear scan of `panel.availableModels` by folder name)
- `panel.activePresetId`

If `modelId` is not found in `availableModels`, `selectedModelIdx` is left unchanged and a warning is logged.

## Building a GenerationSettings snapshot from the view

The controller's `buildGenerationSettings(view)` helper captures current state:

```cpp
GenerationSettings gs;
gs.prompt         = sp.positiveArea.getText();
gs.negativePrompt = sp.negativeArea.getText();
gs.modelId        = filesystem::path(sp.getSelectedModelDir()).filename().string();
gs.steps          = sp.generationParams.numSteps;
gs.cfg            = sp.generationParams.guidanceScale;
gs.width          = sp.generationParams.width;
gs.height         = sp.generationParams.height;
gs.presetId       = sp.activePresetId; // empty if none
```

## Tracking active preset

`SettingsPanel::activePresetId` (empty string) indicates which preset is currently applied.
Set by `applyPresetToSettings`; the controller also sets it after `createFromGeneration`.
`MenuBar` displays a ✓ next to the active preset in the dropdown.

---

# ID generation

IDs are millisecond-precision Unix timestamps prefixed with `"preset_"`.
Collision is not possible in normal interactive use.
