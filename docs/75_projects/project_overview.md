# Project System Overview

The project system organises generation work into **asset packs**. Each pack holds a shared style theme, a list of asset types, and a set of generation constraints that compile into prompts automatically.

Asset types can now be created either:
- from `Blank`
- from built-in templates in the project workspace `+ Asset` picker

---

## Data model

```
Project
├── id, name, createdAt
├── stylePrompt       — shared Prompt DSL applied to every asset in the pack
├── modelId, loraEntries, width, height
├── PackConstraints   — pack-wide generation rules (see below)
└── assetTypes[]
    ├── id, name
    ├── promptTokens  — asset-specific Prompt DSL layered on top of stylePrompt
    ├── AssetConstraints — per-asset overrides and additions (see below)
    ├── AssetSpec     — formal production contract (see below)
    ├── AssetExportSpec — deterministic post-process/export contract (see below)
    └── reference state — optional reference-driven img2img settings
```

Key files:
- `src/projects/Project.hpp` — all structs
- `src/projects/ProjectManager.cpp` — JSON persistence (`projects.json`)
- `src/projects/AssetTypeTemplate.*` — built-in asset template registry
- `src/controllers/ProjectController.cpp` — resolution and UI handling

---

## PackConstraints

Defined on `Project`. All fields default to `false`.

| Field | Positive tokens injected | Negative tokens injected |
|---|---|---|
| `transparentBg` | `transparent background` | — |
| `isometricAngle` | `isometric view`, `isometric perspective` | — |
| `centeredComposition` | `centered composition` | — |
| `subjectFullyVisible` | `full object visible` | — |
| `noEnvironmentClutter` | — | `background`, `environment`, `clutter` |
| `noFloorPlane` | — | `floor plane`, `ground plane`, `shadow` |

---

## AssetConstraints

Defined on `AssetType`. All fields default to `false`.

| Field | Effect |
|---|---|
| `allowFloorPlane` | Suppresses `PackConstraints::noFloorPlane` for this asset type |
| `allowSceneContext` | Suppresses `PackConstraints::noEnvironmentClutter` for this asset type |
| `tileableEdge` | Adds `seamless edges, tileable` to positive |
| `topSurfaceVisible` | Adds `top surface visible` to positive |

Override semantics are applied in `buildConstraintTokens()` in `ProjectController.cpp` before tokens are emitted.

---

## Prompt layering

At generation time the effective prompt is assembled in this order:

```
stylePrompt
  + constraintTokens        ← compiled from PackConstraints + AssetConstraints
  + assetTypeTokens
  + userDsl                 ← what the user typed in the generator text areas
```

Each layer is merged via `PromptMerge::merge(base, patch)` — patch tokens deduplicate against base, subject overrides if set, weights are taken from the patch on collision.

The compiled constraint tokens are stored in `ResolvedProjectContext::constraintTokens` and merged in `ImageGeneratorController.cpp` during generation setup.

---

## AssetSpec

Defined on `AssetType`. Describes the formal production contract the generated asset must satisfy.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `canvasWidth` / `canvasHeight` | `int` | `0` | Override canvas; `0` inherits from project |
| `anchor` | `Anchor{x,y}` | `{0,0}` | Pixel anchor point within the canvas |
| `orientation` | `Orientation` | `Unset` | Isometric plane the asset lives on |
| `expectedBounds` | `OccupiedBounds{x,y,w,h}` | `{0,0,0,0}` | Expected occupied region in canvas pixels |
| `targetFillRatio` | `float` | `0.6` | Ideal subject-to-canvas fill |
| `minFillRatio` / `maxFillRatio` | `float` | `0.3` / `0.9` | Acceptable fill range |
| `requiresTransparency` | `bool` | `true` | Asset must have a transparent background |
| `shapePolicy` | `ShapePolicy` | `Freeform` | How strictly shape must match a reference |
| `fitMode` | `AssetFitMode` | `ObjectFit` | How the image is fit to the target canvas |
| `isTileable` | `bool` | `false` | Asset must tile seamlessly |
| `validation` | `ValidationPolicy` | see below | Which checks to enforce |

### Orientation values
`Unset`, `LeftWall`, `RightWall`, `FloorTile`, `Prop`, `Character`

### ShapePolicy values
`Freeform` — no shape enforcement  
`Bounded` — subject must stay within expected bounds  
`SilhouetteLocked` — output silhouette must match a reference

### AssetFitMode values
`ObjectFit`, `TileExact`, `NoResize`

### ValidationPolicy
| Field | Default |
|---|---|
| `enforceCanvasSize` | `true` |
| `enforceTransparency` | `true` |
| `enforceSilhouette` | `false` |
| `enforceAnchor` | `false` |
| `maxSilhouetteDeviation` | `0.0` |

`AssetSpec` is fully persisted in `projects.json` under each asset type. Old project files load cleanly — all fields default gracefully.

---

## AssetExportSpec

Defined on `AssetType`. Describes how raw generated output is normalized after generation.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `exportWidth` / `exportHeight` | `int` | `128` / `128` | Exact processed output canvas |
| `maxObjectWidth` / `maxObjectHeight` | `int` | `112` / `112` | Object-fit limits inside the export canvas |
| `paddingPx` | `int` | `8` | Minimum padding target for `ObjectFit` |
| `fitMode` | `AssetFitMode` | `ObjectFit` | Post-process mode |
| `requireAlpha` | `bool` | `true` | Export should preserve transparency |

`AssetExportSpec` is persisted in `projects.json` under each asset type and is used by `AssetPostProcessor` after generation.

### AssetFitMode values
`ObjectFit`, `TileExact`, `NoResize`

---

## Reference-driven generation state

Defined on `AssetType`.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `referenceEnabled` | `bool` | `false` | Whether project generation should use the reference image via img2img |
| `referenceImagePath` | `std::string` | empty | Reference image path, relative to the working directory unless absolute |
| `structureStrength` | `float` | `0.45` | Img2img denoise strength used for structure preservation |

Current built-in behavior:

- `wall_left` ships with a default reference image:
  - `assets/references/wall_left.pgm`
- `wall_left` enables reference-driven img2img by default with `structureStrength = 0.34`

Generation only uses this path in project asset mode and only when the reference file exists.

Current wall-focused UI controls:

- `Ref Shape On/Off` toggle in the asset detail panel
- `Structure strength` slider, clamped to `0.30 - 0.60`
- result-panel `Processed` / `Raw` preview toggle

The result panel also shows whether the selected candidate used a reference-driven path and, when applicable, the structure strength used for that generation.

---

## Asset templates

Built-in templates are defined in code in `src/projects/AssetTypeTemplate.cpp`.

Current templates:
- `Wall Left` (`wall_left`) — `LeftWall`, `Bounded`, tileable, fixed generation canvas `512×768`, default reference enabled
- `Floor Tile` — `FloorTile`, `Bounded`, tileable, fill 0.6–0.95
- `Corner Wall` — `Unset`, `Bounded`, fill 0.4–0.85
- `Door` — `LeftWall`, `Bounded`, fill 0.45–0.8
- `Stairs` — `Unset`, `Bounded`, fill 0.45–0.85
- `Prop` — `Prop`, `Bounded`, fill 0.38–0.78

Each template provides:
- `id`, `label`, `defaultName`
- starter `Prompt` tokens
- starter `AssetConstraints`
- starter `AssetSpec` (orientation, shape policy, fill range, tileable flag)
- starter `AssetExportSpec`
- optional reference defaults
- tags for future UI use

Templates are **not** persisted directly. Only the created `AssetType` result is written to `projects.json`.

Creation flow:
- click `+ Asset` in `ProjectView`
- choose `Blank` or a built-in template
- if a template is chosen, `ProjectController` creates the asset type immediately using the template prompt tokens and constraints
- the resulting asset type remains fully editable like any other asset type

---

## ResolvedProjectContext

Carries everything `ImageGeneratorController` needs for one generation session:

| Field | Source |
|---|---|
| `projectId`, `projectName` | `Project` |
| `assetTypeId`, `assetTypeName` | `AssetType` |
| `stylePrompt` | `Project::stylePrompt` (raw, no constraints baked in) |
| `constraintTokens` | compiled from `PackConstraints` + `AssetConstraints` |
| `assetTypeTokens` | `AssetType::promptTokens` |
| `exportSpec` | `AssetType::exportSpec` — deterministic post-process/export contract |
| `referenceEnabled`, `referenceImagePath`, `structureStrength` | reference-driven img2img state copied from the active asset type |
| `outputSubpath` | `sanitiseName(project) / sanitiseName(assetType)` |
| `allAssetTypes` | all asset types in the project (used to populate gallery tabs) |
| `spec` | `AssetType::spec` — production contract forwarded from the active asset type |

`constraintTokens` is recomputed every frame in `buildSelectedContext()` and compared via compiled string equality in `syncGeneratorSession()` to trigger session refresh when any toggle changes.

---

## Persistence

All fields serialize to `projects.json` (array of project objects). Constraint fields use `value(..., false)` defaults so older project files load cleanly with all constraints off.

---

---

## AssetValidator

`src/postprocess/AssetValidator` runs pixel-level checks against `AssetSpec` immediately after an image is selected in the gallery.

Entry point: `AssetValidator::validate(const sf::Image&, const AssetSpec&) → Result`

Five checks are implemented:

| Check | Condition | Pass | Warning | Fail |
|---|---|---|---|---|
| Canvas | `canvasWidth > 0 \|\| canvasHeight > 0` | dims match spec | mismatch + `!enforceCanvasSize` | mismatch + `enforceCanvasSize` |
| Alpha | `requiresTransparency` | any pixel alpha < 255 | none found + `!enforceTransparency` | none found + `enforceTransparency` |
| Fill | always | ratio in `[minFillRatio, maxFillRatio]` | above max | below min |
| Bounds | `expectedBounds.w > 0 && expectedBounds.h > 0` | occupied bounds match slot within tolerance | drift for freeform / mild bounded drift | strong bounded drift |
| Anchor | `anchor.x != 0 \|\| anchor.y != 0` | bottom-center content anchor near target | drift + `!enforceAnchor` | drift + `enforceAnchor` |

`Result::exportReady()` returns `true` when no check is status 2 (fail).

Results are forwarded to `ResultPanel::validationChips` and rendered as a color-coded chip strip below the image frame: green = pass, gold = warning, red = fail. Chips are cleared when no image is selected or no project context is active.

When project asset mode is active, validation runs on the **processed** asset output shown in the gallery rather than the raw Stable Diffusion image.

If the user switches the result panel to `Raw`, validation chips are hidden and the raw generation is shown directly. This is intentionally a debugging view for checking whether a failure came from generation or from normalization.

---

## Post-processing output layout

For project assets, generation now writes:

```text
assets/generated/<project>/<asset>/
    raw/
        img_<id>.png
    processed/
        img_<id>.png
        img_<id>.json
```

The `processed/` variant is the normalized export candidate produced by `AssetPostProcessor`.

The project gallery scans `processed/` so review, overlays, and validation are aligned with the deterministic export candidate.

---

## Invariants

- Never access `ProjectManager` from `ImageGeneratorController` — use `ResolvedProjectContext` as the data carrier.
- Never bake constraint tokens into `stylePrompt` on the project struct — keep them separate so text areas show only user-authored content.
- `ProjectController` is the only place that reads `ProjectManager` and builds `ResolvedProjectContext`.
- `AssetSpec` drives prompt compilation via `buildConstraintTokens()` — orientation, shape policy, transparency, and tileability all inject locked tokens automatically.
- Spec toggle clicks call `projectManager_.updateAssetType()` directly — no dirty flag, no save button.
- `ResolvedProjectContext::spec` carries the active asset type's `AssetSpec` — `ImageGeneratorController` reads it for transparency, validation, and gallery display without querying `ProjectManager`.
