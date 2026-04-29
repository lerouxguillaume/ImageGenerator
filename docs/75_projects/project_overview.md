# Project System Overview

The project system organises generation work into **asset packs**. Each pack holds a shared style theme, a list of asset types, and a set of generation constraints that compile into prompts automatically.

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
    └── AssetConstraints — per-asset overrides and additions (see below)
```

Key files:
- `src/projects/Project.hpp` — all structs
- `src/projects/ProjectManager.cpp` — JSON persistence (`projects.json`)
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

## ResolvedProjectContext

Carries everything `ImageGeneratorController` needs for one generation session:

| Field | Source |
|---|---|
| `projectId`, `projectName` | `Project` |
| `assetTypeId`, `assetTypeName` | `AssetType` |
| `stylePrompt` | `Project::stylePrompt` (raw, no constraints baked in) |
| `constraintTokens` | compiled from `PackConstraints` + `AssetConstraints` |
| `assetTypeTokens` | `AssetType::promptTokens` |
| `outputSubpath` | `sanitiseName(project) / sanitiseName(assetType)` |
| `allAssetTypes` | all asset types in the project (used to populate gallery tabs) |

`constraintTokens` is recomputed every frame in `buildSelectedContext()` and compared via compiled string equality in `syncGeneratorSession()` to trigger session refresh when any toggle changes.

---

## Persistence

All fields serialize to `projects.json` (array of project objects). Constraint fields use `value(..., false)` defaults so older project files load cleanly with all constraints off.

---

## Invariants

- Never access `ProjectManager` from `ImageGeneratorController` — use `ResolvedProjectContext` as the data carrier.
- Never bake constraint tokens into `stylePrompt` on the project struct — keep them separate so text areas show only user-authored content.
- `ProjectController` is the only place that reads `ProjectManager` and builds `ResolvedProjectContext`.
