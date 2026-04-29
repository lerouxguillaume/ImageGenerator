# Isometric Asset Pack Implementation Plan

Goal: turn the current project workspace into a tool that helps a user produce game-ready isometric 2D assets with minimal cleanup before adding them to a game.

This plan assumes the primary target is:
- stylized isometric 2D game assets
- transparent-background deliverables
- pack-level visual consistency
- low manual prompt work

---

# Product objective

The tool should optimize for:

1. defining a pack-wide theme once
2. generating asset types with game-safe defaults
3. producing consistent variants quickly
4. normalizing outputs so they are closer to drop-in game assets
5. showing progress at the pack level, not only per image

---

# Phase 1 — Game-ready generation constraints ✓ DONE

## Goal

Make the system enforce the rules that users currently have to remember manually.

## Implemented

**`PackConstraints`** (on `Project`) — 6 boolean flags:
- `transparentBg` → positive: `transparent background`
- `isometricAngle` → positive: `isometric view`, `isometric perspective`
- `centeredComposition` → positive: `centered composition`
- `subjectFullyVisible` → positive: `full object visible`
- `noEnvironmentClutter` → negative: `background`, `environment`, `clutter`
- `noFloorPlane` → negative: `floor plane`, `ground plane`, `shadow`

**`AssetConstraints`** (on `AssetType`) — 4 boolean flags:
- `allowFloorPlane` — suppresses `noFloorPlane` for this asset type
- `allowSceneContext` — suppresses `noEnvironmentClutter` for this asset type
- `tileableEdge` → positive: `seamless edges, tileable`
- `topSurfaceVisible` → positive: `top surface visible`

Prompt layering order: `stylePrompt → constraintTokens → assetTypeTokens → userDsl`

UI: pack constraint chips (2 rows × 3) in the theme box; asset constraint chips (2 × 2) in the asset detail panel. Toggling auto-saves immediately.

Persisted to `projects.json` with `false` defaults for backward compatibility.

→ See `docs/75_projects/project_overview.md` for full data model and invariants.

## Files changed

- `src/projects/Project.hpp`
- `src/projects/ProjectManager.cpp`
- `src/controllers/ProjectController.cpp`
- `src/controllers/ImageGeneratorController.cpp`
- `src/views/ProjectView.hpp`
- `src/views/ProjectView.cpp`

---

# Phase 2 — Asset-type templates

## Goal

Remove blank-page prompting for common isometric pack pieces.

## Work

- Introduce built-in asset templates for common categories:
  - wall
  - floor tile
  - corner wall
  - door
  - stairs
  - prop
  - tree
  - rock
  - torch
  - roof piece
- Each template should carry:
  - starter subject prompt
  - starter negative prompt
  - recommended constraints
  - optional tags such as `tileable`, `modular`, `transparent background`
- Add a `New Asset Type From Template` flow in the project workspace.
- Preserve support for fully custom asset types.

## Files likely affected

- `src/projects/*`
- `src/controllers/ProjectController.*`
- `src/views/ProjectView.*`
- new template definitions under a small dedicated module, likely `src/projects/` or `src/prompt/`

## Acceptance criteria

- The user can add a new asset type from a template in one action.
- Generated starter prompts are visibly better than starting from empty text areas.
- Templates remain editable after creation.

---

# Phase 3 — Pack consistency controls

## Goal

Make consistency a first-class feature rather than an implicit hope.

## Work

- Extend the project theme model with structured style controls:
  - material language
  - palette keywords
  - detail density
  - outline/edge preference
  - lighting direction
  - stylization level
- Add optional “style lock” toggles for:
  - palette consistency
  - silhouette cleanliness
  - lighting consistency
  - camera/framing consistency
- Show a compact “effective pack style” summary in the project workspace.
- Use these structured settings when building prompts for all asset types.

## Files likely affected

- `src/projects/Project.hpp`
- `src/projects/ProjectManager.cpp`
- `src/views/ProjectView.*`
- `src/controllers/ProjectController.*`
- `src/prompt/*`

## Acceptance criteria

- Two different asset types generated from the same project look materially closer in style than they do today.
- The user can inspect and edit pack-wide consistency rules without digging through long freeform prompts.

---

# Phase 4 — Variant generation workflow

## Goal

Help the user generate controlled variations from an existing asset result with minimal friction.

## Work

- Add quick variant actions from a selected result:
  - more damaged
  - cleaner
  - mossy
  - icy
  - ruined
  - darker material
  - lighter material
- Support “generate N variants from this asset” in the project workspace.
- Treat variant requests as a structured delta layered on top of the asset type, not as a full rewrite.
- Store variant provenance so the user can tell which result was derived from which base.

## Files likely affected

- `src/ui/widgets/ResultPanel.*`
- `src/controllers/ProjectController.*`
- `src/controllers/ImageGeneratorController.*`
- possibly `src/projects/*` if variant history is persisted

## Acceptance criteria

- The user can produce multiple related variants from a selected asset without leaving the project workflow.
- Variant prompts are derived from the selected asset context rather than typed from scratch each time.

---

# Phase 5 — Post-processing for game readiness

## Goal

Reduce cleanup work between generation and engine import.

## Work

- Add post-processing options:
  - trim transparent bounds
  - add configurable padding
  - center subject on canvas
  - normalize export canvas size
  - optional scale-to-target sprite size
  - alpha cleanup / edge cleanup
- Add project-level export defaults:
  - target sprite canvas
  - padding
  - naming scheme
- Keep original generated image and processed export as separate outputs where useful.

## Files likely affected

- `src/controllers/ImageGeneratorController.*`
- `src/controllers/ProjectController.*`
- new image post-processing helpers, likely under `src/ui/` or a new utility module
- project persistence files

## Acceptance criteria

- The user can export a centered, padded, transparent asset without external tooling.
- Post-processing is optional but easy to apply consistently across a pack.

---

# Phase 6 — Pack coverage tracking

## Goal

Make the workspace track progress as a pack, not only as a gallery.

## Work

- Add asset-type status tracking:
  - not started
  - draft
  - ready
  - needs variants
- Show counts:
  - number of generated results
  - number of approved exports
  - number of variants
- Add compact pack progress UI in the project workspace.
- Support filtering by missing/incomplete asset types.

## Files likely affected

- `src/projects/*`
- `src/controllers/ProjectController.*`
- `src/views/ProjectView.*`

## Acceptance criteria

- The user can see what is missing in the pack at a glance.
- The project workspace feels like a production tracker, not just a prompt launcher.

---

# Phase 7 — Export pipeline

## Goal

Let the user produce a game-usable asset pack with consistent filenames and structure.

## Work

- Add export actions:
  - export selected asset
  - export current asset type
  - export full pack
- Normalize file naming:
  - `<asset_type>__v001.png`
  - `<asset_type>__variant_mossy__v002.png`
- Export metadata file for the pack:
  - project name
  - asset types
  - source image path
  - variant label
  - canvas/export settings
- Keep export folders separate from raw generation history where practical.

## Files likely affected

- `src/controllers/ProjectController.*`
- `src/projects/*`
- export utility modules

## Acceptance criteria

- The user can export a clean pack without manually renaming or reorganizing files.
- Pack metadata is sufficient to reproduce or audit outputs later.

---

# Suggested implementation order

Recommended order:

1. Phase 1 — game-ready generation constraints
2. Phase 2 — asset-type templates
3. Phase 4 — variant generation workflow
4. Phase 5 — post-processing for game readiness
5. Phase 6 — pack coverage tracking
6. Phase 3 — pack consistency controls
7. Phase 7 — export pipeline

Rationale:
- Phase 1 and 2 reduce prompt burden immediately.
- Phase 4 and 5 reduce the time from image generation to usable output.
- Phase 6 improves project usability once the pack has enough content to manage.
- Phase 3 is important, but some of it may be informed by what proves necessary in Phases 1, 2, and 4.
- Phase 7 should happen once the asset workflow is stable enough to export consistently.

---

# Recommended first milestone

If only one short milestone is implemented next, it should be:

## Milestone A

- Phase 1 subset: ✓ DONE (all 6 pack constraints + 4 asset constraints implemented)
- Phase 2 subset:
  - wall
  - floor tile
  - prop
  - door
- small project UI additions to expose these defaults clearly ✓ DONE

## Why

This gives the user immediate value:
- fewer bad generations
- faster asset-type setup
- less prompt repetition
- better “drop into the game” readiness

---

# Risks and design constraints

- Do not keep adding freeform prompt fields. Prefer structured settings where behavior is repeated and predictable.
- Do not overload `ImageGeneratorView` with project-specific UI. `ProjectView` should remain a first-class workspace.
- Avoid mixing raw history, approved outputs, and exported assets into one folder view without clear status semantics.
- Keep project defaults and runtime generation overrides distinct.
- Prefer project-level conventions over one-off per-run tweaks where possible.

---

# Open questions

- Should approval/final selection be explicit per result before export?
- Should templates live in code, JSON data, or project presets?
- Should post-processing produce sidecar exports or replace the displayed result?
- How much variant history needs to be persisted in `projects.json`?
- Should pack coverage status be derived automatically from output folders or stored explicitly in project data?
