# Isometric Asset Pack Implementation Plan

Goal: evolve the current project workspace from a prompt-first image generator into a reproducible, constrained production tool for game-ready isometric asset production and replacement.

The key product shift is this:
- current workflow: "generate a nice image for this prompt"
- target workflow: "produce a usable asset that fits an existing game slot"

This plan assumes the primary target is:
- stylized isometric 2D game assets
- transparent-background deliverables
- pack-level visual consistency
- both net-new assets and slot-compatible replacement assets
- low manual cleanup before engine import

---

# Product objective

The tool should optimize for:

1. defining pack-wide visual language once
2. defining hard asset-slot requirements per asset type
3. generating assets that respect existing silhouette/orientation constraints
4. validating that outputs are actually usable in-game
5. exporting consumer-ready assets with minimal external tooling

The system should treat each asset as a production artifact with:
- a target canvas size
- a consistent visual scale
- a target orientation
- a target occupied shape or silhouette
- a target anchor point
- background and export requirements
- style consistency requirements

Prompt quality still matters, but it is no longer the main control surface.

---

# Core asset model

Each asset should be modeled as a constrained production artifact, not just prompt text.

Core fields:
- fixed canvas size
- consistent pixel density or scale expectations
- defined anchor point
- orientation / isometric plane
- expected occupied bounds
- transparency requirements
- shape policy:
  - free
  - bounded
  - locked
- export constraints

Prompt becomes a secondary control layered on top of the asset contract.

---

# Critical invariants

These should be enforced early because they shape both the data model and the pipeline.

## 1. Scale consistency

Assets in the same pack should read at a consistent visual scale.

This should be modeled as:
- target fill ratio
- acceptable fill ratio range
- optional future pixel-density hints

Validation should catch:
- assets that are too zoomed in
- assets that are too small within the canvas
- assets that break pack-scale consistency

## 2. Single final alpha authority

Final transparency must be deterministic at the export stage.

Exactly one final alpha source should govern the exported asset:
- reference or supplied mask
- derived alpha
- validated post-processed alpha

The system should not mix multiple alpha sources implicitly.

## 3. Deterministic reference normalization

All structural references should go through the same preprocessing path:

`input -> resize -> center -> mask extraction -> aligned tensor`

Direct raw reference usage should be avoided because it breaks reproducibility and makes validation harder.

## 4. Structure over style

For constrained assets, structural constraints must dominate style variation.

This principle should hold across prompt compilation, img2img strength control, mask handling, and validation.

## 5. Strict vs flexible validation

Not every asset type should enforce the same level of rigidity.

Validation policies should allow per-asset configuration of:
- required canvas size
- required transparency
- silhouette enforcement
- acceptable silhouette deviation
- fill ratio tolerance
- anchor enforcement

---

# Current gap summary

What is missing today is not another prompt enhancer. The missing pieces are the controls that turn image generation into asset production:

1. reference-driven generation for constrained asset classes
2. stronger shape control during generation, not only after the fact
3. raw-vs-processed review tools for debugging generation vs normalization failures
4. in-context preview against the target slot
5. consumer-facing export conventions beyond the current processed output

Several earlier foundational items are now implemented:

- formal asset specifications
- locked asset-type prompt constraints
- transparent background cleanup
- deterministic post-processing/export normalization
- post-generation validation
- bounds/anchor overlay review

These requirements drive the phases below.

---

# Current implementation snapshot

The current project workflow is no longer just:

`prompt -> image`

It now behaves roughly like:

`prompt -> raw image -> optional alpha cleanup -> normalized processed PNG -> metadata -> validation`

Implemented today:

- `AssetSpec` persisted on each asset type
- `AssetExportSpec` persisted on each asset type
- template-driven constrained asset defaults (including `wall_left`)
- locked prompt compilation from orientation / shape / transparency / tileability
- transparent background cleanup via `AlphaCutout`
- deterministic post-processing via `AssetPostProcessor`
- raw and processed outputs saved separately in project asset folders
- metadata JSON per processed asset
- processed outputs shown in the project gallery
- validation for canvas, alpha, fill, bounds, and anchor
- preview overlays for expected bounds and anchor

The biggest remaining weakness is still generation-time structure drift for difficult assets such as:

- left walls
- right walls
- doors
- corner walls
- stairs

---

# Cross-phase pipeline work

The phases below describe product capabilities, but several pipeline capabilities cut across all of them. These should be treated as first-class implementation tracks, not incidental follow-up work.

## 1. Reference conditioning modes

The pipeline should support more than prompt-only generation. The initial supported contract should stay intentionally narrow:
- txt2img
- img2img from a source asset
- masked img2img

This matters especially for modular or slot-constrained assets where text alone is not enough to preserve usable geometry.

Do not start with:
- multi-reference blending
- complex style-transfer modes
- loosely defined reference behaviors

## 2. Mask ingestion and normalization

Masks should be a real pipeline input, not just a UI concept.

Required support:
- alpha mask ingestion
- silhouette mask ingestion
- occupied-bounds mask support
- deterministic resizing and rasterization rules before inference

This is necessary for preserving shape, controlling background removal, and validating fit.

## 3. Transparent-output strategy

Transparency should be defined as a pipeline concern rather than an export-only detail.

The implementation needs a clear strategy for one or more of:
- native transparent-background generation where supported
- mask-guided cutout
- background removal pass
- hybrid generation plus post-processing

The chosen strategy affects both generation quality and cleanup complexity.

## 4. Separate structure and style control

The current strength concept is too coarse for constrained asset production.

The pipeline should distinguish between:
- structure preservation strength
- style variation strength
- mask adherence strength

Without that separation, it is hard to iterate on materials or polish without breaking composition.

## 5. Deterministic refinement workflow

Asset production needs reproducible iteration.

The pipeline should make it easy to:
- reuse seeds
- regenerate from a selected candidate
- preserve structure while varying materials, surface detail, or lighting
- compare revision output against a known baseline

This is important for review, approval, and pack consistency work.

## 6. Metadata emission for validation

Validation should not rely only on UI-side guesses. The pipeline should emit machine-usable metadata such as:
- output canvas size
- alpha presence and alpha bounds
- occupied content bounds
- fill ratio
- optional silhouette deviation metrics
- generation mode used
- seed used
- reference or mask inputs used

This metadata can drive validation, export checks, and contextual preview.

## 7. Post-processing as a formal pipeline stage

Post-processing should be modeled as a named and testable stage, not a loose collection of export options.

Expected operations include:
- trim transparent bounds
- add padding
- center subject
- normalize canvas size
- alpha cleanup
- export conversion

This stage should be composable and reusable across asset types.

## 8. Preview artifact generation

If contextual review is a product requirement, the pipeline should be able to produce artifacts that support it:
- raw candidate image
- processed export image
- overlay preview against source or mask
- context composite preview inputs

This reduces ad hoc logic in the UI layer.

## 9. Asset-profile-aware prompt compilation

Prompt compilation should become more aware of asset class.

Different asset profiles should be able to compile prompts differently, for example:
- freeform asset
- slot-constrained asset
- tileable asset
- isolated transparent object

This prevents one general prompt strategy from overfitting all asset types.

## 10. Constrained-asset failure handling

The pipeline should define failure behavior for common constrained-asset problems:
- missing reference image
- mask and canvas mismatch
- no usable transparency in output
- silhouette drift beyond threshold
- post-processing that removes too much content

These should surface as actionable errors or warnings rather than silent bad output.

---

# Phase 1 — Asset slot specification ✓ DONE

## Goal

Make each asset type describe the contract the generated asset must satisfy.

## Why this is required

Some assets are freeform pack pieces. Others are strict replacements for an existing runtime slot. Without a formal asset spec, the generator optimizes for appearance rather than compatibility.

## Work

Add a structured asset specification model to project asset types.

Each asset type should be able to define:
- target canvas width and height
- anchor point
- orientation / plane:
  - left wall
  - right wall
  - floor tile
  - prop
  - character
- expected occupied bounds
- target fill ratio
- acceptable fill ratio range
- transparency requirement
- shape policy:
  - freeform
  - bounded
  - silhouette-locked
- asset fit mode:
  - object-fit
  - tile-exact
  - no-resize
- validation policy
- export format preference

Optional but useful:
- intended render scale in the consumer game
- minimum readable detail level
- mirrorability flag

## UI work

In the project workspace, each asset type should expose a compact "Asset Spec" section with:
- canvas size inputs
- anchor editor
- orientation selector
- fill ratio controls
- transparency toggle
- shape policy selector

## Files likely affected

- `src/projects/Project.hpp`
- `src/projects/ProjectManager.cpp`
- `src/controllers/ProjectController.*`
- `src/views/ProjectView.*`
- `docs/75_projects/project_overview.md`

## Acceptance criteria

- ✓ The user can define a replacement-target contract for each asset type.
- ✓ An asset type can express a slot equivalent to `door-left`, not just a prompt.
- ✓ The specification persists in `projects.json`.

## What was built

- `AssetSpec`, `ValidationPolicy`, `Orientation`, `ShapePolicy`, `AssetFitMode`, `Anchor`, `OccupiedBounds` added to `src/projects/Project.hpp`.
- `AssetType` carries an `AssetSpec spec` field and now also an `AssetExportSpec exportSpec`.
- Full JSON round-trip in `ProjectManager.cpp` with human-readable enum strings; old files load cleanly.
- All 6 built-in templates ship with tuned spec defaults (orientation, shape policy, fill range, tileable).
- `ProjectView` shows an "Asset spec" section per asset type: orientation radio (6 chips), Transparent/Tileable toggles, shape policy radio, plus numeric slot-contract fields for bounds and anchor. Changes persist immediately.
- `AssetSpec` is forwarded into `ResolvedProjectContext` and is now used by prompt locking, validation, preview overlays, and transparency handling.
- `AssetExportSpec` now drives deterministic post-processing after generation.

---

# Phase 2 — Reference-driven constrained img2img (NEXT)

## Goal

Make the generator start from the target asset shape instead of guessing it from text.

## Why this is required

Prompt-only generation is too loose for modular wall pieces, door inserts, tile-aligned props, and other structurally constrained assets. For those cases, structure must come from a reference, mask, or template. Freeform asset types should still be allowed to skip this path.

## Work

Add support for per-asset structural references for constrained asset classes.

Initial scope should stay narrow:
- persisted `referenceImagePath`
- persisted `referenceEnabled`
- persisted `structureStrength`
- deterministic reference normalization to the generation canvas
- project-asset generation routed to the existing img2img path when reference mode is enabled

Do not include yet:
- SVG rasterization
- user-authored masks
- multi-reference inputs
- silhouette metrics
- ControlNet-like branches

Generation modes for this phase:
- text-only
- img2img from reference asset

## UI work

Add a compact "Reference Shape" block in the asset detail panel:
- source asset path / picker
- reference enable/disable
- structure-strength control

Recommended default asset classes for V1:
- `wall_left` enabled
- `wall_right` enabled
- `door` enabled
- `corner_wall` enabled
- `stairs` enabled
- `floor_tile` disabled
- `prop` disabled
- `character` disabled

## Files likely affected

- `src/projects/Project.hpp`
- `src/projects/ProjectManager.cpp`
- `src/projects/AssetTypeTemplate.*`
- `src/assets/ReferenceNormalizer.*`
- `src/controllers/ImageGeneratorController.*`
- `src/controllers/ProjectController.*`
- `src/views/ProjectView.*`
- pipeline docs under `docs/10_pipeline/`

## Acceptance criteria

- The user can attach or use a default reference image as a structural reference.
- The generator can preserve left-wall vs right-wall composition instead of drifting into a new scene.
- Asset types that do not need references can continue using prompt-first txt2img.
- The existing post-process/export path remains unchanged.
- Raw and processed outputs remain inspectable.

---

# Phase 3 — Asset-type locked prompt constraints ✓ DONE

## Goal

Move critical geometry and composition rules out of freeform prompting and into hard asset-type constraints.

## Why this is required

For slot-compatible assets, the asset type should own more of the generation intent than the ad hoc prompt. The user should refine style and material, not restate geometry every time.

## Work

Extend asset templates and project constraints so asset types can inject locked prompt behavior:
- orientation locks
- "single isolated asset" locks
- "transparent background" locks
- "no environment" locks
- "no floor plane" locks
- silhouette-preservation hints
- framing constraints

Example asset-type policies:
- `door-left` always means left wall plane, narrow wall segment, no room scene
- `door-right` always means mirrored equivalent
- `wall-left` and `wall-right` preserve shared structural proportions

Prompt assembly should distinguish:
- pack-wide style tokens
- asset-type locked structural tokens
- user-editable refinement tokens
- negative constraints

## Files likely affected

- `src/projects/AssetTypeTemplate.*`
- `src/controllers/ImageGeneratorController.*`
- `src/controllers/ProjectController.*`
- `src/prompt/*`
- `docs/85_prompt/prompt_dsl.md`

## Acceptance criteria

- ✓ Users no longer need to repeatedly type orientation and isolation constraints.
- ✓ Asset templates are materially better at producing slot-compatible assets than a blank prompt.
- ✓ The generator is less likely to invent background scenes for modular pieces.

## What was built

`buildConstraintTokens()` in `ProjectController.cpp` now accepts `AssetSpec` as a third argument and injects locked tokens automatically:

| Spec field | Positive | Negative |
|---|---|---|
| `requiresTransparency` | `transparent background` | — |
| `isTileable` | `seamless edges, tileable` | — |
| `LeftWall` | `isometric left wall`, `left-facing wall plane` | `floor plane`, `ground plane`, `shadow` (unless `allowFloorPlane`) |
| `RightWall` | `isometric right wall`, `right-facing wall plane` | same |
| `FloorTile` | `isometric floor plane`, `top-down surface` | — |
| `Prop` | `single isolated prop` | `background scene`, `room`, `environment` (unless `allowSceneContext`) |
| `Character` | `isometric character`, `facing camera` | — |
| `Bounded` | `single isolated subject`, `full object visible` | `cropped`, `partially visible` + environment negatives |
| `SilhouetteLocked` | same + `exact silhouette preserved` | same |

`requiresTransparency` and `isTileable` fire unconditionally — the spec is self-contained and merge deduplicates any overlap with pack/asset constraint tokens. The existing `allowFloorPlane` / `allowSceneContext` overrides still suppress the corresponding negatives.

Because `constraintTokens` is compiled to a string for session sync comparison, any spec change (orientation, shape policy, toggles) automatically triggers a session refresh in `syncGeneratorSession` — no extra logic needed.

---

# Phase 4 — Transparent asset workflow ✓ DONE

## Goal

Produce actual game-usable assets instead of opaque full-frame images.

## Why this is required

Many game assets need to sit cleanly on top of a scene or tile map. The pipeline therefore needs a proper transparency path, whether the asset is a replacement piece or a newly authored pack element.

## Work

Add a transparent-output workflow with:
- transparent background prompting where the model supports it
- optional background removal pass
- alpha edge cleanup
- transparent padding normalization
- export preview on checkerboard

Keep both:
- raw generated image
- processed transparent export

This phase should explicitly support isolated asset extraction, even when the base model does not natively produce perfect transparency.

## Files likely affected

- `src/controllers/ImageGeneratorController.*`
- `src/controllers/ProjectController.*`
- new image post-processing helpers
- project persistence files
- export docs

## Acceptance criteria

- ✓ The user can produce a transparent asset export from a project workflow.
- ✓ Edge halos are reduced enough for in-game use.
- ✓ Transparency is visible and reviewable before export.

## What was built

**`src/postprocess/AlphaCutout`** — new module implementing corner flood-fill background removal:
1. Estimates background colour from 3×3 regions at all four corners
2. BFS flood-fill from all four corners with configurable Euclidean colour distance tolerance (default 30)
3. Optional 1px foreground erosion (defringe) to eliminate halo fringe pixels
4. Multi-pass feathering: alpha blur at the fg/bg boundary, only allowing alpha to decrease

**File naming convention** — transparent derivatives are saved as `<stem>_t.png` alongside the raw image. Gallery scanning excludes `_t.png` files; they are treated as hidden derivatives.

**Integration:**
- `ResolvedProjectContext` now carries `AssetSpec spec` so controllers can read `requiresTransparency` without re-querying `ProjectManager`
- `launchGeneration` captures `requiresTransparency` and runs `AlphaCutout::removeBackground` in the generation thread after `generateFromPrompt` returns, for all images in the batch
- `selectGalleryImage` prefers the `_t.png` sibling when `projectContext_.spec.requiresTransparency` is true
- `ResultPanel` renders a two-tone checkerboard (200/155 grey) behind the image preview when `showCheckerboard=true`
- `ProjectController::syncGeneratorSession` sets `showCheckerboard` from `ctx.spec.requiresTransparency` every frame

---

# Phase 5 — Validation and compatibility checks ✓ DONE

## Goal

Check whether a generated asset is actually compatible with its target slot before it is accepted.

## Why this is required

A visually good asset can still fail production requirements:
- wrong silhouette
- wrong occupied bounds
- wrong anchor
- opaque background
- unreadable detail at runtime scale

## Work

Add post-generation validation rules for project assets:
- canvas size check
- alpha presence / transparency check
- fill ratio check
- occupied-bounds check
- silhouette overlap or deviation check
- anchor compatibility check
- orientation consistency check

Validation output should be lightweight but actionable:
- pass
- warning
- fail

Common validation and pipeline failure cases should be handled explicitly:
- missing reference image
- mask and canvas mismatch
- no usable transparency in output
- silhouette drift beyond threshold
- post-processing that removes too much content

## UI work

Show a small validation summary on selected results:
- canvas
- transparency
- fill ratio
- shape fit
- anchor fit
- export readiness

## Files likely affected

- `src/controllers/ProjectController.*`
- `src/controllers/ImageGeneratorController.*`
- `src/ui/widgets/ResultPanel.*`
- new validation helpers module

## Acceptance criteria

- ✓ The tool can flag an asset that looks good but does not fit its slot.
- ✓ The user can tell whether a result is game-ready without manually exporting and testing it first.

## What was built

**`src/postprocess/AssetValidator`** — new module with a single entry point `validate(sf::Image&, AssetSpec&) → Result`:

| Check | Pass | Warning | Fail |
|---|---|---|---|
| **Canvas** | dims match `spec.canvasWidth/Height` | mismatch + `!enforceCanvasSize` | mismatch + `enforceCanvasSize` |
| **Alpha** | any pixel `alpha < 255` (when `requiresTransparency`) | no transparent pixels + `!enforceTransparency` | no transparent pixels + `enforceTransparency` |
| **Fill** | opaque pixel ratio within `[minFillRatio, maxFillRatio]` | above `maxFillRatio` | below `minFillRatio` |
| **Bounds** | occupied bounds match expected slot within tolerance | drift for freeform / mild bounded drift | strong bounded drift |
| **Anchor** | bottom-center content anchor is near target anchor | anchor drift + `!enforceAnchor` | anchor drift + `enforceAnchor` |

Canvas check is skipped when both `canvasWidth` and `canvasHeight` are 0 (no canvas contract defined). Alpha check is skipped when `requiresTransparency` is false. Bounds and anchor checks operate on the selected displayed asset, which now means the processed project asset output when project mode is active.

`Result::exportReady()` returns `true` only when no check has `status == 2` (fail).

**`ResultPanel`** — `validationChips` vector added (`name`, `status`, `detail`). Chips render as a horizontal strip just below the image frame, spanning the full frame width. Color coding: green border = pass, gold = warning, red = fail. The strip only appears when at least one chip is present.

**`ImageGeneratorController::selectGalleryImage`** — now loads the displayed image as `sf::Image` first (to allow pixel-level inspection), creates the `sf::Texture` from it via `loadFromImage`, then runs `AssetValidator::validate` and populates `rp.validationChips`. Chips are cleared on deselect and when no project context is active.

---

# Phase 6 — Contextual preview and review loop

## Goal

Let the user judge asset fit in the context where the asset will actually be used.

## Why this is required

Prompt-only review is too indirect. Asset review needs context:
- overlay against the source silhouette
- anchor preview
- occupied-bounds preview
- target-context preview

## Work

Add a review mode for generated assets with:
- current asset vs candidate comparison
- overlay / onion-skin comparison
- mask and occupied-bounds overlay
- anchor visualization
- preview in the target context where available

The core system should support context-aware preview in a general way. Specific preview adapters can then be added for particular consumer contexts, such as isometric rooms, tile maps, or inventory presentation.

## UI work

The result panel should allow:
- compare to source
- compare to approved asset
- preview in context
- approve
- reject
- regenerate from this candidate

## Files likely affected

- `src/views/ProjectView.*`
- `src/ui/widgets/ResultPanel.*`
- `src/controllers/ProjectController.*`
- possibly dedicated preview helpers

## Acceptance criteria

- The user can review an asset in context before exporting it.
- Anchor and silhouette mistakes are visible immediately.
- The workflow supports iterative replacement rather than blind batch generation.

---

# Phase 7 — Consumer-ready export

## Goal

Export assets directly into the format expected by the game or downstream repo.

## Why this is required

The final output is not just "an image in the gallery". It is a file that can be imported directly into a game or UI runtime, sometimes as a replacement for an existing asset, sometimes as a new asset under a project-defined convention.

## Work

Add export controls for:
- exact target dimensions
- output format (`png`, `webp`)
- naming rules
- optional sidecar metadata for anchor / footprint
- output directory mapping by asset type

Support an export profile per project or target consumer app.

For example, an asset profile should be able to say:
- export as `door_left.png`
- canvas `128x256`
- anchor metadata `96,224`
- save under a configured destination tree

## Files likely affected

- `src/projects/*`
- `src/controllers/ProjectController.*`
- `src/controllers/ImageGeneratorController.*`
- export helpers

## Acceptance criteria

- The user can export a generated asset as a direct replacement candidate.
- Export output matches the target slot contract without manual renaming or resizing.
- Multiple asset types can share a project-level export convention.

---

# Phase 8 — Pack consistency and coverage tracking

## Goal

Track the asset pack as a production set, not just a gallery of unrelated outputs.

## Why this is still needed

Even with slot-compatible generation, pack production still needs consistency and progress tracking across many assets.

## Work

Retain and extend pack-level controls:
- material language
- palette keywords
- detail density
- lighting direction
- stylization level

Add production tracking per asset type:
- not started
- in progress
- candidate ready
- approved
- exported

Show counts:
- generated candidates
- validated candidates
- approved exports
- missing asset types

## Files likely affected

- `src/projects/*`
- `src/controllers/ProjectController.*`
- `src/views/ProjectView.*`

## Acceptance criteria

- The user can see what is missing from the pack at a glance.
- Pack-wide style rules remain easy to inspect and edit.
- The workspace behaves like a lightweight asset production tracker.

---

# Implementation priority

The most useful rollout is foundation-first. Pipeline structure should land before higher-level workflow polish.

## Phase A — Foundation

1. ✓ Asset specification model
2. Reference normalization
3. ✓ Metadata emission
4. ✓ Post-processing pipeline stage

## Phase B — Control

1. Simple img2img reference workflow
2. Mask ingestion and normalization
3. ✓ Asset-type locked prompt constraints

## Phase C — Usability

1. ✓ Transparent asset workflow
2. ✓ Basic validation:
   - ✓ canvas
   - ✓ alpha
   - ✓ fill ratio
   - ✓ bounds
   - ✓ anchor
3. ✓ Candidate export preview via processed project outputs

## Phase D — Iteration

1. Overlay preview:
   - source vs candidate
   - ✓ bounds
   - ✓ anchor
2. Candidate-to-reference refinement loop
3. Context preview adapters where needed

## Phase E — Advanced

1. Silhouette validation
2. Context-specific preview adapters
3. Pack tracking and approval state

That order gets the tool from "generate asset-like images" to "produce plausible replacement assets" with the least wasted work.

## Immediate next milestone

The next milestone should be:

**Wall generation hardening**

The repo already has the first reference-driven step for `wall_left`:

- persisted reference state on `AssetType`
- built-in `wall_left` reference asset
- deterministic `ReferenceNormalizer`
- automatic routing into the existing img2img path
- metadata fields for reference usage

So the next highest-value work is to make wall generation usable and debuggable from the UI rather than widening scope immediately to every constrained asset class.

Current focus:

- `wall_left`

User-facing gain:

- compare prompt-only vs reference-driven walls
- tune structure rigidity without touching code
- tell whether a bad result came from generation or post-processing

Implemented in the current pass:

1. `Ref Shape On/Off` control for the active asset
2. `Structure strength` slider
3. `Processed / Raw` preview toggle
4. reference-use status in the result panel

Next step for the wall-only focus:

1. run `wall_left` A/B comparisons with reference on vs off
2. tune the `wall_left` reference image and default `structureStrength`
3. tighten the wall prompt/defaults only where the A/B results show drift
4. only after that, extend the same workflow to `door`, `corner_wall`, and `stairs`

---

# Non-goals for the first pass

The first pass should not try to solve every art-pipeline problem at once.

Explicit non-goals:
- full vector regeneration
- automatic rigging / animation support
- arbitrary consumer-engine integrations
- procedural tileset authoring
- complete asset library management

The first pass only needs to make replacement-grade static 2D isometric assets practical.

---

# Final principle

`input constraints -> controlled generation -> normalized output -> validated asset`

Not:

`prompt -> image -> hope it works`
