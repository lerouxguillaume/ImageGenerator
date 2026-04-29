# Isometric Asset Pack Implementation Plan

Goal: evolve the current project workspace from a prompt-first image generator into a production tool for game-ready isometric asset production and replacement.

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
- a target orientation
- a target occupied shape or silhouette
- a target anchor point
- background and export requirements
- style consistency requirements

Prompt quality still matters, but it is no longer the main control surface.

---

# Current gap summary

What is missing today is not another prompt enhancer. The missing pieces are the controls that turn image generation into asset production:

1. formal asset specifications
2. reference-driven generation
3. shape and silhouette control
4. transparent-background export workflow
5. post-generation validation
6. in-context preview against the target slot

These requirements drive the phases below.

---

# Cross-phase pipeline work

The phases below describe product capabilities, but several pipeline capabilities cut across all of them. These should be treated as first-class implementation tracks, not incidental follow-up work.

## 1. Reference conditioning modes

The pipeline should support more than prompt-only generation. At minimum it needs a defined contract for:
- txt2img
- img2img from a source asset
- masked img2img
- reference-driven structure preservation

This matters especially for modular or slot-constrained assets where text alone is not enough to preserve usable geometry.

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
- optional silhouette deviation metrics
- generation mode used
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

# Phase 1 — Asset slot specification

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
- transparency requirement
- shape policy:
  - freeform
  - bounded
  - silhouette-locked
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
- transparency toggle
- shape policy selector

## Files likely affected

- `src/projects/Project.hpp`
- `src/projects/ProjectManager.cpp`
- `src/controllers/ProjectController.*`
- `src/views/ProjectView.*`
- `docs/75_projects/project_overview.md`

## Acceptance criteria

- The user can define a replacement-target contract for each asset type.
- An asset type can express a slot equivalent to `door-left`, not just a prompt.
- The specification persists in `projects.json`.

---

# Phase 2 — Reference and silhouette inputs

## Goal

Make the generator start from the target asset shape instead of guessing it from text.

## Why this is required

Prompt-only generation is too loose for modular wall pieces, door inserts, tile-aligned props, and other structurally constrained assets. For those cases, structure must come from a reference, mask, or template. Freeform asset types should still be allowed to skip this path.

## Work

Add support for asset-reference inputs per asset type:
- reference image input
- optional rasterized source SVG input
- optional silhouette mask
- optional bounds mask

Generation modes should include:
- text-only
- img2img from reference asset
- masked img2img
- style transfer from reference shape

The most important initial path is:
1. rasterize existing SVG or source art
2. use it as structure reference
3. regenerate style/material/detail while preserving placement and silhouette

## UI work

Add a "Reference Shape" block in the asset detail panel:
- source asset picker
- mask preview
- reference enable/disable
- strength control specific to structure preservation

## Files likely affected

- `src/projects/Project.hpp`
- `src/projects/ProjectManager.cpp`
- `src/controllers/ImageGeneratorController.*`
- `src/controllers/ProjectController.*`
- `src/views/ProjectView.*`
- `src/portraits/PortraitGeneratorAi.*`
- pipeline docs under `docs/10_pipeline/`

## Acceptance criteria

- The user can attach a source asset as a structural reference.
- The generator can preserve left-wall vs right-wall composition instead of drifting into a new scene.
- Existing SVG assets can be used as a starting point for replacement generation.
- Asset types that do not need structural references can continue using a lighter prompt-first workflow.

---

# Phase 3 — Asset-type locked prompt constraints

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

- Users no longer need to repeatedly type orientation and isolation constraints.
- Asset templates are materially better at producing slot-compatible assets than a blank prompt.
- The generator is less likely to invent background scenes for modular pieces.

---

# Phase 4 — Transparent asset workflow

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

- The user can produce a transparent asset export from a project workflow.
- Edge halos are reduced enough for in-game use.
- Transparency is visible and reviewable before export.

---

# Phase 5 — Validation and compatibility checks

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
- occupied-bounds check
- silhouette overlap or deviation check
- anchor compatibility check
- orientation consistency check

Validation output should be lightweight but actionable:
- pass
- warning
- fail

## UI work

Show a small validation summary on selected results:
- canvas
- transparency
- shape fit
- anchor fit
- export readiness

## Files likely affected

- `src/controllers/ProjectController.*`
- `src/controllers/ImageGeneratorController.*`
- `src/ui/widgets/ResultPanel.*`
- new validation helpers module

## Acceptance criteria

- The tool can flag an asset that looks good but does not fit its slot.
- The user can tell whether a result is game-ready without manually exporting and testing it first.

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

# Immediate implementation order

If the goal is to support structurally constrained assets as soon as possible, the shortest useful sequence is:

1. Asset slot specification
2. Reference shape input
3. Masked / reference-driven img2img
4. Transparent export workflow
5. In-room preview with anchor visualization

That order gets the tool from "generate asset-like images" to "produce plausible replacement assets" with the least wasted work.

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
