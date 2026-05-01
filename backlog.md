# Technical Debt Backlog

## Completed

- Create an artifact/run repository for output paths, metadata, manifests, and gallery discovery.
  Added `AssetArtifactStore` to own standard asset paths, candidate run layouts, metadata sidecar
  paths, transparent derivative paths, gallery directory selection, and gallery image
  enumeration. `ImageGeneratorController`, `CandidateRunPipeline`, and
  `ProjectController::refreshPatron` now use it instead of reconstructing path conventions
  locally.

- Extract a shared asset output path helper.
  Superseded by `AssetArtifactStore`, which now owns `config.outputDir / outputSubpath`,
  `raw/`, `processed/`, `.reference_cache/`, `runs/`, and `patron.png` path construction.

- Extract `processGeneratedAsset` from generation and candidate run flows.
  Added `GeneratedAssetProcessor` to centralise raw image loading, alpha cutout, standalone
  transparent derivative output, `AssetPostProcessor` execution, processed image writing, and
  metadata sidecar writing. `ImageGeneratorController` and `CandidateRunPipeline` now pass
  typed request metadata instead of each owning their own image-processing block.

- Move image scoring out of `CandidateRunPipeline`.
  Added `CandidateScorer` to own current candidate score calculation and the `CandidateScore`
  result type. `CandidateRunPipeline` and gallery ranking now call the scorer instead of keeping
  correctness scoring inside the run orchestration code.

- Decompose `refreshGallery` into focused functions.
  Split gallery refresh into file-local helpers for loading artifact entries, scoring candidate
  entries, sorting gallery entries, building `ResultPanel::GalleryItem` values, scheduling
  thumbnail loads, and restoring selection. Thumbnail scheduling behavior is unchanged.

- Decompose `selectGalleryImage` into path resolution, loading, metadata, and validation.
  Split selected-image handling into helpers for resolving processed/raw/transparent display
  paths, loading gallery image data, reading metadata sidecars, building validation chips, and
  clearing selected-image state. The controller now mainly applies loaded data to `ResultPanel`.

- Make `AlphaCutout` options configurable at runtime via `AssetExportSpec`.
  Added `AlphaCutoutSpec` to `AssetExportSpec` with tolerance, feather radius, and defringe
  settings. Project JSON persists the settings, and both `GeneratedAssetProcessor` and
  `CandidateScorer` convert them to `AlphaCutout::Options` at the processing boundary.

- Move `CandidateRunSettings` onto `AssetType`.
  Added persisted `CandidateRunSettings` to `AssetType`, `AssetTypeTemplate`, and
  `ResolvedProjectContext`. Candidate runs now use the active asset type's exploration/refinement
  counts, strengths, and score threshold instead of controller-local hardcoded defaults.

- Add orientation-specific shapes to `PatronGenerator`.
  `PatronGenerator` now resolves patron bounds through a shared helper and draws a mirrored
  parallelogram for `RightWall`, an isometric diamond for `FloorTile`, and a tall ellipse for
  `Character`. `Unset` and `Prop` retain the rectangular fallback.

## High Priority

- Split cached model state from per-run mutable state in the SD pipeline.
  Today `ModelManager` returns a shared `GenerationContext` that is mutated during each run
  (guidance, embeddings, time ids, latent shape, `run_opts`). This blocks safe concurrent
  generation and makes future background work fragile. Introduce an immutable `LoadedModel`
  for sessions/config/tokenizer-facing assets and a fresh `RunContext` per generation.

- Add bounded eviction to model and LoRA caches.
  `ModelManager` and the static LoRA injector cache both grow without limits as models and
  LoRA combinations accumulate. Add explicit ownership plus LRU or size-based eviction so the
  app degrades gracefully instead of drifting toward OOM on long sessions.

- Parameterise candidate scoring as `scoreCandidate(spec, ScoringPolicy)`.
  (`src/assets/CandidateScorer.cpp`) Scoring is now isolated from run orchestration, but the
  fill-ratio weights, bounds error caps, and anchor penalties are still wall-oriented magic
  constants. Introduce a `ScoringPolicy` struct and store defaults per orientation in
  `AssetSpec` or `AssetTypeTemplate`.

- Decouple `ProjectController` from `ImageGeneratorController`.
  (`ProjectController.hpp:39`, `ProjectController.cpp:330–353, 531–541`) `ProjectController`
  owns `ImageGeneratorController` as a direct member and manually routes events to it in three
  separate locations. This prevents testing generation in isolation and means new shared widgets
  require touching event routing in multiple places. Extract focused services
  (`GenerationService`, `GalleryService`, `PromptEnhancementService`) and replace direct
  ownership with an interface or callback so failures are easier to isolate and the UI layer
  stays thin as features grow.

- Introduce a typed `GenerationJob` / `GenerationService` boundary.
  `ImageGeneratorController` currently owns model browsing, LLM futures, thumbnail futures,
  generation threads, cancellation, progress, gallery refresh, and project context state. Add a
  service-level API that accepts a typed generation request and emits typed progress/result/error
  events. This should become the shared orchestration boundary for standalone generation, edit
  mode, and project candidate runs.

## Medium Priority

- Replace per-image thumbnail `std::async` spawning with a bounded thumbnail worker queue.
  (`ImageGeneratorController.cpp:1162–1177`) Gallery refresh currently launches one async task
  per image with no upper bound. Move to a fixed-size worker pool (4–8 workers), lazy loading
  for visible items, and an on-disk or in-memory thumbnail cache keyed by image path + mtime.

- Add project UI controls for `CandidateRunSettings`.
  Candidate run settings are now persisted on each `AssetType`, but the project asset detail
  panel still does not expose controls for exploration count, candidate count, refine variants,
  exploration strength, refinement strength, or score threshold. Add controls hidden for
  `Standard` workflow assets and save them through `ProjectManager::updateAssetType`.

- Extract project/config JSON codecs out of manager classes.
  `ProjectManager.cpp` owns enum string mapping, `AssetSpec` JSON, `AssetExportSpec` JSON,
  project JSON, ID generation, load/save, and mutations. Move serialization into dedicated
  codec files such as `ProjectJson.*` and `ConfigJson.*` so managers focus on storage semantics
  and project mutation rules.

- Add atomic save and backup behavior for user-owned JSON files.
  `ProjectManager::save()`, `PresetManager::save()`, and `AppConfig::save()` write directly to
  their target files. Write to a temporary file, flush, then rename into place, optionally
  retaining one `.bak`, so project/config/preset state is not corrupted if the app exits during
  a save.

- Add a headless test harness for domain logic.
  The project has no visible test target. Add a small CTest-enabled executable covering prompt
  parse/merge/compile behavior, project JSON codecs, output path resolution, post-processing
  invariants, and candidate scoring without opening SFML windows or loading ONNX models.

## Low Priority

- Centralise constraint token string literals.
  (`ProjectController.cpp` — `buildConstraintTokens`) `"transparent background"`,
  `"isometric view"`, `"isometric perspective"` etc. are inline string literals. A token rename
  requires a grep across the codebase. Move to a `ConstraintTokens` namespace with named
  string constants.

- Replace `BrowseTarget` enum with a string-keyed map.
  (`ImageGeneratorController.hpp:86–88`) Each new configurable path requires a new enum value,
  a new switch case in `update()`, and a new entry in `openSettings()`. A map of browse
  targets and their descriptions removes all three edits.

- Add `valid()` invariant check to `ResolvedProjectContext`.
  (`src/projects/Project.hpp:108–126`) The struct can be constructed with `projectId` set but
  `assetTypeId` empty. The only guard is `empty()`. Add explicit invariant checks callable from
  `buildSelectedContext`.

- Replace `GLOB_RECURSE` in CMake with explicit file lists (or document the tradeoff).
  (`CMakeLists.txt:40–43`) Any `.cpp` dropped in `src/` compiles automatically, including
  accidental debug files. Either switch to explicit lists or add a comment so future
  contributors know the current approach is intentional.

- Replace timestamp-only project, asset type, and preset IDs with an ID provider.
  `ProjectManager` and `PresetManager` generate IDs from millisecond timestamps. Introduce a
  small `IdGenerator` abstraction so IDs are collision-resistant and tests can inject
  deterministic IDs.

- Enrich candidate run manifests for reproducibility and debugging.
  `manifest.json` currently records paths, scores, and status. Add model id/path, LoRA set,
  seed, generation parameters, asset spec, export spec, scoring policy, app/schema version, and
  prompt hashes so a run can be reproduced and inspected after the fact.
