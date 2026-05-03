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

- Introduce the first typed `GenerationJob` / `GenerationService` boundary.
  Added `GenerationService` with typed generation, candidate-run, reference, post-process,
  progress, and result structs.
  The plain single-image/multi-image generation path now builds a `GenerationJob` in
  `ImageGeneratorController` and delegates execution, reference normalization, model invocation,
  and generated asset post-processing to the service. Candidate-run execution now builds a
  `CandidateRunJob` and delegates exploration, candidate selection, refinement, and manifest
  writing through the same service boundary. The controller now uses one shared generation-task
  launcher for completion and error propagation. Broader UI orchestration still needs to move out
  of the controller.

- Add typed `GenerationStage` progress reporting.
  Added `GenerationStage` enum to `src/enum/enums.hpp` covering all pipeline phases for both
  standard generation (`LoadingModel → EncodingText → EncodingImage → Denoising →
  DecodingImage → PostProcessing → Done`) and candidate runs (`Exploring → Scoring →
  Refining → WritingManifest → Done`). `GenerationProgress` gains a nullable
  `atomic<GenerationStage>*` stage pointer alongside the existing step and image counters.
  `SdPipeline::runPipeline()` writes stage transitions; `GenerationService::run()` owns
  `PostProcessing` and `Done`. `CandidateRunPipeline` owns coarser outer transitions and does
  not forward stage into inner pipeline calls. `ResultPanel` reads `generationStage` each frame
  and renders a stage-specific label instead of a bare step counter.

- Fix initial model registry/default ordering in `ImageGeneratorController::update()`.
  Moved registry mtime detection and model-list population before the model-default/model-type
  refresh block. Registry rebuilds now reset `viewInitialized` and `lastModelIdx` so startup
  and import-triggered model list changes apply defaults and infer type against the real
  selected model instead of the fallback `"models"` path.

- `loadCapabilities()` should cross-check file existence, not only the flag.
  Done: after reading the manifest flag, `ImportedModelRegistry::loadCapabilities()` now also
  requires `onnxPath / "vae_encoder.onnx"` to exist. The registry-facing UI capability state now
  matches `SdLoader`'s inference-time VAE encoder check, so img2img/reference controls are not
  shown for models that cannot actually encode an init/reference image.

- Replace `inferModelType()` with `arch` from the registry.
  Done: `SettingsPanel` now carries `availableModelTypes` alongside the other model vectors,
  populated from `ImportedModel::arch` when the registry is loaded. `ImageGeneratorController`
  refreshes `cachedModelType_` from the selected model's typed registry value instead of
  reopening and scanning `model.json`, and the old `inferModelType()` helper is gone.

- Show model type badge in the model selector dropdown.
  Done: `SettingsPanel::render()` now shows a compact right-aligned `SD1.5` or `SDXL` label
  for each model dropdown row using `availableModelTypes`, with no extra disk reads and no
  change to the collapsed selector layout.

- Prune missing models from the registry at load time.
  Done: `ImportedModelRegistry::load()` now skips entries with an empty or missing `onnxPath`
  and logs the skipped id/name/path. The pruning is in-memory only for now, so the registry file
  is not rewritten implicitly during load.

- Import script: write accurate `vae_encoder_available` based on export output.
  Done: `scripts/import_model.py` now writes `capabilities.vae_encoder_available` from the
  actual presence of `vae_encoder.onnx` in the export output directory instead of hardcoding
  `true`. LoRA compatibility still needs a separate concrete compatibility rule before changing
  `lora_compatible`.

- Add atomic save and backup behavior for user-owned JSON files.
  Done: added `JsonFileIO::atomicWrite()` and routed `ProjectManager::save()`,
  `PresetManager::save()`, `AppConfig::save()`, and `ImportedModelRegistry::save()` through it.
  Saves now write a same-directory `.tmp`, retain one `.bak` when replacing an existing file,
  and rename the temp file into place.

- Wire up a practical static-analysis / formatting baseline.
  Done: CMake now exports `compile_commands.json`, the repo has a `.clang-format`, and
  `docs/00_overview/build_system.md` documents targeted `clang-tidy` and formatting commands.
  clang-tidy is intentionally not enabled globally in every build yet because the profile is
  strict and the full codebase is not at a zero-warning baseline.

- Clear current compiler warnings from a clean rebuild.
  Done: handled the unchecked `chdir` result in `Subprocess.cpp`, scoped the GenAI timing helper
  to `USE_GENAI`, removed the unused `race` parameter name in `PromptBuilder.hpp`, stopped
  parsing the unused ONNX external-data offset, and changed the manifest key loop in
  `SdLoader.cpp` to avoid binding `std::string&` to string literal temporaries.

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

- ~~Extend the typed `GenerationJob` / `GenerationService` boundary across generation workflows.~~
  ~~Done: `GenerationCallbacks` / `CandidateRunCallbacks` added; both service methods return `void`~~
  ~~and catch exceptions internally, routing them through `onError`. `startGenerationTask` is now~~
  ~~a minimal thread spawner with no exception handling or business logic.~~

- Add a headless test harness for domain logic.
  The project currently has no visible test target (`ctest --test-dir build` reports no tests).
  Add a small CTest-enabled executable covering prompt parse/merge/compile behavior, project
  JSON codecs, output path resolution, post-processing invariants, registry capability loading,
  and candidate scoring without opening SFML windows or loading ONNX models. This should happen
  before more feature work so cleanup can proceed with regression coverage.

## Medium Priority

- Import script: detect actual LoRA compatibility.
  `import_model.py` still writes `lora_compatible: true` in the capabilities block. Define the
  compatibility rule in terms of exported companion weight files and/or key compatibility, then
  write `lora_compatible` from that check instead of defaulting to true.

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

## Low Priority

- Clean local/project hygiene entries.
  The repo has ignored build/package/dependency artifacts as expected, plus an untracked
  `.codex` file, an empty `src/services` directory, and an unignored empty `cache_dir`. Decide
  whether `cache_dir` and `src/services` are intentional, then remove or ignore/document them.

- Centralise constraint token string literals.
  (`ProjectController.cpp` — `buildConstraintTokens`) `"transparent background"`,
  `"isometric view"`, `"isometric perspective"` etc. are inline string literals. A token rename
  requires a grep across the codebase. Move to a `ConstraintTokens` namespace with named
  string constants.

- Replace `BrowseTarget` enum with a string-keyed map.
  (`ImageGeneratorController.hpp:86–88`) Each new configurable path requires a new enum value,
  a new switch case in `update()`, and a new entry in `openSettings()`. A map of browse
  targets and their descriptions removes all three edits.

- Sort the model dropdown by most recently imported.
  `ImportedModel::importedAt` is populated on import, round-tripped through `registry.json`,
  and loaded back into the registry — but never used. Sort `availableModels` (and all parallel
  vectors) by descending `importedAt` when populating the `SettingsPanel` model list so the
  most recently imported model is pre-selected. Optionally show the date in the dropdown
  tooltip.

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
