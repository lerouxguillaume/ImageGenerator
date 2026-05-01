# Technical Debt Backlog

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

- Parameterise `scoreWallCandidate` as `scoreCandidate(spec, ScoringPolicy)`.
  (`ImageGeneratorController.cpp:282–356`) Scoring is hardcoded for walls: fill ratio weights,
  bounds error caps, and anchor penalties are all magic float constants. Adding a Character or
  Prop asset type means duplicating the entire function. Introduce a `ScoringPolicy` struct and
  store defaults per orientation in `AssetSpec` or `AssetTypeTemplate`.

- Decouple `ProjectController` from `ImageGeneratorController`.
  (`ProjectController.hpp:39`, `ProjectController.cpp:330–353, 531–541`) `ProjectController`
  owns `ImageGeneratorController` as a direct member and manually routes events to it in three
  separate locations. This prevents testing generation in isolation and means new shared widgets
  require touching event routing in multiple places. Extract focused services
  (`GenerationService`, `GalleryService`, `PromptEnhancementService`) and replace direct
  ownership with an interface or callback so failures are easier to isolate and the UI layer
  stays thin as features grow.

## Medium Priority

- Replace per-image thumbnail `std::async` spawning with a bounded thumbnail worker queue.
  (`ImageGeneratorController.cpp:1162–1177`) Gallery refresh currently launches one async task
  per image with no upper bound. Move to a fixed-size worker pool (4–8 workers), lazy loading
  for visible items, and an on-disk or in-memory thumbnail cache keyed by image path + mtime.

- Decompose `refreshGallery` into focused functions.
  (`ImageGeneratorController.cpp:1070–1202`) Directory selection, file enumeration,
  scoring/sorting, async thumbnail scheduling, and selection-state restoration are all inlined
  in 130 lines. Extract: `selectGalleryDir`, `scanGalleryDir`, `sortGalleryEntries`,
  `scheduleThumbLoading`.

- Extract `processGeneratedAsset` from both generation thread lambdas.
  (`ImageGeneratorController.cpp:644–728` and `839–878`) Both lambdas contain a nested
  `processOutput` closure — alpha cutout → post-processor → metadata JSON write — with slight
  variations between them. A single standalone function eliminates both.

- Move `CandidateRunSettings` onto `AssetType` so they are configurable per asset type.
  (`ImageGeneratorController.hpp:102–110`) `minExploreImages`, `candidateCount`,
  `refineVariants`, `explorationStrength`, and `refinementStrength` are hardcoded controller
  defaults. Projects that need different exploration budgets cannot change them without
  recompiling. Persist in `projects.json` and add UI controls in the asset detail panel.

- Add orientation-specific shapes to `PatronGenerator`.
  (`src/projects/PatronGenerator.cpp:21–46`) Only `LeftWall` has a real parallelogram shape;
  all other orientations fall back to a plain rectangle. Add geometry for `RightWall`
  (mirrored parallelogram), `FloorTile` (isometric diamond), and `Character` (tall centered
  rectangle or ellipse).

- Extract a shared `assetOutputDir(config, context)` path helper.
  `config.outputDir / outputSubpath` is assembled independently in `launchGeneration`,
  `launchCandidateRun`, and `ProjectController::refreshPatron`. One helper removes all three.

- Make `AlphaCutout` options configurable at runtime via `AssetExportSpec`.
  (`ImageGeneratorController.cpp:680`, `src/postprocess/AlphaCutout.hpp`) Alpha cutout is
  always called with default Options. Add an `alphaCutoutOptions` field to `AssetExportSpec`
  and pass it through so feathering and threshold can be tuned per asset type.

- Decompose `selectGalleryImage` into path resolution, loading, metadata, and validation.
  (`ImageGeneratorController.cpp:1014–1068`) Interleaved path resolution, SFML image loading,
  metadata JSON parsing, and validation checks make it hard to add a new metadata field or
  change texture format. Extract `loadGalleryImage(path, spec) → GalleryImageData`.

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
