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

- Add a headless test harness for domain logic.
  The project currently has no test target (`ctest --test-dir build` reports no tests). Add a
  small CTest-enabled executable covering prompt parse/merge/compile behavior, the prompt DSL,
  registry capability loading, and output-path/gallery resolution — without opening SFML windows
  or loading ONNX models — so cleanup can proceed with regression coverage.

## Medium Priority

- Import script: detect actual LoRA compatibility.
  `import_model.py` still writes `lora_compatible: true` in the capabilities block. Define the
  compatibility rule in terms of exported companion weight files and/or key compatibility, then
  write `lora_compatible` from that check instead of defaulting to true.

- Replace per-image thumbnail `std::async` spawning with a bounded thumbnail worker queue.
  Gallery refresh (`scheduleThumbnailLoad` / `flushPendingThumbs` in `ImageGeneratorController`)
  launches one async task per image with no upper bound. Move to a fixed-size worker pool
  (4–8 workers), lazy loading for visible items, and an on-disk or in-memory thumbnail cache
  keyed by image path + mtime.

- VRAM-aware SDXL hires: warn/limit instead of OOM-mid-run.
  SDXL hires at 1.5× (1536px) OOMs the VAE decode on a ~12 GB card — in-app the UNet session's
  CUDA arena stays resident, leaving no room for the mid-block self-attention buffer (O(N²),
  ~2.7 GB at 1536px). Today the user gets a clear "out of GPU memory" error only *after* sitting
  through the (slow, high-res) hires denoise. Improve the UX: surface the VRAM limitation in the
  SettingsPanel hires controls (e.g. a note that high scales need lots of VRAM, or a soft cap
  derived from detected VRAM), so the failure is explained up front rather than discovered
  mid-run. The heavier structural fix — tiled VAE decode (`VaeDecodeStrategy` stub in `SdVae.cpp`)
  so large decodes never allocate one giant attention buffer — would remove the ceiling entirely;
  pair it with the arena-shrinkage option (release the UNet arena before decode) if pursued.
  Related: the "split cached model state from per-run mutable state" high-priority item.

## Low Priority

- Clean local/project hygiene entries.
  The repo has an untracked `.codex` file, an empty `src/services` directory, and an unignored
  empty `cache_dir`. Decide whether `cache_dir` and `src/services` are intentional, then remove
  or ignore/document them.

- Replace `BrowseTarget` enum with a string-keyed map.
  (`ImageGeneratorController.hpp`) Each new configurable path requires a new enum value, a new
  switch case in `update()`, and a new entry in `openSettings()`. A map of browse targets and
  their descriptions removes all three edits.

- Sort the model dropdown by most recently imported.
  `ImportedModel::importedAt` is populated on import and round-tripped through `registry.json`
  but never used. Sort the model list by descending `importedAt` when populating
  `SettingsPanel::models` so the most recently imported model is pre-selected. Optionally show
  the date in the dropdown.

- Replace `GLOB_RECURSE` in CMake with explicit file lists (or document the tradeoff).
  (`CMakeLists.txt`) Any `.cpp` dropped in `src/` compiles automatically, including accidental
  debug files. Either switch to explicit lists or add a comment so future contributors know the
  current approach is intentional.

- Replace timestamp-only preset IDs with an ID provider.
  `PresetManager` generates IDs from millisecond timestamps (`generateId()`). Introduce a small
  `IdGenerator` abstraction so IDs are collision-resistant and tests can inject deterministic IDs.
