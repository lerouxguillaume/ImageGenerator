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

## Medium Priority

- Replace per-image thumbnail `std::async` spawning with a bounded thumbnail worker queue.
  Gallery refresh currently launches one async task per image and rebuilds the full thumbnail
  workload on every refresh. Move to a fixed-size worker pool, lazy loading for visible items,
  and an on-disk or in-memory thumbnail cache keyed by image path + mtime.

- Split controller responsibilities into focused services.
  `ImageGeneratorController` currently owns generation orchestration, gallery management,
  thumbnail futures, prompt enhancement, settings side effects, and preset wiring. Extract
  `GenerationService`, `GalleryService`, and `PromptEnhancementService` so failures are easier
  to isolate and the UI layer stays thin as features grow.
