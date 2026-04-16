# Cancellation System

## What this file explains
How generation is cancelled mid-run: the watcher thread, `SetTerminate`, and why the VAE is deliberately exempt.

## When to use this
- Debugging cancelled generations that still complete VAE decode
- Understanding why `ctx.run_opts` must not be touched inside `runPipeline()`
- Diagnosing a "cancel did nothing" bug

## Key invariants
- `ctx.run_opts` is reset to a fresh `Ort::RunOptions{}` by `ModelManager::get()` at the start of **every** run — before `runPipeline()` is entered. Do not reset it manually inside `runPipeline()`.
- The VAE uses `Ort::RunOptions{nullptr}` (not `ctx.run_opts`) and is **never** affected by cancellation.
- The watcher thread is started fresh per generation and joined/detached at cancellation or completion.

## Mental model

```
User clicks Cancel
       │
       ▼
cancelToken.store(true)
       │
       ▼
watcher thread (polls every 20ms)
  detects cancelToken == true
       │
       ▼
ctx.run_opts.SetTerminate()
       │
       ▼
In-flight ORT Run() (UNet or text encoder) throws OrtException
       │
       ▼
runPipeline() catches, returns early
       │
       ▼
VAE decode is NOT called (loop already broken)
```

## Implementation details

### Watcher thread (`SdPipeline.cpp`)

```cpp
std::thread watcher([&]() {
    while (!cancelToken.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    ctx.run_opts.SetTerminate();
});
```

The thread is joined after the denoising loop exits (either naturally or via the thrown exception).

### `RunOptions` reset (`ModelManager.cpp`)

```cpp
// Inside ModelManager::get() — runs on every call, hit or miss
ctx.run_opts = Ort::RunOptions{};
```

This clears any `SetTerminate()` from a previous cancelled run. Without this reset, the first UNet call on the next generation would throw immediately.

### VAE exemption (`SdVae.cpp`)

```cpp
// decodeLatent uses nullptr RunOptions — not ctx.run_opts
vae_decoder.Run(Ort::RunOptions{nullptr}, ...);
```

The VAE is not cancellable by design. If cancellation reached the VAE stage, the denoising loop already ran to completion and the image is essentially done — there is no value in aborting the decode.

### Cancellation latency

Worst case: `20 ms` (one polling interval). The watcher sleeps 20 ms between polls. After `SetTerminate()` is called, the in-flight ORT `Run()` may take a few additional milliseconds to respond.

## Common pitfalls
- Passing `ctx.run_opts` to the VAE `Run()` call — cancellation would abort a near-complete decode and return a partial image instead of the blank that the calling code expects.
- Calling `cpu_unet` with `ctx.session_opts` — the CPU UNet must use `ctx.cpu_session_opts` (no GPU EP). This is separate from the RunOptions question.
- Not resetting `ctx.run_opts` between runs — a stale `SetTerminate()` from a previous cancel causes the next generation to throw on its first UNet call.
- Stopping the watcher thread before the denoising loop exits — the thread must be joined, not detached, so that `SetTerminate()` timing is deterministic.

## Related files
- [`docs/10_pipeline/pipeline_orchestration.md`](pipeline_orchestration.md) — loop structure
- [`docs/10_pipeline/gpu_fallback.md`](gpu_fallback.md) — cpu_unet session options
- [`docs/20_models/model_cache.md`](../20_models/model_cache.md) — `ModelManager::get()` reset logic
- `src/portraits/sd/SdPipeline.cpp` — watcher thread, denoising loop
- `src/portraits/sd/SdVae.cpp` — `decodeLatent`, `Ort::RunOptions{nullptr}`
