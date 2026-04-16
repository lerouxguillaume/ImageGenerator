# Performance

## Session warmup

`SdLoader.cpp` calls `warmupSession()` for every ORT session immediately after creation: `text_encoder`, `unet`, `cpu_unet` (if present), `vae_decoder`, and `text_encoder_2` (SDXL only).

The helper:
1. Queries each input's shape from session metadata, replacing dynamic dims (`-1`) with `1`.
2. Allocates zero-filled tensors with the correct dtype (float16, float32, or int64).
3. Calls `session.Run()` once to trigger ORT's JIT kernel compilation.

Warmup is **non-fatal** — any exception is caught and logged as a warning. Without warmup the first real inference call pays the JIT cost; with it that cost is absorbed at load time before the user clicks Generate.

Do not remove warmup calls — the latency reduction is significant for the first image after a model or LoRA change.

## Timing instrumentation

All pipeline stages log timing via `fmtMs()` (`SdUtils.hpp`). Check `image_generator.log` for per-stage breakdown:

- Text encoding time
- Per-step UNet time (CFG = 2 passes per step)
- VAE decode time
- Total pipeline time

## LoRA cache hit vs miss

First run is expensive (matmuls + I/O); subsequent runs are cheap memcpy. Full details and memory profile in [`docs/30_lora/lora_caching.md`](../30_lora/lora_caching.md).

The log line `ModelManager: cache hit — reusing loaded sessions.` confirms a full session cache hit (no session rebuild at all).

## GPU fallback cost

If the DML UNet throws, `dmlFailed` is set to `true` and all subsequent steps use `cpu_unet`. CPU inference is significantly slower than GPU. The flag persists in the cached `GenerationContext` — once failed, always CPU for that cache entry. Check the log for `UNet GPU pass failed, switching to CPU fallback` to detect this.

## Cancellation overhead

A watcher `std::thread` polls `cancelToken` every 20 ms. When set it calls `ctx.run_opts.SetTerminate()` to abort any in-flight ORT `Run()`. The 20 ms polling interval is the worst-case latency between cancel request and abort signal.
