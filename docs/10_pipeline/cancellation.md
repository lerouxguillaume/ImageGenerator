# Cancellation System

ImageGenerator supports async cancellation during inference.

---

# Mechanism

Each generation runs on a `std::jthread` owned by `ImageGeneratorController`.  
`std::stop_token` (C++20) is passed through the call stack to `runPipeline`.  
A `std::stop_callback` registered at the top of `runPipeline` calls `ctx.run_opts.SetTerminate()`
**immediately** when `request_stop()` is called — no polling thread, no latency.

---

# Cancellation flow

1. User clicks Cancel → `ResultPanel::cancelToken` flag set, `generating = false`
2. Controller sees the flag in `handleEvent` → calls `generationThread_.request_stop()`
3. `std::stop_callback` fires synchronously → `ctx.run_opts.SetTerminate()`
4. ONNX UNet execution aborts with `Ort::Exception`
5. `runPipeline` catches it, checks `stopToken.stop_requested()` → logs cancellation
6. Thread exits, sets `generationDone`
7. `update()` joins the thread (instant — already done)

---

# Thread ownership and serialisation

`ImageGeneratorController` holds a single `std::jthread generationThread_`.  
Assigning a new `jthread` to it **automatically** calls `request_stop()` + `join()` on the
previous one before the new thread starts. This guarantees:

- Only one pipeline runs at a time
- No thread is ever abandoned (old detach() bug eliminated)
- No double model-load from concurrent threads

---

# Important exception

VAE is NOT affected by cancellation:
- uses `Ort::RunOptions(nullptr)`
- cannot be terminated mid-run

---

# Reset behavior

Each run:
- `ModelManager::get()` resets `run_opts` before every call

Prevents stale termination flags from a previous cancelled run.

---

# Rules

- Do NOT pass `ctx.run_opts` to VAE `Run()` — VAE must use `Ort::RunOptions{nullptr}` to stay uncancellable
- Do NOT reset `ctx.run_opts` manually inside `runPipeline()` — `ModelManager::get()` already does this before every run
- Do NOT use `std::atomic<bool>*` for cancellation — use the `std::stop_token` passed to `runPipeline`