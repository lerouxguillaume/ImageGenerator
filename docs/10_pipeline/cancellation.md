# Cancellation System

ImageGenerator supports async cancellation during inference.

---

# Mechanism

- `cancelToken` shared state
- watcher thread polls every 20ms

---

# Cancellation flow

1. cancelToken set
2. watcher thread triggers
3. `ctx.run_opts.SetTerminate()` called
4. ONNX UNet execution aborts

---

# Important exception

VAE is NOT affected:
- uses `Ort::RunOptions(nullptr)`
- cannot be terminated mid-run

---

# Reset behavior

Each run:
- `ModelManager::get()` resets `run_opts`

Prevents stale termination flags.

---

# Rules

- Do NOT pass `ctx.run_opts` to VAE `Run()` — VAE must use `Ort::RunOptions{nullptr}` to stay uncancellable
- Do NOT reset `ctx.run_opts` manually inside `runPipeline()` — `ModelManager::get()` already does this before every run; double-reset is a no-op but signals misunderstanding of ownership