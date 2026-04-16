# GPU Fallback System

Handles failure of DirectML execution.

---

# Trigger condition

If UNet DML execution throws:
- `dmlFailed = true`

---

# Behavior after failure

All future runs:

- use `cpu_unet` session
- skip GPU execution entirely

---

# Persistence

Failure flag is stored in:
- `GenerationContext`

It persists across runs.

---

# Rules

- Do NOT call `cpu_unet` with `ctx.session_opts` — must use `ctx.cpu_session_opts` (no GPU EP registered)
- Do NOT assume `dmlFailed` resets between runs — it persists in the cached context; only a cache eviction resets it