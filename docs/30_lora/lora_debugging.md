# LoRA Debugging

## What this file explains
How to diagnose LoRA issues.

## When to use this
- No effect
- NaNs

## Key invariants
- Needs weights file
- Needs matching keys

## Mental model
Failure = match / dtype / data issue

## Implementation details
Check logs:
- external initializers count
- patch count

## Common pitfalls
- Missing *_weights.safetensors
- fp16 conversion errors

## Related files
- [lora_key_format.md](lora_key_format.md) — Kohya key format, prefix stripping, suffix matching
- [lora_key_format.md](lora_key_format.md) — Kohya key format and prefix stripping
- [lora_caching.md](lora_caching.md) — stale session cache diagnosis
- [../50_export/export_overview.md](../50_export/export_overview.md) — companion weights export requirements