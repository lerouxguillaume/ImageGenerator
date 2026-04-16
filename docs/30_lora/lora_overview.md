# LoRA Overview

## What this file explains
How LoRA is integrated.

## When to use this
- Understanding LoRA system

## Key invariants
- Applied at load time
- Uses external initializers

## Mental model
LoRA = weight delta merged before inference.

## Implementation details
- LoraInjector
- computeLoraDelta

## Common pitfalls
- Expecting runtime application

## Related files
- [lora_key_format.md](lora_key_format.md) — Kohya key anatomy, prefix stripping, delta formula
- [lora_key_format.md](lora_key_format.md) — suffix matching pipeline (merged from lora_matching)
- [lora_caching.md](lora_caching.md) — session + injector cache layers
- [lora_debugging.md](lora_debugging.md) — diagnosing mismatches and NaN