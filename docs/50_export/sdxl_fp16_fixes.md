# SDXL fp16 Fixes

## What this file explains
Why SDXL exports break and how to fix them.

## When to use this
- ORT dtype errors

## Key invariants
- Mixed fp32/fp16 breaks ORT

## Mental model
Fix graph inconsistencies.

## Implementation details
- fix_fp32_constants
- fix_resize_fp16_input

## Common pitfalls
- Resize requires fp32 scales

## Related files
- export_overview.md