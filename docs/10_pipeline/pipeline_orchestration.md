# Pipeline Orchestration

## What this file explains
How the full SD pipeline executes.

## When to use this
- Understanding generation flow
- Debugging pipeline issues

## Key invariants
- Order: tokenize → encode → UNet loop → VAE
- CFG uses 2 passes
- Scheduler controls sigma progression

## Mental model
Pipeline is a deterministic loop over noise refinement.

## Implementation details
- Entry: SdPipeline::runPipeline
- Calls:
    - encodeText
    - runUNetCFG
    - decodeLatent

## Common pitfalls
- Wrong dtype → silent corruption
- Skipping CFG → bad images

## Related files
- [scheduler.md](scheduler.md)
- [cancellation.md](cancellation.md)
- [dtype_handling.md](dtype_handling.md)
- [sdxl_specifics.md](sdxl_specifics.md)