# LLM Overview

## What this file explains
Prompt enhancement system.

## When to use this
- Prompt refinement

## Key invariants
- Stateless transform()

## Mental model
LLM rewrites prompt + negative

## Implementation details
- IPromptEnhancer
- OrtLlmEnhancer

## Common pitfalls
- JSON parsing failure

## Related files
- prompt_transform.md