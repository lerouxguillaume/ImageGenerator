# Prompt Transform

## Prompt structure

`buildTransformPrompt()` (`OrtLlmEnhancer.cpp`) assembles a Llama 3 single-turn chat prompt. The assistant turn is pre-filled with `{` so the model continues directly inside a JSON object — it cannot emit preamble even if it tries.

The system message is built from three injected blocks:

### `modelTypeGuidance(model)`

| Model | Guidance |
|---|---|
| SD 1.5 | Concise comma-separated tags. Order: subject → attributes → environment → composition → lighting → style → quality. Strong visual keywords, no long sentences. |
| SDXL | Natural-language descriptive sentences, same component order. Coherent flow; no keyword spam. |

### `strengthGuidance(strength)`

| Range | Tier | Instruction |
|---|---|---|
| ≤ 0.3 | minimal | Fix grammar and add 1–2 quality tags only |
| ≤ 0.7 | moderate | Improve detail, lighting, composition; keep subject unchanged |
| > 0.7 | strong | Substantially enhance style, atmosphere, technical quality; preserve core subject |

### Prompt structure block

The system message explicitly enumerates the 7 components to cover in order when relevant:

1. **SUBJECT** — who or what is depicted
2. **ATTRIBUTES** — appearance, clothing, pose, expression
3. **ENVIRONMENT** — setting, background, time of day
4. **COMPOSITION** — framing, camera angle, shot type
5. **LIGHTING** — cinematic, soft, dramatic, golden hour, etc.
6. **STYLE** — realism, anime, oil painting, photographic, etc.
7. **QUALITY TAGS** — masterpiece, best quality, highly detailed, sharp focus

## Negative prompt rules

The system message instructs the model to:
- Always include base negatives: `worst quality, low quality, blurry, bad anatomy`.
- Add context-aware negatives (e.g. `extra limbs` for characters, `blurry background` for landscapes, `watermark, text`).

## Instruction fallback

```
effectiveInstruction = instruction.empty()
    ? "Improve the prompt quality and detail"
    : instruction
```

The controller passes `req.instruction` from `view.instructionArea`. If empty, it falls back to `AppConfig::llmHint` for the current model, then to the generic default above.

## Example output

**Input:** `"portrait of a woman"`

**SD 1.5 output:**
```
portrait of a woman, detailed face, soft studio lighting, clean background,
sharp focus, masterpiece, best quality, highly detailed
```

**SDXL output:**
```
A detailed portrait of a woman with soft studio lighting, sharp focus on
facial features, a clean neutral background, rendered in photorealistic high quality.
```

## Anti-patterns the model is instructed to avoid

- Repeating the same words
- Adding elements unrelated to the subject or instruction
- Removing key elements from the original prompt
- Turning the prompt into an explanation of intent
