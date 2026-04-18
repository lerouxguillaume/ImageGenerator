# LLM Prompt Enhancement

Optional system using ORT GenAI.

---

# Architecture

- `IPromptEnhancer` interface (`src/llm/IPromptEnhancer.hpp`)
- `OrtLlmEnhancer` implementation
- `NullPromptEnhancer` fallback (no-op, always returns original)

---

# Mode

Stateless transform operating on raw strings:

```
LLMRequest { prompt, instruction, model, strength }
    → IPromptEnhancer::transform()
    → LLMResponse { prompt, negative_prompt }
```

---

# Integration with Prompt DSL (Phase 7 Step 2)

The controller does **not** replace the user's text with LLM output directly.
Instead it applies the result as a DSL merge patch:

```
original text  → parse → base DSL
LLM output     → parse → patch DSL
merge(base, patch) → compile(SDXL) → setText() back to UI
```

This preserves user edits and tokens not mentioned by the LLM.
The original text is captured into `LlmBar::originalPositive/originalNegative`
at enhancement launch so the merge base is always pre-LLM state.

---

# Rules

- Do NOT add an `enhance()` method to `IPromptEnhancer` — `transform()` is the only mode
- Empty `instruction` signals generic improvement and is handled internally
- LLM output is treated as a DSL patch, never as a full prompt replacement
