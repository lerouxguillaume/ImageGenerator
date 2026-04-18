# Prompt Transform

Transforms a prompt using a local LLM and merges the result back into the DSL.

---

# Input (`LLMRequest`)

| Field | Type | Notes |
|---|---|---|
| `prompt` | `std::string` | Compiled positive prompt (SDXL form) |
| `instruction` | `std::string` | User instruction; empty = generic improve |
| `model` | `ModelType` | SD15 or SDXL — informs LLM style hints |
| `strength` | `float` | 0.0–1.0 enhancement intensity |

---

# Output (`LLMResponse`)

| Field | Type | Notes |
|---|---|---|
| `prompt` | `std::string` | Enhanced positive prompt |
| `negative_prompt` | `std::string` | Generated negative prompt |

---

# Behaviour

Uses structured JSON output enforced by prompt engineering.  
Pre-fills the assistant turn with `{` so the model generates valid JSON directly.  
`JsonStreamTracker` detects structural JSON completion and stops generation early.  
Falls back to original prompt + default negatives on parse failure.

---

# Merge flow (controller side)

The controller applies the LLM output as a DSL patch, not a replacement:

```cpp
// launchEnhancement — capture originals before launching thread
llm.originalPositive = sp.positiveArea.getText();
llm.originalNegative = sp.negativeArea.getText();

// update() — when enhanceDone fires
const Prompt base   = PromptParser::parse(llm.originalPositive, llm.originalNegative);
const Prompt patch  = PromptParser::parse(llm.enhancedPositive, llm.enhancedNegative);
const Prompt merged = PromptMerge::merge(base, patch);
sp.positiveArea.setText(PromptCompiler::compile(merged, ModelType::SDXL));
sp.negativeArea.setText(PromptCompiler::compileNegative(merged, ModelType::SDXL));
```

Merge rules: subject overrides if set; tokens union-deduplicated; patch weight wins on collision.
