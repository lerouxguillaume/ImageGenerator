# LLM Prompt Enhancement

Optional system using ORT GenAI.

---

# Architecture

- IPromptEnhancer interface
- OrtLlmEnhancer implementation
- NullPromptEnhancer fallback

---

# Mode

Stateless transform:
- input prompt → output prompt + negative prompt

---

# Rules

- Do NOT add an `enhance()` method to `IPromptEnhancer` — `transform()` is the only mode; empty `instruction` signals generic improvement and is handled internally