# Prompt DSL System

Model-agnostic structured prompt representation with model-specific compilation.

---

# Principles

1. DSL is model-agnostic — no model logic inside the structs
2. Compiler is a pure renderer — ordering, formatting, joining only
3. Semantic content (quality boosters) lives in config, not compiler logic
4. Token.value contains raw semantic text only — never embedded A1111 syntax
5. Subject is a normal Token placed first — no special wrapping or weighting
6. Deterministic: same DSL → same compiled string

---

# Data structures (`src/prompt/Prompt.hpp`)

```cpp
struct Token {
    std::string value;      // raw semantic text only — no "(text:1.2)" embedded
    float       weight = 1.0f;
};

struct Prompt {
    std::optional<Token> subject;  // first comma-token (ordering hint only)
    std::vector<Token>   positive; // remaining positive tokens
    std::vector<Token>   negative; // negative tokens
};
```

`subject` is an `optional<Token>` — it carries a clean value and weight, exactly
like any other positive token. It is placed first in the compiled output.

---

# Parser (`src/prompt/PromptParser.hpp/cpp`)

```cpp
Prompt PromptParser::parse(const std::string& positiveRaw,
                            const std::string& negativeRaw = {});
```

## Rules

- Split by `,`, trim whitespace, skip empty segments
- First positive segment → `subject` via A1111 weight parsing
- Remaining segments → `positive` tokens via A1111 weight parsing
- Negative segments → `negative` tokens via A1111 weight parsing

All segments go through the same `parseToken()` path — no special-casing.

## A1111 weight parsing

| Input | Token result |
|---|---|
| `(text:1.3)` | `{text, 1.3f}` |
| `(text)` | `{text, 1.1f}` (A1111 bare-parens convention) |
| `text` | `{text, 1.0f}` |

A1111 syntax is **always stripped** from `value` — weight is extracted into the `weight` field.
No raw `(text:1.2)` strings are stored inside any Token.

## Roundtrip invariant

```
compile(parse(x)) ≈ x
```

Tokens with `weight == 1.0` round-trip exactly; weighted tokens reformat as `(value:weight)`.

---

# Compiler (`src/prompt/PromptCompiler.hpp/cpp`)

```cpp
std::string PromptCompiler::compile(const Prompt& p, ModelType model);
std::string PromptCompiler::compileNegative(const Prompt& p);
```

## Responsibilities (strictly limited)

1. Order tokens: subject → positive
2. Format each token: `value` if weight≈1.0, else `(value:%.2f)`
3. Join with `, `

The compiler does **not** add, remove, or modify semantic content.

## Positive output order

```
1. subject (if set)
2. positive tokens
```

## Model differences

SDXL and SD1.5 are compiled identically. Differences are limited to formatting style;
neither model adds tokens or quality boosters inside the compiler.

If model-specific tokens are needed, add them to `ModelDefaults.qualityBoosters` in
`config.json` — they are injected as normal DSL tokens at generation time, not
by the compiler.

## Token formatting

```
weight ≈ 1.0  →  value
weight ≠ 1.0  →  (value:1.30)
```

## Logging

Every `compile()` call emits:

```
[PromptCompiler] model=SD15 output="girl, cinematic lighting, (85mm:1.30)"
```

---

# Quality boosters (`AppConfig` / `ModelDefaults`)

Quality boosters are **not** compiler logic. They live in `config.json`:

```json
"modelConfigs": {
  "anything_v5": {
    "qualityBoosters": ["masterpiece", "best quality"]
  }
}
```

At generation time, `ImageGeneratorController::launchGeneration` reads
`ModelDefaults.qualityBoosters` and appends any missing values as normal
`Token{booster, 1.0f}` entries into the DSL before compiling. Duplicates
(already present in user's positive tokens) are skipped.

This keeps boosters as config data and the compiler as a pure renderer.

---

# Merge (`src/prompt/PromptMerge.hpp/cpp`)

```cpp
Prompt PromptMerge::merge(const Prompt& base, const Prompt& patch);
```

Used by LLM enhancement to non-destructively apply an LLM-generated patch
without clobbering user edits.

## Rules

| Field | Behaviour |
|---|---|
| `subject` | patch overrides if set |
| `positive` | deduplicated by value; patch weight overrides on collision |
| `negative` | same as positive |

---

# JSON serialisation (`src/prompt/PromptJson.hpp`)

Header-only ADL hooks for `nlohmann_json`. Used by the preset system.

`subject` is serialised as a Token object:

```json
{
  "subject": {"value": "girl", "weight": 1.0},
  "positive": [
    {"value": "soft lighting", "weight": 1.0},
    {"value": "85mm",          "weight": 1.3}
  ],
  "negative": [
    {"value": "blurry", "weight": 1.0}
  ]
}
```

**Legacy compatibility:** old `presets.json` files where `subject` is a plain string
are loaded automatically — the string is converted to `Token{value, 1.0f}`.

---

# Integration points

## Generation (`ImageGeneratorController::launchGeneration`)

```
positiveArea.getText() + negativeArea.getText()
    → PromptParser::parse()
    → Prompt
    → injectBoosters(dsl, ModelDefaults)   ← dedup against subject + positive
    → PromptCompiler::compile(p, modelType)
    → std::string  →  PortraitGeneratorAi::generateFromPrompt()
```

`injectBoosters` is a file-local helper in `ImageGeneratorController.cpp`, shared by
`launchGeneration` and the compiled preview path in `update()`.

## LLM enhancement (`ImageGeneratorController::update`)

```
original text → parse → base DSL
LLM output   → parse → patch DSL
merge(base, patch) → compile(SDXL) → setText() back to UI
```

## Preset save (`buildGenerationSettings`)

```
positiveArea.getText() + negativeArea.getText()
    → PromptParser::parse()
    → GenerationSettings::dsl  →  PresetManager::createFromGeneration()
```

## Preset load (`applyPresetToSettings`)

```
Preset::dsl → PromptCompiler::compile(dsl, SDXL)  →  positiveArea.setText()
             → PromptCompiler::compileNegative(dsl, SDXL) → negativeArea.setText()
```

SDXL (neutral) form is used for display; generation re-compiles with actual model type.

---

# UI display (`SettingsPanel`)

Two DSL-derived elements are updated every frame by the controller:

## Token chips

Read-only chip row between positive area and negative label.

- Subject chip: gold border + gold text; weight suffix shown when non-default
- Positive tokens: neutral border; blue border if `weight > 1`, muted if `weight < 1`
- Weight shown as `label 2.0×` when `weight ≠ 1.0`
- Wraps to a second row (max 2 rows)

## Compiled preview

Single muted line below the negative area, **only visible when SD1.5 is selected**.
Shows the full compiled positive string including quality boosters from `ModelDefaults.qualityBoosters`,
so the user sees exactly what will be sent to the model.
Hidden for SDXL since output matches input.

---

# File map

```
src/prompt/
├── Prompt.hpp         — Token, Prompt structs
├── PromptParser.hpp   — parse() declaration
├── PromptParser.cpp   — parse() + A1111 weight parsing (all tokens including subject)
├── PromptCompiler.hpp — compile(), compileNegative() declarations
├── PromptCompiler.cpp — formatting/ordering only; no semantic injection
├── PromptMerge.hpp    — merge() declaration
├── PromptMerge.cpp    — merge() implementation
└── PromptJson.hpp     — nlohmann_json ADL hooks (header-only)
```
