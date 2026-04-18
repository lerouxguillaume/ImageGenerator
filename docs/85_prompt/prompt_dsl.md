# Prompt DSL System

Model-agnostic structured prompt representation with model-specific compilation.

---

# Principles

1. DSL is model-agnostic — no model logic inside the structs
2. Compilation is model-specific — compiler adapts output per `ModelType`
3. Deterministic: same DSL → same compiled string
4. Prompt string is output only — never internal state
5. Structure is optional; free tokens always work

---

# Data structures (`src/prompt/Prompt.hpp`)

```cpp
struct Token {
    std::string value;
    float       weight = 1.0f; // DSL-level weight (distinct from A1111 syntax in value)
};

struct Prompt {
    std::optional<std::string> subject;  // first comma-token (heuristic)
    std::vector<std::string>   styles;   // high-level style grouping (set via UI future)
    std::vector<Token>         positive; // remaining positive tokens
    std::vector<Token>         negative; // negative tokens
};
```

---

# Parser (`src/prompt/PromptParser.hpp/cpp`)

```cpp
Prompt PromptParser::parse(const std::string& positiveRaw,
                            const std::string& negativeRaw = {});
```

## Rules

- Split by `,`, trim whitespace, skip empty segments
- First positive segment → `subject` (verbatim, no weight stripping)
- Remaining segments → `positive` tokens via A1111 weight parsing
- Negative segments → `negative` tokens via A1111 weight parsing

## A1111 weight parsing (Phase 9)

| Input | Token result |
|---|---|
| `(text:1.3)` | `{text, 1.3f}` |
| `(text)` | `{text, 1.1f}` (A1111 bare-parens convention) |
| `text` | `{text, 1.0f}` |

Subject is kept verbatim so it may contain A1111 syntax — the compiler detects this
and avoids double-wrapping.

## Roundtrip invariant

```
compile(parse(x), SDXL) ≈ x
```

Approximate because SDXL compilation is neutral (no additions).

---

# Compiler (`src/prompt/PromptCompiler.hpp/cpp`)

```cpp
std::string PromptCompiler::compile(const Prompt& p, ModelType model);
std::string PromptCompiler::compileNegative(const Prompt& p, ModelType model);
```

## Positive output order

```
1. subject
2. styles
3. positive tokens
4. quality boosters (SD1.5 only)
```

## Model differences

### SDXL

- Natural phrasing, no boosters
- Subject passed through verbatim
- Weighted tokens: `(value:weight)` format

### SD1.5

- Subject wrapped as `(subject:1.20)` unless it already contains A1111 weight syntax
- Quality boosters appended: `masterpiece, best quality`
- Weighted tokens: `(value:weight)` format

## Logging

Every `compile()` call emits:

```
[PromptCompiler] model=SD15 output="(girl:1.20), cinematic, masterpiece, best quality"
```

---

# Merge (`src/prompt/PromptMerge.hpp/cpp`)

```cpp
Prompt PromptMerge::merge(const Prompt& base, const Prompt& patch);
```

Used by LLM enhancement (Phase 7 Step 2) to non-destructively apply an LLM-generated
patch without clobbering user edits.

## Rules

| Field | Behaviour |
|---|---|
| `subject` | patch overrides if set |
| `styles` | union, deduplicated, base order preserved |
| `positive` | deduplicated by value; patch weight overrides on collision |
| `negative` | same as positive |

---

# JSON serialisation (`src/prompt/PromptJson.hpp`)

Header-only ADL hooks for `nlohmann_json`. Used by the preset system.

```json
{
  "subject": "girl",
  "styles": ["cinematic"],
  "positive": [
    {"value": "soft lighting", "weight": 1.0},
    {"value": "85mm",          "weight": 1.3}
  ],
  "negative": [
    {"value": "blurry", "weight": 1.0}
  ]
}
```

---

# Integration points

## Generation (`ImageGeneratorController::launchGeneration`)

```
positiveArea.getText() + negativeArea.getText()
    → PromptParser::parse()
    → Prompt
    → PromptCompiler::compile(p, inferModelType(modelDir))
    → std::string  →  PortraitGeneratorAi::generateFromPrompt()
```

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

## Token chips (Phase 8)

Read-only chip row between positive area and negative label.

- Subject chip: gold border + gold text
- Positive tokens: neutral border; blue border if `weight > 1`, muted if `weight < 1`
- Weight shown as `label 2.0×` when `weight ≠ 1.0`
- Wraps to a second row (max 2 rows)

## Compiled preview (Phase 6)

Single muted line below the negative area, **only visible when SD1.5 is selected**.
Shows the full compiled positive string including subject boost and quality boosters.
Hidden for SDXL (output matches input).

---

# File map

```
src/prompt/
├── Prompt.hpp         — Token, Prompt structs
├── PromptParser.hpp   — parse() declaration
├── PromptParser.cpp   — parse() + A1111 weight parsing
├── PromptCompiler.hpp — compile(), compileNegative() declarations
├── PromptCompiler.cpp — model-specific compilation + logging
├── PromptMerge.hpp    — merge() declaration
├── PromptMerge.cpp    — merge() implementation
└── PromptJson.hpp     — nlohmann_json ADL hooks (header-only)
```
