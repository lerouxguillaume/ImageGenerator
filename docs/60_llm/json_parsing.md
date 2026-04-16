# JSON Parsing

## Streaming completion: `JsonStreamTracker`

Fed one character at a time from the token stream. Signals when the outermost JSON object is structurally complete so generation stops immediately — before the model can drift into trailing text.

State machine (outside string context):
- `{` → `depth++`
- `}` → `depth--`; if `depth == 0` → `done = true`, returns `true`
- `"` → enter string context

Inside string context:
- `\` → arm escape (next char consumed without interpretation)
- `"` (unescaped) → leave string context

The pre-filled `{` that ends the chat prompt is fed to the tracker before the generation loop starts, so the tracker begins at `depth = 1`.

Fallback: if no streaming completion fires (e.g. model emits whitespace before `}`), the loop runs to natural EOS or `max_length`, and `extractFirstValidResponse()` scans the full collected output.

## Post-generation extraction: `extractFirstValidResponse`

Scans `raw` for `{` characters. For each one, calls `extractJsonObject()` (balanced-brace extractor respecting quoted strings), then `validateSchema()`. Returns the first candidate that passes schema validation.

"First" is correct: with the `{` prefill the model cannot emit preamble before the object, so the first valid object is the intended output. Returning immediately avoids being fooled by trailing garbage before EOS.

## Schema validator: `validateSchema`

Strict requirements:
- Exactly two keys: `"prompt"` and `"negative_prompt"`.
- Both must be non-empty strings.
- No extra keys allowed.

Returns `LLMResponse` on success, `nullopt` on any failure. Falls back to the original prompt + `defaultNeg` if no valid response is extracted.

## Generation parameters

| Parameter | Value | Rationale |
|---|---|---|
| `max_length` | 800 | Long enough for a rich prompt + negative |
| `temperature` | 0.15 | Near-greedy — structural tokens are deterministic; content still varies |
| `top_p` | 0.9 | Nucleus sampling for content diversity |
| `repetition_penalty` | 1.05 | Light penalty to discourage keyword repetition |

## EOS token stripping

Before extraction, known EOS tokens are stripped from the raw output:
`<|eot_id|>`, `</s>`, `<|end|>`, `<eos>`

This handles the EOS/max_length stop path where the model appends a terminator after the closing `}`.
