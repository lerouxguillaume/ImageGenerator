# JSON Parsing

LLM output parsing is robust against malformed output.

---

# Strategy

1. parse full output
2. fallback substring extraction
3. final optional failure fallback

---

# Goal

Never crash pipeline due to malformed LLM output.