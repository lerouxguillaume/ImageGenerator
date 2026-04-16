# MultiLineTextArea

Custom text input widget.

---

# Features

- word wrap
- cursor navigation
- copy/paste
- scrolling
- selection support

---

# Constraints

- ASCII input only (filtered)
- char limit enforced
- self-contained state machine

---

# Rules

- Do NOT manipulate cursor, scroll, or text fields directly — use the public API (`setText`, `setActive`, `handleEvent`, `handleScroll`, `handleClick`)
- Do NOT call `computeLines()` or `drawPromptField()` — those functions no longer exist
- Do NOT render or route clicks to `instructionArea` when `view.promptEnhancerAvailable` is false — the field only exists in the UI when an LLM model is loaded