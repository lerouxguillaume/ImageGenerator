# UI Architecture

SFML-based retained UI system.

---

# Structure

- View: rendering only
- Controller: input + logic
- Widgets: self-contained components

---

# Key principle

Widgets own their state fully.
No external cursor management.