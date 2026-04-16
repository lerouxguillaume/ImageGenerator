# Adding a New Model

To support a new diffusion model:

---

# Step 1: register type

Add to `ModelType` enum.

---

# Step 2: detect in loader

Extend `loadModelConfig()`:
- map string → enum

---

# Step 3: load sessions

Add session loading logic in `loadModels()`.

---

# Step 4: text encoding

Add new encoder function if required.

---

# Step 5: pipeline wiring

Extend `runPipeline()` branching.

---

# Step 6: export script

Create ONNX export script for model.