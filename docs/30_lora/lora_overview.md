# LoRA System Overview

LoRA is applied at session creation time using ONNX external initializers.

---

# Core idea

- No runtime graph patching
- No inference-time modification
- All injection happens before session creation

---

# Components

- LoraInjector
- LoraParser
- LoraMath
- SdLoraMatch

---

# Flow

safetensors → parse → match → delta compute → inject → session create