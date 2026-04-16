# System Overview

## What this file explains
High-level understanding of the GuildMaster image generation system.

## When to use this
- First contact with the project
- Understanding overall behavior
- Before diving into specific subsystems

## System purpose
GuildMaster generates character portraits using a fully local Stable Diffusion pipeline implemented in C++ on top of ONNX Runtime.

No Python is used at runtime. The system supports SD 1.5 and SDXL models and allows runtime LoRA application and optional LLM-based prompt enhancement.

---

## High-level flow

1. User provides prompt (and optional instruction)
2. (Optional) LLM refines prompt
3. Text is tokenized and encoded
4. Latent is iteratively denoised via UNet (CFG)
5. Final latent is decoded via VAE
6. Image is written to disk

---

## Core subsystems

- **UI layer** — input handling and rendering (SFML)
- **Pipeline** — orchestrates generation steps
- **Model system** — loads and caches ONNX sessions
- **LoRA system** — applies weight deltas at load time
- **Export pipeline** — prepares ONNX models offline
- **LLM enhancer** — optional prompt transformation

---

## Key constraints

- Entire pipeline runs locally (no external services)
- ONNX Runtime handles all tensor execution
- External tensor data is owned by ORT
- LoRA must be applied before inference (not during)
- GPU fallback must not break determinism

---

## Mental model

The system is a deterministic pipeline where:
- inputs → transformed → processed → decoded → output

Each subsystem is isolated and communicates through well-defined data structures.

---

## Related files

- [architecture_summary.md](architecture_summary.md)
- [build_system.md](build_system.md)
- [../10_pipeline/pipeline_orchestration.md](../10_pipeline/pipeline_orchestration.md)