# Build System

ImageGenerator uses CMake as its primary build system with optional GPU backends.

## Path variables

- `ONNXRUNTIME_ROOT` — ONNX Runtime root (defaults are OS/backend-specific; override for a custom install).
- `ORT_GENAI_ROOT` — ONNX Runtime GenAI root; only consulted when `USE_GENAI=ON`.

---

# Configuration flags

## USE_CUDA
Enables CUDA Execution Provider for ONNX Runtime.

Applies to:
- UNet
- CLIP text encoders
- VAE (if supported by runtime)

This provides full GPU acceleration when CUDA-compatible hardware is available.

---

## USE_DML
Enables DirectML Execution Provider (Windows only).

Applies to:
- UNet only

Limitations:
- CLIP text encoders run on CPU
- VAE runs on CPU due to known ONNX/DML operator limitations (e.g. Reshape issues)

---

## USE_GENAI
Enables ONNX Runtime GenAI for the optional offline LLM prompt enhancer
(`OrtLlmEnhancer`). Requires `ORT_GENAI_ROOT`. Off by default; when off, a
null enhancer is used and the LLM bar stays hidden.

---

# Important CMake behavior

The project uses `GLOB_RECURSE ... CONFIGURE_DEPENDS` for source discovery.
`CONFIGURE_DEPENDS` asks the generator to re-glob on the next build, so newly
added/removed `.cpp` files are usually picked up automatically. This is not
100% reliable across all generators, so after adding or removing a `.cpp` the
safe habit is still to re-run:

```bash
cmake -B build
```

## Post-build copy steps

The build copies runtime assets next to the binary via POST_BUILD commands:
- `config.json`
- `models/` (tokenizer files `vocab.json` + `merges.txt`)
- `scripts/` (the model-import Python)
- creates `assets/generated/` (output dir) with a `.gitkeep`
- on Windows: required runtime DLLs

---

# Static analysis and formatting

CMake exports `build/compile_commands.json` for editor tooling and targeted
`clang-tidy` runs.

Targeted lint command:

```bash
clang-tidy -p build src/import/ImportedModelRegistry.cpp
```

Run lint on focused files while cleaning up a subsystem. Do not enable clang-tidy
globally in every build yet; the current `.clang-tidy` profile is intentionally
strict and the whole codebase is not at a zero-warning baseline.

Formatting is defined by the repo-root `.clang-format`. Format touched C++ files
with:

```bash
clang-format -i src/path/to/file.cpp
```
