# Build System

GuildMaster uses CMake as its primary build system with optional GPU backends.

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

# Important CMake behavior

The project uses:

- `GLOB_RECURSE` for source discovery

⚠️ Implication:
New `.cpp` files are NOT automatically picked up until CMake is re-run.

You must run:

```bash
cmake -B build
```

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
