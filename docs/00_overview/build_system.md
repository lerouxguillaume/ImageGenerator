# Build System

## What this file explains
CMake configuration, build flags, and platform-specific build instructions for GuildMaster.

## When to use this
- Setting up a new build environment
- Enabling GPU acceleration (CUDA or DML)
- Adding new `.cpp` files to the project
- Cross-compiling for Windows from Linux

## Key invariants
- CMake uses `GLOB_RECURSE` for source discovery — **re-run `cmake -B build` after adding any new `.cpp` file**; forgetting this causes silent link errors from missing translation units.
- `USE_DML=ON` only covers the UNet. VAE and text encoders always run on CPU under DML.
- ORT paths have defaults in CMakeLists.txt but can be overridden.

## Mental model
Two orthogonal axes: **platform** (Linux native vs Windows cross-compile) and **EP** (CPU-only vs CUDA vs DML). EP selection is compile-time, not runtime.

## Implementation details

### Build commands

```bash
# Linux — CPU only
cmake -B build && cmake --build build -j$(nproc)

# Linux — CUDA EP for all models
cmake -B build -DUSE_CUDA=ON && cmake --build build -j$(nproc)

# Windows cross-compile from Linux
cmake -B build -DCMAKE_TOOLCHAIN_FILE=cmake/mingw-w64.cmake
cmake --build build -j$(nproc)
```

### CMake flags

| Flag | Effect |
|---|---|
| `USE_CUDA=ON` | Enables CUDA EP for UNet, VAE, and text encoders |
| `USE_DML=ON` | Enables DirectML EP for UNet only |
| `USE_GENAI=ON` | Enables ORT GenAI for LLM prompt enhancement |
| `-DONNXRUNTIME_ROOT=<path>` | Override default ORT installation path |

### DML session options (do not change without Windows testing)
```cpp
session_opts.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
session_opts.DisableMemPattern();
session_opts.SetExecutionMode(ORT_SEQUENTIAL);
```

## Common pitfalls
- Adding a `.cpp` file without re-running CMake configure step → new file silently excluded from build.
- Building with `USE_DML=ON` on Linux → DML is a Windows-only EP; build will fail or produce a non-functional binary.
- Forgetting `-j$(nproc)` → very slow single-threaded build.

## Related files
- [`docs/10_pipeline/gpu_fallback.md`](../10_pipeline/gpu_fallback.md) — runtime DML fallback behaviour
- `cmake/mingw-w64.cmake` — Windows cross-compilation toolchain
- `CMakeLists.txt` — source lists and EP configuration
