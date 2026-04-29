# ImageGenerator

Character portraits are generated locally using Stable Diffusion (SD 1.5 or SDXL) via ONNX Runtime — no cloud API required.

Built with C++17, SFML, and ONNX Runtime. Runs on Linux and Windows; GPU acceleration via CUDA or DirectML.

---

## Features

- **Local SD inference** — SD 1.5 and SDXL pipelines implemented from scratch (DPM++ 2M Karras scheduler, CFG, CLIP tokenizer)
- **GPU acceleration** — CUDA (Linux/Windows) or DirectML (Windows)
- **SFML GUI** — prompt editor, model selector, edit instruction field, strength controls, live progress overlay
- **Multi-image generation** — generate N images in one run with cancellation support
- **Image editing** — gallery-driven img2img workflow with `Edit`, init-image reuse, and strength presets
- **Model hot-swap** — any model exported to ONNX under `models/` is automatically listed in the UI

---

## Project structure

```
├── src/
│   ├── portraits/          # SD pipeline: tokenizer, prompt builder, ONNX inference
│   ├── controllers/        # Input handling (MVC controller layer)
│   ├── views/              # SFML rendering (MVC view layer)
│   ├── presenters/         # State mutations (MVC presenter layer)
│   ├── services/           # Background thread wrappers
│   ├── managers/           # Logger
│   ├── ui/                 # Shared widgets, theme, helpers
│   └── enum/               # Enums and layout/colour constants
├── scripts/
│   ├── export_onnx_models.py       # Export SD 1.5 model → ONNX
│   └── sdxl_export_onnx_models.py  # Export SDXL model → ONNX
├── models/                 # ONNX model directories + tokenizer files
│   ├── vocab.json
│   ├── merges.txt
│   └── <model_name>/       # One subdirectory per exported model
│       ├── text_encoder.onnx
│       ├── unet.onnx
│       ├── vae_decoder.onnx
│       ├── text_encoder_2.onnx  # SDXL only
│       └── model.json           # {"type": "sdxl"} or absent → SD 1.5
├── assets/
│   └── generated/          # Output images written here
├── deps/windows/           # Pre-built Windows dependencies (SFML, ORT, OpenCV)
└── CMakeLists.txt
```

---

## Dependencies

| Library | Version | Purpose |
|---|---|---|
| SFML | 2.5+ | Window, rendering, input |
| ONNX Runtime | 1.20+ | Model inference |
| OpenCV | 4.x | Image encoding (PNG write) |
| nlohmann/json | 3.x | `model.json` parsing |
| SQLite3 | 3.x | Game data persistence |

---

## Building

### Linux — CPU only

```bash
sudo apt install libsfml-dev libopencv-dev nlohmann-json3-dev libsqlite3-dev

# Download ONNX Runtime CPU build and unpack to ~/libs/onnxruntime/
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### Linux — CUDA

```bash
# Set ONNXRUNTIME_ROOT to the GPU build
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DUSE_CUDA=ON \
      -DONNXRUNTIME_ROOT=~/libs/onnxruntime/onnxruntime-linux-x64-gpu-1.24.4
cmake --build build -j$(nproc)
```

### Windows — CPU (cross-compiled from Linux with MinGW)

```bash
cmake -B build-windows \
      -DCMAKE_TOOLCHAIN_FILE=cmake/mingw-w64.cmake \
      -DCMAKE_BUILD_TYPE=Release \
      -DONNXRUNTIME_ROOT=deps/windows/onnxruntime-win-x64-1.20.1 \
      -DSFML_DIR=deps/windows/SFML-2.6.1/lib/cmake/SFML \
      -DOpenCV_DIR=deps/windows/opencv-4.10.0/lib/cmake/opencv4 \
      -DSQLITE3_ROOT=deps/windows/sqlite3
cmake --build build-windows -j$(nproc)
```

### Windows — DirectML (GPU, cross-compiled from Linux)

```bash
cmake -B build-windows \
      -DCMAKE_TOOLCHAIN_FILE=cmake/mingw-w64.cmake \
      -DCMAKE_BUILD_TYPE=Release \
      -DUSE_DML=ON \
      -DONNXRUNTIME_ROOT=deps/windows/onnxruntime-dml-1.20.1 \
      ...
cmake --build build-windows -j$(nproc)
```

### Windows — CUDA

```bash
# Use the CUDA ORT package
cmake -B build-windows \
      -DCMAKE_TOOLCHAIN_FILE=cmake/mingw-w64.cmake \
      -DCMAKE_BUILD_TYPE=Release \
      -DUSE_CUDA=ON \
      -DONNXRUNTIME_ROOT=deps/windows/onnxruntime-cuda-1.20.1 \
      ...
cmake --build build-windows -j$(nproc)
```

---

## Exporting models

Models must be exported to ONNX before use. Export scripts live in `scripts/`.

### SD 1.5

```bash
pip install torch diffusers transformers accelerate onnx onnxsim

# Place your .safetensors file next to the script, then:
cd scripts
python export_onnx_models.py
# Writes models/<model_name>/{text_encoder,unet,vae_decoder}.onnx
```

The SD 1.5 script exports the UNet and VAE as **float16** for best GPU performance.

### SDXL (Illustrious / Pony / NoobAI / SDXL-base)

```bash
cd scripts
python sdxl_export_onnx_models.py
# Writes models/<model_name>/{text_encoder,text_encoder_2,unet,vae_decoder}.onnx + model.json
```

The SDXL script exports UNet and VAE as **float16**. Text encoders stay float32 for accuracy.

A `model.json` sidecar is written automatically so the application detects SDXL at load time:
```json
{ "type": "sdxl" }
```

---

## Running

```bash
./build/image_generator
```

The binary expects `models/vocab.json` and `models/merges.txt` in the working directory (copied automatically by CMake post-build). Model subdirectories are auto-discovered from `models/`.

Generated images are saved to `assets/generated/`.

To edit an existing image, select it in the gallery, click `Edit`, enter an instruction such as `change hair color to copper red`, then adjust the strength preset or slider before generating.

---

## Inference pipeline

```
Prompt text
    └─ ClipTokenizer → token IDs
         └─ text_encoder.onnx  (+ text_encoder_2.onnx for SDXL)
              └─ text embeddings
                   └─ DPM++ 2M Karras loop (N steps)
                        │  unet.onnx  ×2 per step (uncond + cond CFG)
                        └─ denoised latent
                             └─ vae_decoder.onnx
                                  └─ PNG image
```

**Scheduler:** DPM++ 2M Karras with configurable step count (default 20).  
**CFG:** Two separate batch=1 UNet passes per step (static-batch models, required for DirectML).  
**Cancellation:** A watcher thread calls `OrtRunOptions::SetTerminate()` the moment the cancel token is set, aborting any in-flight ORT `Run()` call.

---

## EP selection (execution provider)

| Build flag | UNet | VAE | Text encoders |
|---|---|---|---|
| _(none)_ | CPU | CPU | CPU |
| `USE_CUDA` | CUDA | CUDA | CUDA |
| `USE_DML` | DML (SD 1.5) / CPU (SDXL) | CPU | CPU |

DML limitations: SDXL UNet has Reshape nodes (`node_view_1`) that DML rejects at runtime, so it falls back to CPU. VAE has a Reshape in its attention block (`node_view_2`) that DML also rejects regardless of optimization level.

---

## Logs

The application writes timestamped logs to `image_generator.log` in the working directory. Key log sections:

- `=== loadModels ===` — model load times, IO dtypes, EP assignment
- `DPM++ step N/M` — per-step sigma, uncond/cond UNet timing, eps range, latent stats
- `VAE decoding` — input latent stats, decode time, output image stats
