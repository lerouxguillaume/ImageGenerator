# ImageGenerator

Generate and edit images locally using Stable Diffusion (SD 1.5 or SDXL) via ONNX Runtime — no cloud API required.

Built with C++20, SFML, and ONNX Runtime. Runs on Linux and Windows; GPU acceleration via CUDA or DirectML.

---

## Features

- **Local SD inference** — SD 1.5 and SDXL pipelines implemented from scratch (DPM++ 2M Karras scheduler, CFG, CLIP tokenizer)
- **GPU acceleration** — CUDA (Linux/Windows) or DirectML (Windows)
- **Unified generate/edit screen** — one prompt-first screen; attach an optional input image to switch from txt2img to img2img
- **SFML GUI** — centralized theme system, prompt editor, model selector, edit instruction field, strength controls, live progress overlay
- **Multi-image generation** — generate N images in one run with cancellation support
- **Model import** — import a `.safetensors` checkpoint through the in-app Import Model flow; imported models are listed from the registry
- **Preset system** — save/load named generation configurations

---

## Project structure

```
├── src/
│   ├── portraits/          # SD pipeline: tokenizer, ONNX inference
│   ├── import/             # Model import pipeline + ImportedModelRegistry
│   ├── controllers/        # Input handling (MVC controller layer)
│   ├── views/              # SFML rendering (MVC view layer)
│   ├── presets/            # Preset data model and PresetManager
│   ├── prompt/             # Prompt DSL (parse / compile / merge / JSON)
│   ├── llm/                # Optional ONNX GenAI prompt enhancer
│   ├── managers/           # Logger
│   ├── ui/                 # Shared widgets, theme tokens, helpers
│   └── enum/               # Enums and compatibility layout/colour constants
├── scripts/
│   ├── import_model.py             # Import/export a .safetensors model → ONNX
│   └── build-windows.sh            # Cross-compile Windows release from Linux
├── models/                 # Tokenizer files + imported models
│   ├── vocab.json
│   ├── merges.txt
│   └── imported/           # Imported models (registry-managed)
│       ├── registry.json   # Model registry (source of truth for discovery)
│       └── <id>/           # One directory per imported model
│           ├── text_encoder.onnx
│           ├── unet.onnx
│           ├── vae_decoder.onnx
│           ├── vae_encoder.onnx      # optional (img2img)
│           ├── text_encoder_2.onnx   # SDXL only
│           └── model.json            # arch + capabilities block
├── assets/
│   └── generated/          # Output images (flat, newest-first gallery)
├── deps/windows/           # Pre-built Windows dependencies (SFML, ORT, OpenCV)
├── presets.json            # Saved presets (auto-created)
└── CMakeLists.txt
```

---

## Dependencies

| Library | Version | Purpose |
|---|---|---|
| SFML | 2.5+ | Window, rendering, input |
| ONNX Runtime | 1.24+ | Model inference |
| OpenCV | 4.x | Image encoding (PNG write) |
| nlohmann/json | 3.x | JSON config / presets / model registry |

---

## Building

### Linux — CPU only

```bash
sudo apt install libsfml-dev libopencv-dev nlohmann-json3-dev

# Download ONNX Runtime CPU build and unpack to ~/libs/onnxruntime/
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DONNXRUNTIME_ROOT=~/libs/onnxruntime/onnxruntime-linux-x64-1.24.4
cmake --build build -j$(nproc)
```

### Linux — CUDA

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DUSE_CUDA=ON \
      -DONNXRUNTIME_ROOT=~/libs/onnxruntime/onnxruntime-linux-x64-gpu-1.24.4
cmake --build build -j$(nproc)
```

### Windows (cross-compiled from Linux with MinGW)

Use the automated script — it downloads all Windows dependencies and packages a zip:

```bash
# DirectML (default)
bash scripts/build-windows.sh

# CUDA GPU
bash scripts/build-windows.sh --cuda

# With LLM prompt enhancer
bash scripts/build-windows.sh --genai
bash scripts/build-windows.sh --cuda --genai
```

Prerequisites: `cmake wget unzip gendef x86_64-w64-mingw32-g++-posix`

```bash
sudo apt install cmake wget unzip mingw-w64-tools \
                 gcc-mingw-w64-x86-64-posix g++-mingw-w64-x86-64-posix
```

---

## Exporting models

Models must be exported to ONNX before use. Export scripts live in `scripts/`.

### SD 1.5

```bash
pip install torch diffusers transformers accelerate onnx onnxscript

# Place your .safetensors file next to the script, then:
cd scripts
python export_onnx_models.py
# Writes models/<model_name>/{text_encoder,unet,vae_decoder,vae_encoder}.onnx
```

The SD 1.5 script exports the UNet and VAE as **float16** for best GPU performance.

### SDXL (Illustrious / Pony / NoobAI / SDXL-base)

```bash
cd scripts
python sdxl_export_onnx_models.py
# Writes models/<model_name>/{text_encoder,text_encoder_2,unet,vae_decoder,vae_encoder}.onnx + model.json
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

The binary expects `models/vocab.json` and `models/merges.txt` in the working directory (copied automatically by CMake post-build). Models are discovered from the registry at `models/imported/registry.json` — populated by the in-app Import Model flow.

Generated images are saved to `assets/generated/`.

The main menu opens the generate/edit screen and offers an **Import Model** action:

**Generate / Edit** — a single prompt-first workflow with preset support and the shared dark tool UI. It runs txt2img by default; select a gallery image and click `Edit` (or attach an input image) to switch to img2img, then enter an instruction such as `change hair color to copper red`, adjust the strength slider, and generate.

**Import Model** — import a `.safetensors` checkpoint. A managed Python venv is set up automatically on first use; the model is exported to ONNX under `models/imported/<id>/` and added to the registry.

UI styling is now centralized in `src/ui/Theme.h`:
- `UiColors` — shared palette and semantic surface colors
- `UiMetrics` — spacing, control sizes, and layout constants
- `UiTypography` — shared text sizing

`src/enum/constants.hpp` remains as a compatibility layer while older widgets continue migrating to the centralized theme tokens.

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
                        └─ vae_encoder.onnx for img2img input images
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
