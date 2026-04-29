# ImageGenerator

Generate images locally using Stable Diffusion (SD 1.5 or SDXL) via ONNX Runtime — no cloud API required. Includes a project system for organised game asset generation (isometric tiles, sprites, props, etc.).

Built with C++20, SFML, and ONNX Runtime. Runs on Linux and Windows; GPU acceleration via CUDA or DirectML.

---

## Features

- **Local SD inference** — SD 1.5 and SDXL pipelines implemented from scratch (DPM++ 2M Karras scheduler, CFG, CLIP tokenizer)
- **GPU acceleration** — CUDA (Linux/Windows) or DirectML (Windows)
- **Three entry points** — `Projects`, `Generate Images`, and `Edit Image` from the main menu
- **Project workspace** — build themed asset packs with a project-level theme, per-asset prompts, inline generation controls, and an embedded results gallery
- **SFML GUI** — prompt editor, model selector, edit instruction field, strength controls, live progress overlay
- **Multi-image generation** — generate N images in one run with cancellation support
- **Image editing** — dedicated edit screen plus gallery-driven handoff from generated results
- **Model hot-swap** — any model exported to ONNX under `models/` is automatically listed in the UI
- **Preset system** — save/load named generation configurations independent of projects

---

## Project structure

```
├── src/
│   ├── portraits/          # SD pipeline: tokenizer, prompt builder, ONNX inference
│   ├── controllers/        # Input handling (MVC controller layer)
│   ├── views/              # SFML rendering (MVC view layer)
│   ├── projects/           # Project + AssetType data model and ProjectManager
│   ├── presets/            # Preset data model and PresetManager
│   ├── prompt/             # Prompt DSL (parse / compile / merge / JSON)
│   ├── managers/           # Logger
│   ├── ui/                 # Shared widgets, theme, helpers
│   └── enum/               # Enums and layout/colour constants
├── scripts/
│   ├── export_onnx_models.py       # Export SD 1.5 model → ONNX
│   ├── sdxl_export_onnx_models.py  # Export SDXL model → ONNX
│   └── build-windows.sh            # Cross-compile Windows release from Linux
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
│   └── generated/          # Output images (flat or project/assettype/ subfolders)
├── deps/windows/           # Pre-built Windows dependencies (SFML, ORT, OpenCV)
├── projects.json           # Saved projects (auto-created)
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
| nlohmann/json | 3.x | JSON config / presets / projects |

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

The main menu offers three entry points:

**Projects** — create named projects for themed asset sets (e.g. a game tileset). Each project is a dedicated workspace: define the shared project theme once, define per-asset prompts for asset types such as `Wall`, `Floor`, `Character`, or `Prop`, then generate directly inside the same screen. Outputs are saved to `assets/generated/<project>/<asset_type>/`, and the embedded results panel shows the gallery for the currently selected asset type.

**Generate Images** — standalone prompt-first txt2img workflow with preset support.

**Edit Image** — image-first img2img workflow. From the generate screen, select a gallery image and click `Edit` to open it here with the image preselected. Enter an instruction such as `change hair color to copper red`, adjust the strength slider, and generate.

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
