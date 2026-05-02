# Model Import Pipeline

Converts a `.safetensors` checkpoint into the application's controlled ONNX format without any manual steps from the user.

The application **never uses `.safetensors` at runtime**. ONNX is the only inference format.

---

# Entry point

Menu → **Import Model** button → `ImportModelModal`.

---

# Components

## SafetensorsInspector (`src/import/SafetensorsInspector.*`)

Reads only the JSON header of a `.safetensors` file (first 8 bytes = header size, then header JSON). No tensor data is loaded.

Detection is done by substring search on the raw header string:

| Signal | Detection |
|---|---|
| `conditioner.embedders.1.` present | SDXL |
| `model.diffusion_model.` or `cond_stage_model.` present | SD 1.5 |
| ≥ 5 keys ending in `lora_up.weight` / `lora_down.weight` | LoRA adapter |
| `encoder.` + `decoder.` present, no diffusion model keys | Standalone VAE |

Dominant dtype (`F16` / `F32` / `BF16`) is determined by counting dtype string occurrences in the header.

LoRA adapters are rejected at the inspection stage — they do not go through this pipeline.

---

## PythonEnvManager (`src/import/PythonEnvManager.*`)

Manages a dedicated Python virtual environment. On first use it:

1. Locates a supported system Python (`3.10` / `3.11` / `3.12`; `py -3.12/-3.11/-3.10` is tried on Windows)
2. Resolves the executable path and logs the selected command
3. Runs `python -m venv <venvDir>`
4. Upgrades pip
5. Installs `scripts/requirements_import.txt`
6. Writes a `.setup_complete` sentinel containing the requirements signature

Subsequent runs skip setup when the sentinel matches the current requirements file.

### Venv location (platform-specific)

The venv must be on a local filesystem. Network and shared drives (e.g. VirtualBox `sf_shared_vm`) do not support the symlinks / junctions that Python venv creation requires.

| Platform | Venv path |
|---|---|
| Windows | `%LOCALAPPDATA%\ImageGenerator\python_env\` |
| Linux | `<executableDir>/python_env/` |

### Setup log

Full command output (including the exact Python path used and all pip output) is written to:

| Platform | Log path |
|---|---|
| Windows | `%LOCALAPPDATA%\ImageGenerator\python_env_setup.log` |
| Linux | `<executableDir>/python_env_setup.log` |

Check this file first when setup fails — the modal only shows the last 8 lines.

---

## Subprocess (`src/import/Subprocess.*`)

Cross-platform subprocess wrapper. Both stdout and stderr are captured via a pipe and read line by line from a background thread.

| Platform | Implementation |
|---|---|
| Linux | `fork()` + `execvp()` + anonymous pipe |
| Windows | `CreateProcess()` + anonymous pipe (`STARTF_USESTDHANDLES`) |

`kill()` sends `SIGTERM` on Linux and `TerminateProcess()` on Windows.

---

## ModelImporter (`src/import/ModelImporter.*`)

Orchestrates the full import on a dedicated background thread. All state is accessed from the UI thread via thread-safe atomics and mutex-protected accessors.

### State machine

```
Idle → Analyzing → SettingUpPython → Exporting → Validating → Done
                                                             ↘ Failed
```

### Thread responsibilities

1. Run `SafetensorsInspector` on the source file
2. Determine model ID (sanitized filename stem) and output path (`models/imported/<id>/`)
3. Call `PythonEnvManager::setup()` if the venv sentinel is absent
4. Spawn `scripts/import_model.py` via `Subprocess`
5. Read stdout line by line — lines prefixed `PROGRESS:` advance the state machine; `ERROR:` lines set the status message; all other lines go to the log buffer
6. On clean exit, transition to `Done`; on non-zero exit or cancellation, transition to `Failed`

### Cancellation

`ModelImporter::cancel()` sets `cancelRequested_` and calls `Subprocess::kill()` on the active subprocess immediately. The thread checks `cancelRequested_` between steps and after each line read.

---

## ImportedModelRegistry (`src/import/ImportedModelRegistry.*`)

JSON-backed list of successfully imported models. Stored at:

```
models/imported/registry.json
```

Schema:

```json
{
  "models": [
    {
      "id":         "animagine_xl_v3",
      "name":       "animagine_xl_v3",
      "arch":       "sdxl",
      "onnxPath":   "C:/path/to/models/imported/animagine_xl_v3",
      "importedAt": "2025-05-01T12:00:00Z"
    }
  ]
}
```

`add()` replaces any existing entry with the same id. Registry is loaded at startup and saved after every `add()`.

---

## ImportModelModal (`src/ui/widgets/ImportModelModal.*`)

Overlay modal following the `SettingsModal` action-flag pattern.

| Control | Behaviour |
|---|---|
| Browse button | Sets `browseRequested` flag; controller calls the OS file picker and writes back `filePath` |
| Arch cycle button | Cycles Auto → SD 1.5 → SDXL on each click |
| Import button | Sets `importRequested` when idle and a file is selected |
| Cancel button | Sets `cancelRequested` while import is running |
| Close button | Sets `closeRequested`; ignored while import is running |

`syncFrom(importer)` is called by `MenuController::renderOverlay()` each frame to pull the latest log lines and status message before rendering.

---

# Import script (`scripts/import_model.py`)

Unified Python entry point wrapping the existing per-architecture export scripts.

```
python import_model.py --input <file> --output <dir> --arch auto|sd15|sdxl
```

### PROGRESS: protocol

The C++ parent reads stdout line by line and classifies each line:

| Prefix | Action |
|---|---|
| `PROGRESS:analyzing` | State → Analyzing |
| `PROGRESS:exporting_sd15` or `PROGRESS:exporting_sdxl` | State → Exporting |
| `PROGRESS:validating` | State → Validating |
| `PROGRESS:done` | State → Done |
| `ERROR:<msg>` | Sets status message; state → Failed after subprocess exits |
| (anything else) | Appended to log buffer |

### Architecture detection

If `--arch auto`, the script reads the safetensors header directly (no torch needed) and searches for `conditioner.embedders.1.` (SDXL) or `model.diffusion_model.` / `cond_stage_model.` (SD 1.5). This mirrors the C++ `SafetensorsInspector` logic and serves as a fallback when C++ detection returns Unknown.

### Capabilities block

After export completes, `import_model.py` extends `model.json` with:

```json
{
  "type": "sdxl",
  "vae_scaling_factor": 0.13025,
  "capabilities": {
    "vae_encoder_available": true,
    "lora_compatible": true,
    "components": {
      "text_encoder":   { "dtype": "fp32" },
      "text_encoder_2": { "dtype": "fp32" },
      "unet":           { "dtype": "fp16" },
      "vae_decoder":    { "dtype": "fp16" },
      "vae_encoder":    { "dtype": "fp16" }
    }
  }
}
```

---

# Directory layout

```
models/
  imported/
    registry.json
    <model_id>/
      text_encoder.onnx[.data]
      text_encoder_2.onnx[.data]   (SDXL only)
      unet.onnx[.data]
      vae_decoder.onnx[.data]
      vae_encoder.onnx[.data]
      model.json                   (includes capabilities block)
```

---

# Python prerequisites

Python is **not** required at runtime. It is only required when running the import pipeline.

- **Supported version**: Python 3.10, 3.11, or 3.12
- **Unsupported**: Python 3.13+; newer dependency resolution can leave CLIP weights on the PyTorch `meta` device during export.
- **Must be on system PATH** (`python` / `python3` / `py`)
- The venv and all packages are managed automatically after Python is installed

Required packages (installed automatically into the managed venv):

```
torch>=2.0.0
diffusers>=0.27.0,<0.35.0
transformers>=4.30.0,<4.47.0
onnx>=1.14.0
onnxscript>=0.1.0
onnxruntime>=1.16.0
safetensors>=0.3.0
accelerate>=0.21.0
```

---

# Invariants

- Never use `.safetensors` in the inference pipeline — ONNX only
- Never place the venv on a network or shared drive — use `localDataDir()` which resolves to `%LOCALAPPDATA%` on Windows
- Never call `ImportedModelRegistry::add()` before `ModelImporter::getState() == Done`
- Never read `getOutputDir()` / `getModelId()` before `State::Done`
- Never regenerate the sentinel file manually — it is written only by `PythonEnvManager::setup()` on full success
