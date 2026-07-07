#pragma once
#include <algorithm>
#include <atomic>
#include <stop_token>
#include <string>
#include <vector>

#include "../enum/enums.hpp"
#include "../config/AppConfig.hpp"   // for LoraEntry

// ── Hires-fix configuration ───────────────────────────────────────────────────
// SD1.5 "hires fix": after the base pass at native resolution, run a second
// low-strength denoise at a higher resolution (bilinear latent upscale), then a
// single decode of the final latent. Off by default; inert when disabled.
struct HiresConfig {
    bool        enabled  = false;
    float       scale    = 1.5f;   // target pixel dims = native * scale, snapped to /64
    float       strength = 0.5f;   // pass-2 denoise fraction (UI range 0.3–0.7)
    int         steps    = 0;      // 0 = reuse the base numSteps for pass 2
    UpscaleMode mode     = UpscaleMode::Pixel;  // Pixel adds detail; Latent blurs
};

// Parameters shared by all generation entry points.
struct GenerationParams {
    int     numSteps             = 20;   // DPM++ 2M Karras denoising steps (more = better quality, slower)
    float   guidanceScale        = 8.0f; // CFG scale: how strongly the prompt steers the output (7–12 typical)
    float   negativeGuidanceScale = 0.0f; // Negative prompt CFG scale; 0 = same as guidanceScale (standard CFG)
    float   cfgRescale           = 0.0f; // CFG rescale factor (0 = off, 0.7 typical); reduces oversaturation at high guidance
    int     numImages            = 1;    // Number of images to generate sequentially in one pipeline run
    int     width                = 0;    // Output width in pixels; 0 = use model default (512 for SD1.5, 1024 for SDXL)
    int     height               = 0;    // Output height in pixels; 0 = use model default
    int64_t seed                 = -1;   // RNG seed; -1 = generate randomly. For multi-image runs, seed+i is used per image.
    std::vector<LoraEntry> loras;        // LoRA adapters to apply; empty = none
    std::string initImagePath;           // img2img: path to input image; empty = txt2img
    float       strength         = 1.0f; // img2img: denoising strength [0,1]; 1 = full noise (txt2img)
    int         cropTop          = 0;    // SDXL time_ids: crop_y offset in pixels (0 = no crop)
    int         cropLeft         = 0;    // SDXL time_ids: crop_x offset in pixels (0 = no crop)
    HiresConfig hires;                   // hires-fix pass-2 config; enabled=false → inert

    // ── Hires step accounting (single source of truth for pipeline + UI) ──────
    // The pipeline uses these to drive the pass-2 denoise; the progress bar uses
    // totalDenoiseSteps() as its denominator so it spans both passes.
    int hiresEffectiveSteps() const { return hires.steps > 0 ? hires.steps : numSteps; }
    // img2img-style truncation: pass 2 starts partway down the schedule.
    int hiresStartStep() const {
        const int eff = hiresEffectiveSteps();
        const int s   = static_cast<int>((1.0f - hires.strength) * static_cast<float>(eff));
        return std::max(0, std::min(s, eff - 1));
    }
    // Steps actually executed in pass 2 (the truncated count, not effectiveSteps).
    int hiresExtraSteps() const {
        return hires.enabled ? (hiresEffectiveSteps() - hiresStartStep()) : 0;
    }
    // Cumulative denoise-step count across the base pass and the hires pass.
    int totalDenoiseSteps() const { return numSteps + hiresExtraSteps(); }
};

// Static facade over the full Stable Diffusion pipeline (text encoding → UNet denoising → VAE decode).
// All methods run synchronously on the calling thread; launch in a std::thread for non-blocking use.
// Execution provider (CPU / CUDA / DML) is selected at compile time via USE_CUDA / USE_DML defines.
class PortraitGeneratorAi {
public:
    // Generate an image from an explicit prompt pair.
    // outputPath: destination .png file (parent directory must exist or be creatable).
    // modelDir:   directory containing text_encoder.onnx, unet.onnx, vae_decoder.onnx,
    //             and optionally text_encoder_2.onnx (SDXL) plus model.json.
    // progressStep: incremented after each denoising step.
    // currentImage: set to the 1-based index of the image currently being generated.
    // cancelToken:  set to true from another thread to abort; the pipeline checks it
    //               before each step and also calls OrtRunOptions::SetTerminate() to
    //               abort any in-flight ORT Run() call immediately.
    static void generateFromPrompt(const std::string& prompt,
                                   const std::string& negativePrompt,
                                   const std::string& outputPath,
                                   const GenerationParams& params,
                                   const std::string& modelDir     = "models",
                                   std::atomic<int>* progressStep  = nullptr,
                                   std::atomic<int>* currentImage  = nullptr,
                                   std::stop_token   stopToken     = {},
                                   std::atomic<GenerationStage>* stage = nullptr);
};
