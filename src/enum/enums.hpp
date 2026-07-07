#pragma once
#include <cstdint>

// ─── Inference ────────────────────────────────────────────────────────────────

// Diffusion model architecture. Drives resolution (512 vs 1024), dual text encoder,
// and SDXL-specific UNet inputs (text_embeds, time_ids).
enum class ModelType : std::uint8_t { SD15, SDXL };

// How a hires-fix refinement pass grows the base latent before its second
// denoise.
//   Pixel  — decode base → upscale the sharp RGB → re-encode. Anchors pass 2 on
//            a sharp low-freq structure so it ADDS detail. The good default.
//   Latent — bilinear upscale of the 4-ch latent directly. Cheaper (no extra
//            decode+encode) but off-manifold: the VAE amplifies the interpolation
//            into blur and pass 2 preserves it (measured softer than a plain
//            bicubic upscale). Kept for models without a VAE encoder / for A/B.
// Esrgan is a future variant (learned upscaler in place of cv::resize).
enum class UpscaleMode : std::uint8_t { Pixel, Latent /*, Esrgan (future)*/ };

// Stage reported by the generation thread to the UI thread via atomic<GenerationStage>.
// Generation cycles: LoadingModel → EncodingText → (EncodingImage) → Denoising
//   → (HiresDenoising) → DecodingImage → Done.
// Values are runtime-only (shared between threads within one process, never
// persisted), so inserting HiresDenoising mid-enum is safe.
enum class GenerationStage : int {
    Idle = 0,
    LoadingModel,
    EncodingText,
    EncodingImage,    // img2img VAE encode only
    Denoising,        // step counter is meaningful while in this stage
    HiresDenoising,   // hires-fix pass-2 denoise at the upscaled resolution
    DecodingImage,
    Done
};
