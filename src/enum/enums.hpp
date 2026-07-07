#pragma once
#include <cstdint>

// ─── Inference ────────────────────────────────────────────────────────────────

// Diffusion model architecture. Drives resolution (512 vs 1024), dual text encoder,
// and SDXL-specific UNet inputs (text_embeds, time_ids).
enum class ModelType : std::uint8_t { SD15, SDXL };

// How a hires-fix refinement pass grows the base latent before its second
// denoise. Latent is the v1 route (bilinear latent upscale + VAE decode of the
// larger latent). Pixel/Esrgan are future decode-then-upscale-then-re-encode
// variants; the enum leaves room for them without touching call sites.
enum class UpscaleMode : std::uint8_t { Latent /*, Pixel, Esrgan (future)*/ };

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
