#pragma once
#include <cstdint>

// ─── Inference ────────────────────────────────────────────────────────────────

// Diffusion model architecture. Drives resolution (512 vs 1024), dual text encoder,
// and SDXL-specific UNet inputs (text_embeds, time_ids).
enum class ModelType : std::uint8_t { SD15, SDXL };

// Stage reported by the generation thread to the UI thread via atomic<GenerationStage>.
// Generation cycles: LoadingModel → EncodingText → (EncodingImage) → Denoising
//   → DecodingImage → Done.
enum class GenerationStage : int {
    Idle = 0,
    LoadingModel,
    EncodingText,
    EncodingImage,    // img2img VAE encode only
    Denoising,        // step counter is meaningful while in this stage
    DecodingImage,
    Done
};
