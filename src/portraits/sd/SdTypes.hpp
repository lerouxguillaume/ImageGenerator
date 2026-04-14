#pragma once
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <string>
#include <vector>
#include "../../enum/enums.hpp"

namespace sd {

// ── Model configuration ───────────────────────────────────────────────────────
// Loaded once from model.json and used to drive resolution, encoder choice, etc.
// Add a new ModelType value and a branch in SdLoader::loadModelConfig() to
// support a new model family; everything else adapts through this struct.

struct ModelConfig {
    ModelType type       = ModelType::SD15;
    int       image_w    = 512;
    int       image_h    = 512;
    int       T          = 1000;
    float     beta_start = 0.00085f;
    float     beta_end   = 0.012f;
};

// ── Runtime inference context ─────────────────────────────────────────────────
// Owns all ONNX Runtime sessions and shared state for one pipeline run.
// SDXL-only fields (te2_*, text_embeds_pool, time_ids) are left at default
// for SD 1.5 and simply unused.

struct GenerationContext {
    Ort::Env              env;
    Ort::SessionOptions   session_opts;      // primary EP (GPU for UNet)
    Ort::SessionOptions   vae_session_opts;  // reserved; currently same as cpu for DML
    Ort::SessionOptions   cpu_session_opts;  // CPU-only (text encoders, fallback UNet)
    Ort::Session          text_encoder;
    Ort::Session          text_encoder_2;    // SDXL only: OpenCLIP-G
    Ort::Session          unet;
    Ort::Session          cpu_unet;          // GPU-fallback copy of unet
    Ort::Session          vae_decoder;
    Ort::MemoryInfo       memory_info;
    Ort::AllocatorWithDefaultOptions allocator;

    bool dmlFailed            = false;
    bool cpuFallbackAvailable = false;  // true only if cpu_unet was loaded successfully
    bool unetExpectsFp32      = false;  // queried at load time
    bool vaeExpectsFp32       = false;  // queried at load time

    ModelType model_type = ModelType::SD15;

    // I/O names — populated by loadModels(), used by the encoder/UNet/VAE passes.
    std::string te_input,  te_output;                    // text encoder 1
    std::string te2_input, te2_output, te2_pooled;       // text encoder 2 (SDXL)
    std::string unet_in0, unet_in1, unet_in2,            // latent, timestep, embed
                unet_in3, unet_in4,                      // SDXL: text_embeds, time_ids
                unet_out0;
    std::string vae_in, vae_out;

    // Computed embeddings — filled by the text-encoding stage of runPipeline().
    std::vector<float>   text_embed,   uncond_embed;
    std::vector<int64_t> embed_shape;                    // [1, seq_len, dim]
    std::vector<float>   text_embeds_pool, uncond_embeds_pool; // SDXL pooled (1,1280)
    std::vector<float>   time_ids;                       // SDXL time conditioning (6)

    std::vector<int64_t> latent_shape;   // [1, 4, H/8, W/8]
    int                  latent_size = 0;

    float           guidance_scale     = 8.0f;
    float           neg_guidance_scale = 8.0f; // independent negative-prompt CFG scale
    float           cfg_rescale        = 0.0f; // 0 = off; 0.7 typical for rescaling
    Ort::RunOptions run_opts;                   // shared across UNet steps; can be terminated

    GenerationContext()
        : env(ORT_LOGGING_LEVEL_WARNING, "LocalAI")
        , text_encoder(nullptr)
        , text_encoder_2(nullptr)
        , unet(nullptr)
        , cpu_unet(nullptr)
        , vae_decoder(nullptr)
#if defined(USE_CUDA)
        // Pinned (page-locked) memory avoids an extra copy on the H2D transfer path.
        , memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPUInput))
#else
        , memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
#endif
    {}
};

// ── Loaded model instance ─────────────────────────────────────────────────────
// Bundles everything produced by one loadModels() call: the live ORT sessions
// (via ctx) and the model schema that drove the load.
//
// Immutable after creation: sessions are not hot-swappable.
// Both the no-LoRA and LoRA paths load sessions from the .onnx file path;
// ORT resolves .onnx.data natively.  No byte buffers are held here.

struct ModelInstance {
    GenerationContext ctx;
    ModelConfig       config;
};

} // namespace sd