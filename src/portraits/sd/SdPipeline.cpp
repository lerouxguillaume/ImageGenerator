#include "SdPipeline.hpp"
#include "SdUtils.hpp"
#include "SdLoader.hpp"
#include "ModelManager.hpp"
#include "SdScheduler.hpp"
#include "SdTextEncoder.hpp"
#include "SdUNet.hpp"
#include "SdVae.hpp"
#include "../ClipTokenizer.hpp"
#include "../../managers/Logger.hpp"
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <stop_token>

namespace sd {

// ── DPM++ 2M Karras denoising loop ───────────────────────────────────────────

static std::vector<float> denoiseSingleLatent(const std::vector<float>& sigmas,
                                              int num_steps,
                                              const std::vector<float>& alphas_cumprod,
                                              GenerationContext& ctx,
                                              std::atomic<int>*  progressStep,
                                              std::stop_token    stopToken,
                                              int                startStep = 0,
                                              const std::vector<float>& initLatent = {}) {
    std::vector<float> x(ctx.latent_size);
    if (initLatent.empty() || startStep == 0) {
        // txt2img: pure noise at sigmas[0]
        for (int j = 0; j < ctx.latent_size; ++j) x[j] = randNormal() * sigmas[0];
    } else {
        // img2img: init latent + noise at the truncated start sigma
        const float sigma0 = sigmas[startStep];
        for (int j = 0; j < ctx.latent_size; ++j)
            x[j] = initLatent[j] + randNormal() * sigma0;
    }

    std::vector<float> prev_denoised;
    float h_prev = 0.0f;
    auto tDenoise = Clock::now();

    for (int step = startStep; step < num_steps; ++step) {
        if (stopToken.stop_requested()) {
            Logger::info("Denoising cancelled at step " + std::to_string(step));
            return {};
        }
        float sigma      = sigmas[step];
        float sigma_next = sigmas[step + 1];
        auto  tStep      = Clock::now();
        Logger::info("DPM++ step " + std::to_string(step + 1) + "/" + std::to_string(num_steps)
                     + "  sigma=" + std::to_string(sigma));

        float c_in = 1.0f / std::sqrt(1.0f + sigma * sigma);
        std::vector<float> x_t(ctx.latent_size);
        for (int j = 0; j < ctx.latent_size; ++j) x_t[j] = x[j] * c_in;

        auto eps = runUNetCFG(x_t, sigma, alphas_cumprod, ctx);

        std::vector<float> denoised(ctx.latent_size);
        for (int j = 0; j < ctx.latent_size; ++j)
            denoised[j] = x[j] - sigma * eps[j];

        float ratio = sigma_next / sigma;
        float h     = std::log(sigma / sigma_next);
        float coeff = 1.0f - ratio;

        if (prev_denoised.empty() || sigma_next == 0.0f) {
            for (int j = 0; j < ctx.latent_size; ++j)
                x[j] = ratio * x[j] + coeff * denoised[j];
        } else {
            float r = h_prev / h;
            for (int j = 0; j < ctx.latent_size; ++j) {
                float D = (1.0f + 1.0f / (2.0f * r)) * denoised[j]
                        - (1.0f / (2.0f * r)) * prev_denoised[j];
                x[j] = ratio * x[j] + coeff * D;
            }
        }

        prev_denoised = denoised;
        h_prev        = h;

        if ((step + 1) % 5 == 0 || step == 0 || step + 1 == num_steps) {
            float x_mn = 1e9f, x_mx = -1e9f, x_sum = 0.0f;
            for (float v : x) { x_mn = std::min(x_mn, v); x_mx = std::max(x_mx, v); x_sum += v; }
            Logger::info("  latent stats: min=" + std::to_string(x_mn)
                         + "  max=" + std::to_string(x_mx)
                         + "  mean=" + std::to_string(x_sum / static_cast<float>(ctx.latent_size)));
        }
        Logger::info("  step done in " + fmtMs(tStep));
        if (progressStep) progressStep->fetch_add(1);
    }

    Logger::info("Denoising complete in " + fmtMs(tDenoise));
    return x;
}

// ── Main pipeline ─────────────────────────────────────────────────────────────

void runPipeline(const std::string& prompt,
                 const std::string& neg_prompt,
                 const std::string& outputPath,
                 const GenerationParams& params,
                 const std::string& modelDir,
                 std::atomic<int>*  progressStep,
                 std::atomic<int>*  currentImage,
                 std::stop_token    stopToken) {
    auto tTotal = Clock::now();

    Logger::info("=== runPipeline ===");
    Logger::info("Working directory: " + std::filesystem::current_path().string());
    Logger::info("Output base: " + outputPath);
    std::filesystem::create_directories(std::filesystem::path(outputPath).parent_path());

    ModelConfig cfg = loadModelConfig(modelDir);
    const int native_w = cfg.image_w;
    const int native_h = cfg.image_h;
    if (params.width  > 0) cfg.image_w = params.width;
    if (params.height > 0) cfg.image_h = params.height;
    // UNet spatial dimensions are static (batch-only dynamic axes). If the
    // requested resolution produces a different latent size than the model was
    // exported at, revert and warn — the UNet will reject the wrong shape.
    if (cfg.image_w / 8 != native_w / 8 || cfg.image_h / 8 != native_h / 8) {
        Logger::info("[WARN] Requested resolution " + std::to_string(cfg.image_w) + "x" + std::to_string(cfg.image_h)
                     + " incompatible with model native " + std::to_string(native_w) + "x" + std::to_string(native_h)
                     + " (UNet has static spatial dims) — using native resolution.");
        cfg.image_w = native_w;
        cfg.image_h = native_h;
    }
    const int num_steps  = params.numSteps;
    const int num_images = params.numImages;

    Logger::info("Steps: " + std::to_string(num_steps)
                 + "  guidance: " + std::to_string(params.guidanceScale)
                 + "  images: " + std::to_string(num_images)
                 + "  latent: " + std::to_string(cfg.image_w / 8)
                 + "x" + std::to_string(cfg.image_h / 8));
    Logger::info("Prompt: " + prompt);
    Logger::info("Neg:    " + neg_prompt);

    static ModelManager s_modelManager;
    GenerationContext& ctx = s_modelManager.get(cfg, modelDir, params.loras);
    ctx.guidance_scale     = params.guidanceScale;
    ctx.neg_guidance_scale = (params.negativeGuidanceScale > 0.0f)
                               ? params.negativeGuidanceScale
                               : params.guidanceScale;  // 0 → same as guidance_scale (standard CFG)
    ctx.cfg_rescale        = params.cfgRescale;

    auto tEncode = Clock::now();
    // vocab/merges live in the base model dir (parent of the specific model subdir).
    const std::string baseDir = std::filesystem::path(modelDir).parent_path().string();
    ClipTokenizer tokenizer(baseDir + "/vocab.json", baseDir + "/merges.txt");
    if (cfg.type == ModelType::SDXL) {
        ctx.text_embed   = encodeTextSDXL(prompt,     tokenizer, ctx, ctx.embed_shape, ctx.text_embeds_pool);
        ctx.uncond_embed = encodeTextSDXL(neg_prompt, tokenizer, ctx, ctx.embed_shape, ctx.uncond_embeds_pool);
        const float h    = static_cast<float>(cfg.image_h);
        const float w    = static_cast<float>(cfg.image_w);
        ctx.time_ids     = {h, w, 0.0f, 0.0f, h, w};
    } else {
        ctx.text_embed   = encodeText(prompt,     tokenizer, ctx, ctx.embed_shape);
        ctx.uncond_embed = encodeText(neg_prompt, tokenizer, ctx, ctx.embed_shape);
    }
    Logger::info("Text encoding total: " + fmtMs(tEncode));

    auto alphas_cumprod = buildAlphasCumprod(cfg.T, cfg.beta_start, cfg.beta_end);
    auto sigmas         = buildKarrasSchedule(alphas_cumprod, num_steps);

    // img2img: encode the input image once before the per-image loop.
    // sample=false (posterior mean) is deterministic, so the latent is identical
    // across all N images — no need to re-encode per iteration.
    std::vector<float> initLatent;
    int startStep = 0;
    if (!params.initImagePath.empty()) {
        if (!ctx.vaeEncoderAvailable) {
            Logger::error("img2img requested but vae_encoder.onnx is not loaded — falling back to txt2img.");
        } else {
            cv::Mat initImg = cv::imread(params.initImagePath, cv::IMREAD_COLOR);
            if (initImg.empty()) {
                Logger::error("img2img: could not read '" + params.initImagePath + "' — falling back to txt2img.");
            } else {
                const float clampedStrength = std::max(0.0f, std::min(params.strength, 1.0f));
                startStep  = static_cast<int>((1.0f - clampedStrength) * static_cast<float>(num_steps));
                startStep  = std::max(0, std::min(startStep, num_steps - 1));
                initLatent = encodeImage(initImg, cfg.image_w, cfg.image_h, ctx, /*sample=*/false);
                Logger::info("img2img: strength=" + std::to_string(clampedStrength)
                             + "  startStep=" + std::to_string(startStep)
                             + "/" + std::to_string(num_steps));
            }
        }
    }

    // When request_stop() is called on the owning jthread, this callback fires
    // immediately (no polling delay) and aborts any in-progress ORT Run().
    std::stop_callback stopCallback(stopToken, [&ctx]() {
        ctx.run_opts.SetTerminate();
    });

    std::exception_ptr denoiseException;
    try {
        for (int i = 0; i < num_images; ++i) {
            if (stopToken.stop_requested()) break;

            if (currentImage) currentImage->store(i + 1);
            if (progressStep) progressStep->store(0);
            // seed < 0 → random; for multi-image runs use seed+i so each image differs
            const int64_t imgSeed = (params.seed >= 0) ? params.seed + i : -1;
            seedRng(imgSeed);

            // Insert _N before the extension for multi-image runs.
            std::string outPath = outputPath;
            if (num_images > 1) {
                const auto dot     = outputPath.rfind('.');
                const std::string idx = std::to_string(i + 1);
                outPath = (dot == std::string::npos)
                    ? outputPath + "_" + idx
                    : outputPath.substr(0, dot) + "_" + idx + outputPath.substr(dot);
            }

            Logger::info("--- Image " + std::to_string(i + 1) + "/" + std::to_string(num_images) + " ---");
            auto latent = denoiseSingleLatent(sigmas, num_steps, alphas_cumprod, ctx,
                                              progressStep, stopToken, startStep, initLatent);

            if (stopToken.stop_requested()) break;

            float lat_min = 1e9f, lat_max = -1e9f, lat_sum = 0.0f;
            for (float v : latent) {
                lat_min = std::min(lat_min, v);
                lat_max = std::max(lat_max, v);
                lat_sum += v;
            }
            Logger::info("Latent stats — min: " + std::to_string(lat_min)
                         + "  max: " + std::to_string(lat_max)
                         + "  mean: " + std::to_string(lat_sum / static_cast<float>(latent.size())));

            auto img = decodeLatent(latent, ctx);
            // Normalise separators so cv::imwrite gets a consistent path on Windows.
            std::string normPath = outPath;
            std::replace(normPath.begin(), normPath.end(), '\\', '/');
            std::vector<uchar> encBuf;
            bool encOk = cv::imencode(".png", img, encBuf);
            if (encOk) {
                std::ofstream ofs(normPath, std::ios::binary);
                ofs.write(reinterpret_cast<const char*>(encBuf.data()),
                          static_cast<std::streamsize>(encBuf.size()));
                Logger::info("Image saved: " + normPath);
            } else {
                Logger::error("cv::imencode failed for: " + normPath);
            }
        }
        Logger::info("=== Pipeline complete in " + fmtMs(tTotal) + " ===");
    } catch (const Ort::Exception&) {
        if (stopToken.stop_requested()) {
            Logger::info("Generation cancelled mid-step (ORT terminated).");
        } else {
            Logger::error("ORT exception during inference (not a cancellation).");
            denoiseException = std::current_exception();
        }
    }

    if (denoiseException)
        std::rethrow_exception(denoiseException);
}

} // namespace sd
