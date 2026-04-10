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
#include <opencv2/opencv.hpp>
#include <thread>
#include <chrono>

namespace sd {

// ── DPM++ 2M Karras denoising loop ───────────────────────────────────────────

static std::vector<float> denoiseSingleLatent(const std::vector<float>& sigmas,
                                              int num_steps,
                                              const std::vector<float>& alphas_cumprod,
                                              GenerationContext& ctx,
                                              std::atomic<int>*  progressStep,
                                              std::atomic<bool>* cancelToken) {
    std::vector<float> x(ctx.latent_size);
    for (auto& v : x) v = randNormal() * sigmas[0];

    std::vector<float> prev_denoised;
    float h_prev = 0.0f;
    auto tDenoise = Clock::now();

    for (int step = 0; step < num_steps; ++step) {
        if (cancelToken && cancelToken->load()) {
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
                 std::atomic<bool>* cancelToken) {
    auto tTotal = Clock::now();

    Logger::info("=== runPipeline ===");
    Logger::info("Working directory: " + std::filesystem::current_path().string());
    Logger::info("Output base: " + outputPath);
    std::filesystem::create_directories(std::filesystem::path(outputPath).parent_path());

    ModelConfig cfg = loadModelConfig(modelDir);
    if (params.width  > 0) cfg.image_w = params.width;
    if (params.height > 0) cfg.image_h = params.height;
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

    // Watcher thread: calls SetTerminate() on the shared RunOptions as soon as
    // cancelToken goes true, aborting any in-progress ORT Run() immediately.
    std::atomic<bool> pipelineDone{false};
    std::thread watcher([cancelToken, &ctx, &pipelineDone]() {
        while (!pipelineDone.load()) {
            if (cancelToken && cancelToken->load()) {
                ctx.run_opts.SetTerminate();
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    });

    try {
        for (int i = 0; i < num_images; ++i) {
            if (cancelToken && cancelToken->load()) break;

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
            auto latent = denoiseSingleLatent(sigmas, num_steps, alphas_cumprod, ctx, progressStep, cancelToken);

            if (cancelToken && cancelToken->load()) break;

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
            cv::imwrite(outPath, img);
            Logger::info("Image saved: " + outPath);
        }
        Logger::info("=== Pipeline complete in " + fmtMs(tTotal) + " ===");
    } catch (const Ort::Exception&) {
        Logger::info("Generation cancelled mid-step (ORT terminated).");
    }

    pipelineDone.store(true);
    watcher.join();
}

} // namespace sd