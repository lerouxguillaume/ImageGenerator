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
#include <cassert>
#include <cmath>
#include <cstring>
#include <functional>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <random>
#include <stop_token>

namespace sd {

namespace {

// Offset added to a per-image seed to source the hires pass-2 noise. Keeps the
// hires draw stream disjoint from the base/batch stream so toggling hires never
// shifts the base pass or later batch images. See the seed contract on the hires
// refinement pass built in runPipeline's per-image loop.
constexpr int64_t kHiresSeedOffset = 0x9E3779B9;  // fixed constant (golden ratio bits)

// ── ScopedLatentResolution ────────────────────────────────────────────────────
// The ONLY sanctioned way to point the context at a latent resolution.
// `ctx` is cache-owned — ModelManager::get() returns a reference into the model
// cache — so any mutation of latent_shape/latent_size MUST be restored, even on
// the exception and cancel paths, or the NEXT generation inherits these dims.
// The destructor guarantees the restore; never set those fields directly.
// In Phase 1 this always sets the native resolution (a no-op re-set); it exists
// so a later hires pass can point ctx at a larger resolution safely.
class ScopedLatentResolution {
public:
    ScopedLatentResolution(GenerationContext& ctx, int latentW, int latentH)
        : ctx_(ctx), savedShape_(ctx.latent_shape), savedSize_(ctx.latent_size) {
        ctx_.latent_shape = {1, 4, latentH, latentW};
        ctx_.latent_size  = 4 * latentW * latentH;
    }
    ~ScopedLatentResolution() {
        ctx_.latent_shape = savedShape_;
        ctx_.latent_size  = savedSize_;
    }
    ScopedLatentResolution(const ScopedLatentResolution&)            = delete;
    ScopedLatentResolution& operator=(const ScopedLatentResolution&) = delete;

private:
    GenerationContext&   ctx_;
    std::vector<int64_t> savedShape_;
    int                  savedSize_;
};

// Per-image pipeline seams. Both lists are empty in Phase 1; later phases push
// a hires refinement pass / image post-processor without touching the control
// flow in generateOneImage(). A refinement pass maps a latent to a latent (at a
// possibly different resolution); a post-processor maps a decoded image.
using RefinementPass     = std::function<Latent(Latent, GenerationContext&)>;
using ImagePostProcessor = std::function<cv::Mat(cv::Mat)>;

} // namespace

// ── DPM++ 2M Karras denoising loop ───────────────────────────────────────────

static Latent denoiseSingleLatent(const std::vector<float>& sigmas,
                                  int num_steps,
                                  const std::vector<float>& alphas_cumprod,
                                  GenerationContext& ctx,
                                  std::atomic<int>*  progressStep,
                                  std::stop_token    stopToken,
                                  int                startStep = 0,
                                  const Latent&      initLatent = {}) {
    std::vector<float> x(ctx.latent_size);
    if (initLatent.empty() || startStep == 0) {
        // txt2img: pure noise at sigmas[0]
        for (int j = 0; j < ctx.latent_size; ++j) x[j] = randNormal() * sigmas[0];
    } else {
        // img2img: init latent + noise at the truncated start sigma
        const float sigma0 = sigmas[startStep];
        for (int j = 0; j < ctx.latent_size; ++j)
            x[j] = initLatent.data[j] + randNormal() * sigma0;
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

        float c_in = 1.0f / std::sqrt(1.0f + sigma * sigma);
        std::vector<float> x_t(ctx.latent_size);
        for (int j = 0; j < ctx.latent_size; ++j) x_t[j] = x[j] * c_in;

        auto eps = runUNetCFG(x_t, sigma, alphas_cumprod, ctx);
        if (std::isnan(eps[0]) || std::isinf(eps[0]))
            throw std::runtime_error("NaN/Inf in UNet output at step "
                                     + std::to_string(step + 1)
                                     + " (sigma=" + std::to_string(sigma)
                                     + ") — likely fp16 overflow; try reducing guidance scale");

        std::vector<float> denoised(ctx.latent_size);
        for (int j = 0; j < ctx.latent_size; ++j)
            denoised[j] = x[j] - sigma * eps[j];

        const float ratio       = sigma_next / sigma;
        const float coeff       = 1.0f - ratio;
        const bool  is_last     = (sigma_next == 0.0f);
        // Avoid log(sigma/0) = +Inf on the final step; h is unused there anyway.
        const float h           = is_last ? 0.0f : std::log(sigma / sigma_next);

        if (prev_denoised.empty() || is_last) {
            for (int j = 0; j < ctx.latent_size; ++j)
                x[j] = ratio * x[j] + coeff * denoised[j];
        } else {
            const float r = h_prev / h;
            for (int j = 0; j < ctx.latent_size; ++j) {
                float D = (1.0f + 1.0f / (2.0f * r)) * denoised[j]
                        - (1.0f / (2.0f * r)) * prev_denoised[j];
                x[j] = ratio * x[j] + coeff * D;
            }
        }

        prev_denoised = denoised;
        h_prev        = h;

        {
            float x_mn = 1e9f, x_mx = -1e9f, x_sum = 0.0f;
            bool  nan_seen = false;
            for (float v : x) {
                if (std::isnan(v) || std::isinf(v)) { nan_seen = true; break; }
                x_mn = std::min(x_mn, v); x_mx = std::max(x_mx, v); x_sum += v;
            }
            if (nan_seen)
                Logger::info("  [NaN/Inf] step " + std::to_string(step + 1)
                             + "  σ=" + std::to_string(sigma));
            else
                Logger::info("  step " + std::to_string(step + 1) + "/" + std::to_string(num_steps)
                             + "  σ=" + std::to_string(sigma)
                             + "  lat[" + std::to_string(x_mn) + ", " + std::to_string(x_mx)
                             + "] mean=" + std::to_string(x_sum / static_cast<float>(ctx.latent_size))
                             + "  " + fmtMs(tStep));
        }
        if (progressStep) progressStep->fetch_add(1);
    }

    Logger::info("Denoising complete in " + fmtMs(tDenoise));
    // Carry the latent's resolution out with its data. Read from ctx.latent_shape
    // (the active source of truth, set by ScopedLatentResolution for this pass).
    return Latent{ std::move(x),
                   static_cast<int>(ctx.latent_shape[3]),
                   static_cast<int>(ctx.latent_shape[2]) };
}

// ── Hires helpers ─────────────────────────────────────────────────────────────

// Snap a target pixel dimension to the nearest multiple of 64, forced strictly
// greater than the native dimension. /64 (not /8) because the SD1.5 latent must
// be divisible by 8: the UNet downsamples the latent 3×, and a non-/8 latent
// makes the skip-connection Concat shapes mismatch. pixels /64 ⇒ latent /8.
static int snapTo64(int target, int native) {
    int snapped = ((target + 32) / 64) * 64;
    if (snapped <= native) snapped = (native / 64 + 1) * 64;
    return snapped;
}

// Bilinear per-channel upscale of a [1,4,h,w] latent to a new latent grid.
// v1 implements UpscaleMode::Latent only; future Pixel/Esrgan modes will decode
// → upscale in pixel space → re-encode instead, and branch here.
static Latent upscaleLatent(const Latent& src, int targetW, int targetH, UpscaleMode mode) {
    (void)mode;  // only UpscaleMode::Latent exists today
    const int srcPlane = src.w * src.h;
    const int dstPlane = targetW * targetH;
    std::vector<float> out(static_cast<size_t>(4) * dstPlane);
    for (int c = 0; c < 4; ++c) {
        cv::Mat plane(src.h, src.w, CV_32F,
                      const_cast<float*>(src.data.data()) + static_cast<size_t>(c) * srcPlane);
        cv::Mat resized;
        cv::resize(plane, resized, {targetW, targetH}, 0, 0, cv::INTER_LINEAR);
        std::memcpy(out.data() + static_cast<size_t>(c) * dstPlane, resized.ptr<float>(),
                    sizeof(float) * static_cast<size_t>(dstPlane));
    }
    return Latent{ std::move(out), targetW, targetH };
}

// ── Per-image generation ──────────────────────────────────────────────────────
// One image of a batch, as an explicit pass sequence:
//   seed → base denoise pass → refinement passes → decode → post-processors → save
// Returns false when the run was cancelled/aborted (caller breaks the batch).
// Ort::Exception is NOT caught here — it propagates to runPipeline's handler so
// cancellation via SetTerminate() is classified there exactly as before.
static bool generateOneImage(GenerationContext& ctx,
                             const std::vector<float>& sigmas,
                             const std::vector<float>& alphas_cumprod,
                             int                num_steps,
                             int                startStep,
                             int64_t            imgSeed,
                             int                nativeLatentW,
                             int                nativeLatentH,
                             const Latent&      initLatent,
                             const std::string& outPath,
                             const std::vector<RefinementPass>&     refinements,
                             const std::vector<ImagePostProcessor>& postProcessors,
                             std::atomic<int>*             progressStep,
                             std::stop_token               stopToken,
                             std::atomic<GenerationStage>* stage) {
    // Seed immediately before the base pass. Nothing between here and the first
    // randNormal() draw inside denoiseSingleLatent consumes the RNG, so the draw
    // sequence (and thus the output) is identical to the pre-refactor loop.
    seedRng(imgSeed);

    // Point ctx at this pass's resolution (native in Phase 1) for the whole
    // UNet + decode sequence; restored on scope exit incl. exception/cancel.
    ScopedLatentResolution res(ctx, nativeLatentW, nativeLatentH);

    // ── base pass ──
    if (stage) stage->store(GenerationStage::Denoising);
    Latent latent = denoiseSingleLatent(sigmas, num_steps, alphas_cumprod, ctx,
                                        progressStep, stopToken, startStep, initLatent);
    if (latent.empty()) return false;             // cancelled/aborted inside denoising
    if (stopToken.stop_requested()) return false;

    {
        float lat_min = 1e9f, lat_max = -1e9f, lat_sum = 0.0f;
        for (float v : latent.data) {
            lat_min = std::min(lat_min, v);
            lat_max = std::max(lat_max, v);
            lat_sum += v;
        }
        Logger::info("Latent stats — min: " + std::to_string(lat_min)
                     + "  max: " + std::to_string(lat_max)
                     + "  mean: " + std::to_string(lat_sum / static_cast<float>(latent.data.size())));
    }

    // ── refinement passes (empty in Phase 1) ──
    for (const auto& pass : refinements) {
        latent = pass(std::move(latent), ctx);
        if (latent.empty() || stopToken.stop_requested()) return false;
    }

    // ── decode ──
    if (stage) stage->store(GenerationStage::DecodingImage);
    cv::Mat img = decodeLatent(latent, ctx);

    // ── image post-processors (empty in Phase 1) ──
    for (const auto& pp : postProcessors) img = pp(std::move(img));

    // ── save ──
    // Normalise separators so cv::imwrite gets a consistent path on Windows.
    std::string normPath = outPath;
    std::replace(normPath.begin(), normPath.end(), '\\', '/');
    std::vector<uchar> encBuf;
    if (cv::imencode(".png", img, encBuf)) {
        std::ofstream ofs(normPath, std::ios::binary);
        ofs.write(reinterpret_cast<const char*>(encBuf.data()),
                  static_cast<std::streamsize>(encBuf.size()));
        Logger::info("Image saved: " + normPath);
    } else {
        Logger::error("cv::imencode failed for: " + normPath);
    }
    return true;
}

// ── Main pipeline ─────────────────────────────────────────────────────────────

void runPipeline(const std::string& prompt,
                 const std::string& neg_prompt,
                 const std::string& outputPath,
                 const GenerationParams& params,
                 const std::string& modelDir,
                 std::atomic<int>*             progressStep,
                 std::atomic<int>*             currentImage,
                 std::stop_token               stopToken,
                 std::atomic<GenerationStage>* stage) {
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

    if (stage) stage->store(GenerationStage::LoadingModel);
    static ModelManager s_modelManager;
    GenerationContext& ctx = s_modelManager.get(cfg, modelDir, params.loras);
    ctx.guidance_scale     = params.guidanceScale;
    ctx.neg_guidance_scale = (params.negativeGuidanceScale > 0.0f)
                               ? params.negativeGuidanceScale
                               : params.guidanceScale;  // 0 → same as guidance_scale (standard CFG)
    ctx.cfg_rescale        = params.cfgRescale;

    if (params.guidanceScale > 12.0f && params.cfgRescale == 0.0f)
        Logger::info("[WARN] guidance_scale=" + std::to_string(params.guidanceScale)
                     + " is high — consider cfg_rescale≈0.7 to reduce oversaturation");

    if (stage) stage->store(GenerationStage::EncodingText);
    auto tEncode = Clock::now();
    // vocab/merges live under models/ next to the executable. Resolve against the
    // executable dir (not cwd) — cwd is not reliably the executable dir on every
    // platform (e.g. a VirtualBox shared folder, where chdir into the share fails).
    const auto modelsDir = resourceDir() / "models";
    ClipTokenizer tokenizer((modelsDir / "vocab.json").string(),
                            (modelsDir / "merges.txt").string());
    if (cfg.type == ModelType::SDXL) {
        ctx.text_embed   = encodeTextSDXL(prompt,     tokenizer, ctx, ctx.embed_shape, ctx.text_embeds_pool);
        ctx.uncond_embed = encodeTextSDXL(neg_prompt, tokenizer, ctx, ctx.embed_shape, ctx.uncond_embeds_pool);
        const float h    = static_cast<float>(cfg.image_h);
        const float w    = static_cast<float>(cfg.image_w);
        ctx.time_ids     = {h, w,
                            static_cast<float>(params.cropTop),
                            static_cast<float>(params.cropLeft),
                            h, w};
    } else {
        ctx.text_embed   = encodeText(prompt,     tokenizer, ctx, ctx.embed_shape);
        ctx.uncond_embed = encodeText(neg_prompt, tokenizer, ctx, ctx.embed_shape);
    }
    Logger::info("Text encoding total: " + fmtMs(tEncode));

    {
        // RMS norm and cosine similarity between cond/uncond embeddings.
        // Near-identical embeddings (sim > 0.99) mean CFG has no effect.
        auto rms = [](const std::vector<float>& v) {
            float s = 0.0f;
            for (float x : v) s += x * x;
            return std::sqrt(s / static_cast<float>(v.size()));
        };
        float dot = 0.0f, na = 0.0f, nb = 0.0f;
        for (size_t i = 0; i < ctx.text_embed.size(); ++i) {
            dot += ctx.text_embed[i] * ctx.uncond_embed[i];
            na  += ctx.text_embed[i] * ctx.text_embed[i];
            nb  += ctx.uncond_embed[i] * ctx.uncond_embed[i];
        }
        const float sim = (na > 0.0f && nb > 0.0f) ? dot / (std::sqrt(na) * std::sqrt(nb)) : 0.0f;
        Logger::info("Embeddings: cond_rms=" + std::to_string(rms(ctx.text_embed))
                     + "  uncond_rms=" + std::to_string(rms(ctx.uncond_embed))
                     + "  cosine_sim=" + std::to_string(sim));
        if (sim > 0.99f) {
            Logger::info("[WARN] cond/uncond embeddings nearly identical"
                         " — CFG will have minimal effect; check prompt and negative prompt");
            // Log first few values at BOS (pos 0) and first content token (pos 1)
            // to determine whether the issue is in input_ids or the ONNX model.
            // embed shape: [1, 77, dim]
            if (!ctx.text_embed.empty() && !ctx.uncond_embed.empty()) {
                const int dim = static_cast<int>(ctx.embed_shape.size() >= 3 ? ctx.embed_shape[2] : 768);
                std::string cond_p0, uncond_p0, cond_p1, uncond_p1;
                for (int d = 0; d < std::min(dim, 4); ++d) {
                    cond_p0   += std::to_string(ctx.text_embed[d])   + " ";
                    uncond_p0 += std::to_string(ctx.uncond_embed[d]) + " ";
                    if (dim + d < (int)ctx.text_embed.size()) {
                        cond_p1   += std::to_string(ctx.text_embed[dim + d])   + " ";
                        uncond_p1 += std::to_string(ctx.uncond_embed[dim + d]) + " ";
                    }
                }
                Logger::info("  embed[pos=0]: cond=[" + cond_p0 + "]  uncond=[" + uncond_p0 + "]");
                Logger::info("  embed[pos=1]: cond=[" + cond_p1 + "]  uncond=[" + uncond_p1 + "]");
            }
        }
    }

    auto alphas_cumprod = buildAlphasCumprod(cfg.T, cfg.beta_start, cfg.beta_end);
    auto sigmas         = buildKarrasSchedule(alphas_cumprod, num_steps);
    Logger::info("Schedule: sigma_max=" + std::to_string(sigmas[0])
                 + "  sigma_min=" + std::to_string(sigmas[num_steps - 1])
                 + "  steps=" + std::to_string(num_steps));

    // img2img: encode the input image once before the per-image loop.
    // sample=false (posterior mean) is deterministic, so the latent is identical
    // across all N images — no need to re-encode per iteration.
    Latent initLatent;
    int startStep = 0;
    if (!params.initImagePath.empty()) {
        if (!ctx.vaeEncoderAvailable) {
            Logger::error("img2img requested but vae_encoder.onnx is not loaded — falling back to txt2img.");
        } else {
            cv::Mat initImg = cv::imread(params.initImagePath, cv::IMREAD_COLOR);
            if (initImg.empty()) {
                Logger::error("img2img: could not read '" + params.initImagePath + "' — falling back to txt2img.");
            } else {
                if (stage) stage->store(GenerationStage::EncodingImage);
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

    // Native latent grid; passed to each image so the base pass runs at native
    // resolution. Post-processor seam stays empty (identity) for now.
    const int nativeLatentW = cfg.image_w / 8;
    const int nativeLatentH = cfg.image_h / 8;
    const std::vector<ImagePostProcessor> postProcessors;  // empty (identity) for now

    // ── Hires-fix setup (SD1.5 only) ──────────────────────────────────────────
    // Constant across the batch: snapped target dims, the pass-2 sigma schedule,
    // and the truncated step bounds. Guarded to SD1.5 — the SDXL UNet is
    // spatially static, so hires there is a separate track (see hires_fix_plan).
    bool hiresOn = params.hires.enabled;
    if (hiresOn && cfg.type != ModelType::SD15) {
        Logger::info("[WARN] Hires fix requested for a non-SD1.5 model — ignoring "
                     "(SDXL UNet has static spatial dims; hires is an SD1.5-only feature).");
        hiresOn = false;
    }
    if (hiresOn && !cfg.hiresCapable) {
        // The UI gates on this, but a preset could still carry hires.enabled onto
        // a pre-hires model whose VAE decoder is static-shape. Refuse rather than
        // crash the decode on the larger latent.
        Logger::info("[WARN] Hires fix requested but this model is not hires-capable "
                     "(static-shape VAE decoder) — ignoring. Re-import to enable hires.");
        hiresOn = false;
    }
    int hiresLatentW = 0, hiresLatentH = 0, hiresEff = 0, hiresStart = 0;
    std::vector<float> hiresSigmas;
    if (hiresOn) {
        const int hiresPxW = snapTo64(static_cast<int>(std::lround(cfg.image_w * params.hires.scale)), cfg.image_w);
        const int hiresPxH = snapTo64(static_cast<int>(std::lround(cfg.image_h * params.hires.scale)), cfg.image_h);
        hiresLatentW = hiresPxW / 8;
        hiresLatentH = hiresPxH / 8;
        hiresEff     = params.hiresEffectiveSteps();
        hiresStart   = params.hiresStartStep();
        hiresSigmas  = buildKarrasSchedule(alphas_cumprod, hiresEff);
        const bool pixelMode = params.hires.mode == UpscaleMode::Pixel && ctx.vaeEncoderAvailable;
        Logger::info("Hires fix: mode=" + std::string(pixelMode ? "pixel" : "latent")
                     + "  scale=" + std::to_string(params.hires.scale)
                     + "  target=" + std::to_string(hiresPxW) + "x" + std::to_string(hiresPxH)
                     + " (latent " + std::to_string(hiresLatentW) + "x" + std::to_string(hiresLatentH) + ")"
                     + "  strength=" + std::to_string(params.hires.strength)
                     + "  pass2 steps=" + std::to_string(hiresEff - hiresStart)
                     + "/" + std::to_string(hiresEff));
    }

    std::exception_ptr denoiseException;
    try {
        for (int i = 0; i < num_images; ++i) {
            if (stopToken.stop_requested()) break;

            if (currentImage) currentImage->store(i + 1);
            if (progressStep) progressStep->store(0);
            // Resolve the seed concretely per image (seed+i for a fixed seed;
            // a fresh random_device draw when unset) so the hires pass can derive
            // its own disjoint stream and random runs stay reproducible from the
            // logged seed. Base output is unchanged: seedRng() applied the same
            // value before, just resolved inside instead of here.
            const int64_t imgSeed = (params.seed >= 0)
                                        ? params.seed + i
                                        : static_cast<int64_t>(std::random_device{}());

            // Per-image refinement list. The hires pass owns its noise stream and
            // resolution (see the seed + ScopedLatentResolution contracts inside).
            std::vector<RefinementPass> refinements;
            if (hiresOn) {
                refinements.push_back(
                    [&hiresSigmas, &alphas_cumprod, hiresEff, hiresStart,
                     hiresLatentW, hiresLatentH, imgSeed, progressStep, stopToken, stage,
                     mode = params.hires.mode](Latent base, GenerationContext& c) -> Latent {
                        if (stage) stage->store(GenerationStage::HiresDenoising);
                        const int hiresPxW = hiresLatentW * 8;
                        const int hiresPxH = hiresLatentH * 8;
                        // Grow the base latent to the snapped hires grid.
                        Latent up;
                        if (mode == UpscaleMode::Pixel && c.vaeEncoderAvailable) {
                            // Pixel route: decode base → sharp bicubic RGB upscale →
                            // re-encode. The re-encoded latent is ON-manifold and sharp,
                            // so pass 2 adds detail. (Latent-space bilinear is off-manifold
                            // and the VAE amplifies it into blur — measurably worse than a
                            // plain bicubic upscale, hence not the default.)
                            // Requires a DYNAMIC-shape VAE encoder. Pre-hires exports have a
                            // static 512 encoder that rejects the upscaled image — catch that
                            // and fall back to latent upscale (uncancellable VAE Run() can't
                            // throw a *cancellation* here, so this only catches shape errors).
                            try {
                                cv::Mat baseImg = decodeLatent(base, c);   // native-res BGR
                                cv::Mat bigImg;
                                cv::resize(baseImg, bigImg, {hiresPxW, hiresPxH}, 0, 0, cv::INTER_CUBIC);
                                up = encodeImage(bigImg, hiresPxW, hiresPxH, c, /*sample=*/false);
                            } catch (const Ort::Exception& e) {
                                Logger::error("Hires pixel mode: VAE encoder rejected "
                                    + std::to_string(hiresPxW) + "x" + std::to_string(hiresPxH)
                                    + " (static-shape encoder — RE-IMPORT this model to enable"
                                    " sharp pixel hires). Falling back to latent upscale (softer). ORT: "
                                    + e.what());
                                up = upscaleLatent(base, hiresLatentW, hiresLatentH, UpscaleMode::Latent);
                            }
                        } else {
                            // Latent route (no VAE encoder, or explicitly selected).
                            up = upscaleLatent(base, hiresLatentW, hiresLatentH, mode);
                        }
                        // Latent dims MUST be /8 (pixel dims were snapped to /64):
                        // the SD1.5 UNet downsamples 3× and the skip-connection
                        // Concat shapes mismatch otherwise.
                        assert(up.w % 8 == 0 && up.h % 8 == 0);
                        // Own noise source: seed = imgSeed + fixed offset, disjoint
                        // from the base/batch RNG stream. The base pass has already
                        // consumed all its draws, and the next image reseeds at its
                        // own start, so this reseed shifts neither.
                        seedRng(imgSeed + kHiresSeedOffset);
                        // Point ctx at the hires resolution for the pass-2 UNet;
                        // restored on scope exit incl. exception/cancel. decode reads
                        // dims from the returned Latent, not ctx, so it follows suit.
                        ScopedLatentResolution hiresRes(c, up.w, up.h);
                        return denoiseSingleLatent(hiresSigmas, hiresEff, alphas_cumprod, c,
                                                   progressStep, stopToken, hiresStart, up);
                    });
            }

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
            const bool produced = generateOneImage(
                ctx, sigmas, alphas_cumprod, num_steps, startStep, imgSeed,
                nativeLatentW, nativeLatentH, initLatent, outPath,
                refinements, postProcessors, progressStep, stopToken, stage);
            if (!produced) break;  // cancelled or aborted
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
