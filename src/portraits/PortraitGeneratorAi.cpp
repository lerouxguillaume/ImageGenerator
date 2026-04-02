#include "PortraitGeneratorAi.hpp"
#include "PromptBuilder.hpp"
#include "ClipTokenizer.hpp"
#include <onnxruntime_cxx_api.h>
#ifdef USE_DML
// Forward-declare the DML entry point to avoid pulling in DirectML.h (Windows
// SDK header not available during Linux cross-compilation).
extern "C" OrtStatus *OrtSessionOptionsAppendExecutionProvider_DML(
    OrtSessionOptions *options, int device_id);
#endif
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <random>
#include <filesystem>
#include <chrono>
#include <thread>

#include "../managers/Logger.hpp"

namespace {
    // ── Timing helper ─────────────────────────────────────────────────────────────

    using Clock = std::chrono::steady_clock;

    std::string fmtMs(Clock::time_point start) {
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start).count();
        return std::to_string(ms) + " ms";
    }

    // ── Type conversion helpers ───────────────────────────────────────────────────

    std::vector<Ort::Float16_t> toFp16(const std::vector<float> &src) {
        std::vector<Ort::Float16_t> dst(src.size());
        for (size_t i = 0; i < src.size(); ++i)
            dst[i] = Ort::Float16_t(src[i]);
        return dst;
    }

    // Box-Muller transform — diffusion models require N(0,1) latent initialization.
    float randNormal() {
        const float u1 = (static_cast<float>(rand()) + 1.0f) / (static_cast<float>(RAND_MAX) + 2.0f);
        const float u2 = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX) + 1.0f);
        return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * static_cast<float>(M_PI) * u2);
    }

    void seedRng() {
        std::random_device rd;
        unsigned int seed = rd();
        srand(seed);
        Logger::info("RNG seed: " + std::to_string(seed));
    }

    // Convert a CHW float32 buffer to an OpenCV BGR Mat.
    // SD VAE decoder outputs values nominally in [-1, 1] (RGB channel order).
    // Maps each value as: pixel = clamp((val + 1) / 2, 0, 1) * 255.
    cv::Mat latentToImage(const float *img_data, int img_w, int img_h) {
        const int plane = img_w * img_h;

        for (int c = 0; c < 3; ++c) {
            float mn = 1e9f, mx = -1e9f;
            for (int j = 0; j < plane; ++j) {
                float v = img_data[c * plane + j];
                mn = std::min(mn, v);
                mx = std::max(mx, v);
            }
            Logger::info("VAE ch[" + std::to_string(c) + "] range: [" +
                         std::to_string(mn) + ", " + std::to_string(mx) + "]");
        }

        cv::Mat img(img_h, img_w, CV_8UC3);
        for (int y = 0; y < img_h; ++y) {
            for (int x = 0; x < img_w; ++x) {
                for (int c = 0; c < 3; ++c) {
                    float val = img_data[c * plane + y * img_w + x];
                    val = std::min(std::max((val + 1.0f) / 2.0f, 0.0f), 1.0f) * 255.0f;
                    img.at<cv::Vec3b>(y, x)[c] = static_cast<uint8_t>(val);
                }
            }
        }
        // VAE outputs RGB; cvtColor converts to OpenCV's BGR for correct imwrite output.
        cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
        return img;
    }

    // ── ONNX inference context ────────────────────────────────────────────────────

    struct GenerationContext {
        Ort::Env env;
        Ort::SessionOptions session_opts;     // GPU EP (unet)
        Ort::SessionOptions vae_session_opts; // GPU EP (vae) — basic optimization to avoid DML Reshape issues
        Ort::SessionOptions cpu_session_opts; // CPU only (text encoder + fallback)
        Ort::Session text_encoder;
        Ort::Session unet;
        Ort::Session cpu_unet; // fallback if GPU unet fails
        Ort::Session vae_decoder;
        Ort::MemoryInfo memory_info;
        Ort::AllocatorWithDefaultOptions allocator;
        bool dmlFailed = false;

        std::string te_input, te_output;
        std::string unet_in0, unet_in1, unet_in2, unet_out0;
        std::string vae_in, vae_out;

        std::vector<float> text_embed;
        std::vector<float> uncond_embed;
        std::vector<int64_t> embed_shape; // [1, seq_len, embed_dim]

        std::vector<int64_t> latent_shape; // [1, 4, H, W]
        int latent_size = 0;

        float guidance_scale = 8.0f;
        Ort::RunOptions run_opts;

        GenerationContext()
            : env(ORT_LOGGING_LEVEL_WARNING, "LocalAI")
              , text_encoder(nullptr)
              , unet(nullptr)
              , cpu_unet(nullptr)
              , vae_decoder(nullptr)
#if defined(USE_CUDA)
              // OrtMemTypeCPUInput = pinned (page-locked) host memory; avoids an extra
              // copy on the H2D transfer path when the CUDA EP moves tensors to the GPU.
        , memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPUInput))
#else
              // DML (and CPU) expect regular heap memory for tensor inputs.
              , memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
#endif
        {
        }
    };

    // ── Model I/O logging ─────────────────────────────────────────────────────────

    const char *kOrtTypeNames[] = {
        "undefined", "float", "uint8", "int8", "uint16", "int16", "int32", "int64",
        "string", "bool", "float16", "double", "uint32", "uint64", "complex64", "complex128", "bfloat16"
    };

    void logModelIO(const char *label, Ort::Session &session,
                    Ort::AllocatorWithDefaultOptions &allocator) {
        for (size_t k = 0; k < session.GetInputCount(); ++k) {
            auto info = session.GetInputTypeInfo(k);
            const auto t = info.GetTensorTypeAndShapeInfo().GetElementType();
            std::string tname = (t < 17) ? kOrtTypeNames[t] : std::to_string(t);
            Logger::info(std::string(label) + " input[" + std::to_string(k) + "]: "
                         + session.GetInputNameAllocated(k, allocator).get() + " (" + tname + ")");
        }
        for (size_t k = 0; k < session.GetOutputCount(); ++k) {
            auto info = session.GetOutputTypeInfo(k);
            const auto t = info.GetTensorTypeAndShapeInfo().GetElementType();
            std::string tname = (t < 17) ? kOrtTypeNames[t] : std::to_string(t);
            Logger::info(std::string(label) + " output[" + std::to_string(k) + "]: "
                         + session.GetOutputNameAllocated(k, allocator).get() + " (" + tname + ")");
        }
    }

    // ── Model loading ─────────────────────────────────────────────────────────────

    GenerationContext loadModels(int latent_w, int latent_h, const std::string& modelDir) {
        auto t0 = Clock::now();
        GenerationContext ctx;
        const int numThreads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
        Logger::info("=== loadModels ===");
        Logger::info("Model dir: " + modelDir);
        Logger::info("CPU threads: " + std::to_string(numThreads));
        ctx.session_opts.SetIntraOpNumThreads(numThreads);
        ctx.cpu_session_opts.SetIntraOpNumThreads(numThreads);

#if defined(USE_DML)
        try {
            // DML requires these two flags — without them it raises E_INVALIDARG on
            // reshape nodes because it cannot handle dynamic memory reuse patterns.
            ctx.session_opts.DisableMemPattern();
            ctx.session_opts.SetExecutionMode(ORT_SEQUENTIAL);
            Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(ctx.session_opts, 0));

            // VAE decoder runs on CPU: DirectML rejects node_view_2 (a Reshape in the
            // VAE attention) at both load and inference time regardless of optimization
            // level or dtype. The VAE runs once per image (not per step) so the CPU
            // overhead is small relative to the UNet's 20 GPU steps.
            Logger::info("EP: DirectML (unet=GPU, vae=CPU, text_encoder=CPU)");
        } catch (const Ort::Exception &e) {
            Logger::info(std::string("DirectML EP unavailable, falling back to CPU: ") + e.what());
        }
#elif defined(USE_CUDA)
        try {
            OrtCUDAProviderOptions cuda_options;
            ctx.session_opts.AppendExecutionProvider_CUDA(cuda_options);
            Logger::info("EP: CUDA");
        } catch (const Ort::Exception &e) {
            Logger::info(std::string("CUDA EP unavailable, falling back to CPU: ") + e.what());
        }
#else
        Logger::info("EP: CPU");
#endif

        Logger::info("Checking model files...");
        for (const std::string p: {
                 modelDir + "/text_encoder.onnx", modelDir + "/unet.onnx", modelDir + "/vae_decoder.onnx",
                 std::string("models/vocab.json"), std::string("models/merges.txt")
             }) {
            if (std::filesystem::exists(p))
                Logger::info("  [OK] " + p);
            else
                Logger::info("  [MISSING] " + p);
        }

        auto loadSession = [&](const char *label, auto path, Ort::SessionOptions &opts) {
            Logger::info("Loading " + std::string(label) + "...");
            auto ts = Clock::now();
            Ort::Session s(ctx.env, path, opts);
            Logger::info("  " + std::string(label) + " loaded in " + fmtMs(ts));
            return s;
        };

#ifdef _WIN32
        // ORT on Windows requires wchar_t paths. Model dir names are ASCII, so a
        // simple char-by-char widening is sufficient.
        auto toWide = [](const std::string& s) { return std::wstring(s.begin(), s.end()); };
        const std::wstring wModelDir = toWide(modelDir);
        ctx.text_encoder = loadSession("text_encoder", (wModelDir + L"/text_encoder.onnx").c_str(), ctx.cpu_session_opts);
        ctx.unet         = loadSession("unet",         (wModelDir + L"/unet.onnx").c_str(),         ctx.session_opts);
        ctx.cpu_unet     = loadSession("cpu_unet",     (wModelDir + L"/unet.onnx").c_str(),         ctx.cpu_session_opts);
        ctx.vae_decoder  = loadSession("vae_decoder",  (wModelDir + L"/vae_decoder.onnx").c_str(),  ctx.cpu_session_opts);
#else
        ctx.text_encoder = loadSession("text_encoder", (modelDir + "/text_encoder.onnx").c_str(), ctx.cpu_session_opts);
        ctx.unet         = loadSession("unet",         (modelDir + "/unet.onnx").c_str(),         ctx.session_opts);
        ctx.cpu_unet     = loadSession("cpu_unet",     (modelDir + "/unet.onnx").c_str(),         ctx.cpu_session_opts);
        ctx.vae_decoder  = loadSession("vae_decoder",  (modelDir + "/vae_decoder.onnx").c_str(),  ctx.cpu_session_opts);
#endif

        logModelIO("text_encoder", ctx.text_encoder, ctx.allocator);
        logModelIO("unet", ctx.unet, ctx.allocator);
        logModelIO("vae_decoder", ctx.vae_decoder, ctx.allocator);

        ctx.te_input = ctx.text_encoder.GetInputNameAllocated(0, ctx.allocator).get();
        ctx.te_output = ctx.text_encoder.GetOutputNameAllocated(0, ctx.allocator).get();
        ctx.unet_in0 = ctx.unet.GetInputNameAllocated(0, ctx.allocator).get();
        ctx.unet_in1 = ctx.unet.GetInputNameAllocated(1, ctx.allocator).get();
        ctx.unet_in2 = ctx.unet.GetInputNameAllocated(2, ctx.allocator).get();
        ctx.unet_out0 = ctx.unet.GetOutputNameAllocated(0, ctx.allocator).get();
        ctx.vae_in = ctx.vae_decoder.GetInputNameAllocated(0, ctx.allocator).get();
        ctx.vae_out = ctx.vae_decoder.GetOutputNameAllocated(0, ctx.allocator).get();

        ctx.latent_shape = {1, 4, latent_h, latent_w};
        ctx.latent_size = 1 * 4 * latent_h * latent_w;

        Logger::info("All models loaded in " + fmtMs(t0));
        Logger::info("=================");
        return ctx;
    }

    // ── Text encoding ─────────────────────────────────────────────────────────────

    std::vector<float> encodeText(const std::string &prompt,
                                  ClipTokenizer &tokenizer,
                                  GenerationContext &ctx,
                                  std::vector<int64_t> &out_shape) {
        auto token_ids = tokenizer.encode(prompt);
        Logger::info("encodeText: " + std::to_string(token_ids.size()) + " tokens"
                     + "  prompt=\"" + prompt.substr(0, 80) + (prompt.size() > 80 ? "..." : "") + "\"");

        auto tEnc = Clock::now();
        std::vector<int64_t> token_shape = {1, 77};
        Ort::Value token_tensor = Ort::Value::CreateTensor<int64_t>(
            ctx.memory_info, token_ids.data(), token_ids.size(),
            token_shape.data(), token_shape.size());

        const char *in_names[] = {ctx.te_input.c_str()};
        const char *out_names[] = {ctx.te_output.c_str()};
        auto te_out = ctx.text_encoder.Run(Ort::RunOptions{nullptr},
                                           in_names, &token_tensor, 1,
                                           out_names, 1);
        Logger::info("  text encoder run: " + fmtMs(tEnc));

        auto shape_info = te_out.front().GetTensorTypeAndShapeInfo();
        auto shape = shape_info.GetShape();
        size_t elem_count = shape_info.GetElementCount();

        std::string shape_str = "[";
        for (size_t k = 0; k < shape.size(); ++k)
            shape_str += std::to_string(shape[k]) + (k + 1 < shape.size() ? ", " : "]");
        Logger::info("  embedding shape: " + shape_str);

        const auto *raw = te_out.front().GetTensorData<Ort::Float16_t>();
        std::vector<float> embedding(elem_count);
        for (size_t k = 0; k < elem_count; ++k)
            embedding[k] = static_cast<float>(raw[k]);

        out_shape = std::vector<int64_t>(shape.begin(), shape.end());
        return embedding;
    }

    // ── Noise schedule ────────────────────────────────────────────────────────────

    // DDPM scaled_linear alpha_bar schedule matching the diffusers SD v1.5 default.
    std::vector<float> buildAlphasCumprod(int T, float beta_start, float beta_end) {
        float sqrt_start = std::sqrt(beta_start);
        float sqrt_end = std::sqrt(beta_end);
        std::vector<float> alphas(T);
        float cum = 1.0f;
        for (int ti = 0; ti < T; ++ti) {
            float t_frac = static_cast<float>(ti) / static_cast<float>(T - 1);
            float beta = std::pow(sqrt_start + t_frac * (sqrt_end - sqrt_start), 2.0f);
            cum *= (1.0f - beta);
            alphas[ti] = cum;
        }
        return alphas;
    }

    // DPM++ 2M Karras sigma schedule. Returns num_steps+1 values; last entry is 0.
    std::vector<float> buildKarrasSchedule(const std::vector<float> &alphas_cumprod, int num_steps) {
        auto alphaBarToSigma = [](float ab) { return std::sqrt((1.0f - ab) / ab); };

        const float rho = 7.0f;
        int T = static_cast<int>(alphas_cumprod.size());
        float sigma_max = alphaBarToSigma(alphas_cumprod[T - 1]);
        float sigma_min = alphaBarToSigma(alphas_cumprod[0]);
        float inv_rho_max = std::pow(sigma_max, 1.0f / rho);
        float inv_rho_min = std::pow(sigma_min, 1.0f / rho);

        std::vector<float> sigmas(num_steps + 1);
        for (int s = 0; s < num_steps; ++s) {
            float ramp = static_cast<float>(s) / static_cast<float>(num_steps - 1);
            sigmas[s] = std::pow(inv_rho_max + ramp * (inv_rho_min - inv_rho_max), rho);
        }
        sigmas[num_steps] = 0.0f;
        return sigmas;
    }

    // Map a DPM++ sigma to the nearest DDPM integer timestep.
    int sigmaToTimestep(float sigma, const std::vector<float> &alphas_cumprod) {
        float ab = 1.0f / (1.0f + sigma * sigma);
        int T = static_cast<int>(alphas_cumprod.size());
        int lo = 0, hi = T - 1;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (alphas_cumprod[mid] < ab) hi = mid;
            else lo = mid + 1;
        }
        return lo;
    }

    // ── UNet CFG pass ─────────────────────────────────────────────────────────────

    // Run a single batch=1 UNet pass and return the raw eps output.
    static std::vector<float> runUNetSingle(
        const std::vector<float> &x_t,
        int t,
        const std::vector<float> &embed,
        GenerationContext &ctx) {
        std::vector<float> latent_fp32 = x_t;
        std::vector<float> embed_fp32 = embed;
        std::vector<float> ts_fp32 = {static_cast<float>(t)};

        std::vector<int64_t> latent_shape = {1, ctx.latent_shape[1], ctx.latent_shape[2], ctx.latent_shape[3]};
        std::vector<int64_t> ts_shape = {1};
        std::vector<int64_t> embed_shape = {1, ctx.embed_shape[1], ctx.embed_shape[2]};

        const char *in_names[] = {ctx.unet_in0.c_str(), ctx.unet_in1.c_str(), ctx.unet_in2.c_str()};
        const char *out_names[] = {ctx.unet_out0.c_str()};

        // CPU fallback: unet.onnx expects float16 inputs regardless of EP.
        // Convert to FP16 and handle FP16 or FP32 output.
        auto runOnCpu = [&]() {
            auto latent_fp16_cpu = toFp16(latent_fp32);
            auto ts_fp16_cpu = toFp16(ts_fp32);
            auto embed_fp16_cpu = toFp16(embed_fp32);
            std::vector<Ort::Value> inputs;
            inputs.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(ctx.memory_info, latent_fp16_cpu.data(),
                                                                      latent_fp16_cpu.size(), latent_shape.data(),
                                                                      latent_shape.size()));
            inputs.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(ctx.memory_info, ts_fp16_cpu.data(),
                                                                      ts_fp16_cpu.size(), ts_shape.data(),
                                                                      ts_shape.size()));
            inputs.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(ctx.memory_info, embed_fp16_cpu.data(),
                                                                      embed_fp16_cpu.size(), embed_shape.data(),
                                                                      embed_shape.size()));
            auto output = ctx.cpu_unet.Run(ctx.run_opts, in_names, inputs.data(), inputs.size(), out_names, 1);
            auto &out_tensor = output.front();
            auto elem_type = out_tensor.GetTensorTypeAndShapeInfo().GetElementType();
            std::vector<float> result(ctx.latent_size);
            if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
                const auto *data = out_tensor.GetTensorData<Ort::Float16_t>();
                for (int j = 0; j < ctx.latent_size; ++j)
                    result[j] = static_cast<float>(data[j]);
            } else {
                const auto *data = out_tensor.GetTensorData<float>();
                std::copy(data, data + ctx.latent_size, result.begin());
            }
            return result;
        };

        if (ctx.dmlFailed)
            return runOnCpu();

        // GPU path: the exported ONNX UNet expects float16 for all three inputs.
        // Keep latent/denoising math in float32; convert to FP16 only at the GPU boundary.
        auto latent_fp16 = toFp16(latent_fp32);
        auto ts_fp16 = toFp16(ts_fp32);
        auto embed_fp16 = toFp16(embed_fp32);

        // Sanity check: FP32 values that overflow FP16 become inf and will corrupt DML silently.
        {
            bool hasInf = false;
            for (auto v: latent_fp16) if (!std::isfinite(static_cast<float>(v))) {
                hasInf = true;
                break;
            }
            if (!hasInf)
                for (auto v: embed_fp16) if (!std::isfinite(static_cast<float>(v))) {
                    hasInf = true;
                    break;
                }
            if (hasInf) Logger::info("WARNING: FP16 overflow detected before GPU UNet run");
        }

        std::vector<Ort::Value> gpu_inputs;
        gpu_inputs.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(ctx.memory_info, latent_fp16.data(),
                                                                      latent_fp16.size(), latent_shape.data(),
                                                                      latent_shape.size()));
        gpu_inputs.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(ctx.memory_info, ts_fp16.data(), ts_fp16.size(),
                                                                      ts_shape.data(), ts_shape.size()));
        gpu_inputs.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(ctx.memory_info, embed_fp16.data(),
                                                                      embed_fp16.size(), embed_shape.data(),
                                                                      embed_shape.size()));

        try {
            auto output = ctx.unet.Run(ctx.run_opts, in_names, gpu_inputs.data(), gpu_inputs.size(), out_names, 1);
            auto &out_tensor = output.front();
            auto elem_type = out_tensor.GetTensorTypeAndShapeInfo().GetElementType();
            std::vector<float> result(ctx.latent_size);
            if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
                const auto *data = out_tensor.GetTensorData<Ort::Float16_t>();
                for (int j = 0; j < ctx.latent_size; ++j)
                    result[j] = static_cast<float>(data[j]);
            } else {
                const auto *data = out_tensor.GetTensorData<float>();
                std::copy(data, data + ctx.latent_size, result.begin());
            }
            return result;
        } catch (const Ort::Exception &e) {
            ctx.dmlFailed = true;
            Logger::info(std::string("GPU unet failed, switching to CPU for remaining steps: ") + e.what());
            return runOnCpu();
        }
    }

    // Two batch=1 UNet passes with CFG blending:
    //   eps_cfg = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
    // Using two separate passes instead of one batch=2 pass ensures compatibility
    // with models exported with a static batch size of 1 (required for DML EP).
    std::vector<float> runUNetCFG(const std::vector<float> &x_t,
                                  float sigma,
                                  const std::vector<float> &alphas_cumprod,
                                  GenerationContext &ctx) {
        int t = sigmaToTimestep(sigma, alphas_cumprod);

        auto u_eps_fp16 = runUNetSingle(x_t, t, ctx.uncond_embed, ctx);
        auto c_eps_fp16 = runUNetSingle(x_t, t, ctx.text_embed, ctx);

        std::vector<float> eps(ctx.latent_size);
        for (int j = 0; j < ctx.latent_size; ++j)
            eps[j] = static_cast<float>(u_eps_fp16[j])
                     + ctx.guidance_scale * (static_cast<float>(c_eps_fp16[j]) - static_cast<float>(u_eps_fp16[j]));
        return eps;
    }

    // ── DPM++ 2M Karras denoising loop ───────────────────────────────────────────

    std::vector<float> denoiseSingleLatent(const std::vector<float> &sigmas,
                                           int num_steps,
                                           const std::vector<float> &alphas_cumprod,
                                           GenerationContext &ctx,
                                           std::atomic<int> *progressStep,
                                           std::atomic<bool> *cancelToken) {
        std::vector<float> x(ctx.latent_size);
        for (auto &v: x) v = randNormal() * sigmas[0];

        std::vector<float> prev_denoised;
        float h_prev = 0.0f;
        auto tDenoise = Clock::now();

        for (int step = 0; step < num_steps; ++step) {
            if (cancelToken && cancelToken->load()) {
                Logger::info("Denoising cancelled at step " + std::to_string(step));
                return {};
            }
            float sigma = sigmas[step];
            float sigma_next = sigmas[step + 1];
            auto tStep = Clock::now();
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
            float h = std::log(sigma / sigma_next);
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
            h_prev = h;

            Logger::info("  step done in " + fmtMs(tStep));
            if (progressStep) progressStep->fetch_add(1);
        }

        Logger::info("Denoising complete in " + fmtMs(tDenoise));
        return x;
    }

    // ── VAE decoding ──────────────────────────────────────────────────────────────

    // The exported ONNX model handles the 0.18215 unscaling internally.
    cv::Mat decodeLatent(const std::vector<float> &x, GenerationContext &ctx) {
        // Log input latent stats to catch numerical issues before they hit the GPU.
        {
            float vMin = 1e9f, vMax = -1e9f, vSum = 0.0f;
            for (float v : x) { vMin = std::min(vMin, v); vMax = std::max(vMax, v); vSum += v; }
            Logger::info("VAE input latent — min: " + std::to_string(vMin)
                         + "  max: " + std::to_string(vMax)
                         + "  mean: " + std::to_string(vSum / static_cast<float>(x.size())));
        }

        // VAE model is exported as FP16 so inputs must match — ORT throws on type mismatch.
        auto vae_latent_fp16 = toFp16(x);

        Ort::Value vae_input = Ort::Value::CreateTensor<Ort::Float16_t>(
            ctx.memory_info, vae_latent_fp16.data(), vae_latent_fp16.size(),
            ctx.latent_shape.data(), ctx.latent_shape.size());

        const char *vae_in_names[] = {ctx.vae_in.c_str()};
        const char *vae_out_names[] = {ctx.vae_out.c_str()};
        auto tVae = Clock::now();
        Logger::info("VAE decoding latent → image...");
        auto vae_out = ctx.vae_decoder.Run(Ort::RunOptions{nullptr},
                                           vae_in_names, &vae_input, 1,
                                           vae_out_names, 1);
        Logger::info("VAE decode done in " + fmtMs(tVae));

        auto &vae_tensor = vae_out.front();
        auto shape_info = vae_tensor.GetTensorTypeAndShapeInfo();
        auto shape = shape_info.GetShape();
        int img_h = static_cast<int>(shape[2]);
        int img_w = static_cast<int>(shape[3]);
        size_t elem_count = shape_info.GetElementCount();
        auto elem_type = shape_info.GetElementType();
        Logger::info("VAE output shape: [1, 3, " + std::to_string(img_h) + ", " + std::to_string(img_w) + "]"
                     + "  dtype=" + ((elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) ? "fp16" : "fp32"));

        std::vector<float> img_float(elem_count);
        if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
            const auto *raw = vae_tensor.GetTensorData<Ort::Float16_t>();
            for (size_t j = 0; j < elem_count; ++j)
                img_float[j] = static_cast<float>(raw[j]);
        } else {
            const auto *raw = vae_tensor.GetTensorData<float>();
            std::copy(raw, raw + elem_count, img_float.begin());
        }

        return latentToImage(img_float.data(), img_w, img_h);
    }

    // ── Shared generation pipeline ────────────────────────────────────────────────

    void runPipeline(const std::string &prompt,
                     const std::string &neg_prompt,
                     const std::string &outputPath,
                     const GenerationParams &params,
                     const std::string &modelDir,
                     std::atomic<int> *progressStep,
                     std::atomic<int> *currentImage,
                     std::atomic<bool> *cancelToken) {
        auto tTotal = Clock::now();

        Logger::info("=== runPipeline ===");
        Logger::info("Working directory: " + std::filesystem::current_path().string());
        Logger::info("Output base: " + outputPath);
        std::filesystem::create_directories("assets/generated");

        constexpr int image_w = 512;
        constexpr int image_h = 512;
        const int num_steps = params.numSteps;
        const int num_images = params.numImages;
        constexpr int T = 1000;
        constexpr float beta_start = 0.00085f;
        constexpr float beta_end = 0.012f;

        Logger::info("Steps: " + std::to_string(num_steps)
                     + "  guidance: " + std::to_string(params.guidanceScale)
                     + "  images: " + std::to_string(num_images)
                     + "  latent: " + std::to_string(image_w / 8) + "x" + std::to_string(image_h / 8));
        Logger::info("Prompt: " + prompt);
        Logger::info("Neg:    " + neg_prompt);

        // Load models once for all images in this run.
        auto ctx = loadModels(image_w / 8, image_h / 8, modelDir);
        ctx.guidance_scale = params.guidanceScale;

        // Encode text once — same prompt for every image in the batch.
        auto tEncode = Clock::now();
        ClipTokenizer tokenizer("models/vocab.json", "models/merges.txt");
        ctx.text_embed = encodeText(prompt, tokenizer, ctx, ctx.embed_shape);
        ctx.uncond_embed = encodeText(neg_prompt, tokenizer, ctx, ctx.embed_shape);
        Logger::info("Text encoding total: " + fmtMs(tEncode));

        auto alphas_cumprod = buildAlphasCumprod(T, beta_start, beta_end);
        auto sigmas = buildKarrasSchedule(alphas_cumprod, num_steps);

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
                seedRng();

                // Build per-image output path: insert _N before extension for multi-image runs.
                std::string outPath = outputPath;
                if (num_images > 1) {
                    const auto dot = outputPath.rfind('.');
                    const std::string idx = std::to_string(i + 1);
                    outPath = (dot == std::string::npos)
                        ? outputPath + "_" + idx
                        : outputPath.substr(0, dot) + "_" + idx + outputPath.substr(dot);
                }

                Logger::info("--- Image " + std::to_string(i + 1) + "/" + std::to_string(num_images) + " ---");
                auto latent = denoiseSingleLatent(sigmas, num_steps, alphas_cumprod, ctx, progressStep, cancelToken);

                if (cancelToken && cancelToken->load()) break;

                float lat_min = 1e9f, lat_max = -1e9f, lat_sum = 0.0f;
                for (float v: latent) {
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
        } catch (const Ort::Exception &) {
            Logger::info("Generation cancelled mid-step (ORT terminated).");
        }

        pipelineDone.store(true);
        watcher.join();
    }
} // namespace

// ── Public entry points ───────────────────────────────────────────────────────

void PortraitGeneratorAi::generatePortrait(const Race race, Gender gender, const GenerationParams &params,
                                           std::atomic<int> *progressStep) {

    const std::string prompt = buildCharacterPrompt(race, gender).build();
    const std::string neg_prompt = buildNegativePrompt(race, gender).build();
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::seconds>(
        now.time_since_epoch()
    ).count();
    const std::string outputPath = "assets/generated/portrait_" + std::to_string(timestamp) + ".png";

    runPipeline(prompt, neg_prompt, outputPath, params, "models", progressStep, nullptr, nullptr);
}

void PortraitGeneratorAi::generateFromPrompt(const std::string &prompt,
                                             const std::string &negativePrompt,
                                             const std::string &outputPath,
                                             const GenerationParams &params,
                                             const std::string &modelDir,
                                             std::atomic<int> *progressStep,
                                             std::atomic<int> *currentImage,
                                             std::atomic<bool> *cancelToken) {
    Logger::info("generateFromPrompt — model: " + modelDir);
    runPipeline(prompt, negativePrompt, outputPath, params, modelDir, progressStep, currentImage, cancelToken);
}
