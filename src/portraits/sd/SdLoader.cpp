#include "SdLoader.hpp"
#include "SdUtils.hpp"
#include "SdSafetensors.hpp"
#include "SdOnnxPatcher.hpp"
#include "SdLoraApply.hpp"
#include <filesystem>
#include <fstream>
#include <thread>
#include <nlohmann/json.hpp>
#include "../../managers/Logger.hpp"
#include <unordered_map>

#ifdef USE_DML
// Forward-declare the DML entry point to avoid pulling in DirectML.h (Windows
// SDK header not available during Linux cross-compilation).
extern "C" OrtStatus* OrtSessionOptionsAppendExecutionProvider_DML(
    OrtSessionOptions* options, int device_id);
#endif

namespace sd {
    // ── Model I/O logging ─────────────────────────────────────────────────────────

    static const char* kOrtTypeNames[] = {
        "undefined", "float", "uint8", "int8", "uint16", "int16", "int32", "int64",
        "string", "bool", "float16", "double", "uint32", "uint64",
        "complex64", "complex128", "bfloat16"
    };

    static void logModelIO(const char* label,
                           Ort::Session& session,
                           Ort::AllocatorWithDefaultOptions& allocator) {
        for (size_t k = 0; k < session.GetInputCount(); ++k) {
            auto info = session.GetInputTypeInfo(k);
            const auto t = info.GetTensorTypeAndShapeInfo().GetElementType();
            std::string tname = (t < 17) ? kOrtTypeNames[t] : std::to_string(t);
            Logger::info(std::string(label) + " input[" + std::to_string(k) + "]: "
                         + session.GetInputNameAllocated(k, allocator).get()
                         + " (" + tname + ")");
        }
        for (size_t k = 0; k < session.GetOutputCount(); ++k) {
            auto info = session.GetOutputTypeInfo(k);
            const auto t = info.GetTensorTypeAndShapeInfo().GetElementType();
            std::string tname = (t < 17) ? kOrtTypeNames[t] : std::to_string(t);
            Logger::info(std::string(label) + " output[" + std::to_string(k) + "]: "
                         + session.GetOutputNameAllocated(k, allocator).get()
                         + " (" + tname + ")");
        }
    }

    // ── Model config ──────────────────────────────────────────────────────────────

    ModelConfig loadModelConfig(const std::string& modelDir) {
        ModelConfig cfg;
        const std::string jsonPath = modelDir + "/model.json";
        if (!std::filesystem::exists(jsonPath)) {
            Logger::info("No model.json found — assuming SD 1.5");
            return cfg;
        }
        try {
            std::ifstream f(jsonPath);
            const auto j = nlohmann::json::parse(f);
            const std::string type = j.value("type", "sd15");
            if (type == "sdxl") {
                cfg.type    = ModelType::SDXL;
                cfg.image_w = 1024;
                cfg.image_h = 1024;
            }
            Logger::info("model.json: type=" + type
                         + "  resolution=" + std::to_string(cfg.image_w)
                         + "x" + std::to_string(cfg.image_h));
        } catch (const std::exception& e) {
            Logger::info(std::string("model.json parse error, defaulting to SD 1.5: ") + e.what());
        }
        return cfg;
    }

    // ── External index cache ──────────────────────────────────────────────────────
    // Maps absolute .onnx path → parsed external-data index.
    // Used exclusively by the LoRA path (no-LoRA path loads sessions from the
    // file path directly; ORT resolves .onnx.data natively).

    static std::unordered_map<std::string, OnnxExternalIndex> s_extIndexCache;

    static const OnnxExternalIndex&
    cachedExternalIndex(const OnnxModelBundle& bundle) {
        const std::string key = bundle.onnxPath.string();
        auto it = s_extIndexCache.find(key);
        if (it != s_extIndexCache.end()) {
            Logger::info("  Ext-index cache hit: " + bundle.onnxPath.filename().string());
            return it->second;
        }
        Logger::info("  Ext-index cache miss — parsing: "
                     + bundle.onnxPath.filename().string());
        auto [ins, ok] = s_extIndexCache.emplace(key, parseExternalIndex(bundle));
        (void)ok;
        return ins->second;
    }

    // ── Session loading ───────────────────────────────────────────────────────────

    ModelInstance loadModels(const ModelConfig&            cfg,
                             const std::string&            modelDir,
                             const std::vector<LoraEntry>& loras) {
        const int latent_w = cfg.image_w / 8;
        const int latent_h = cfg.image_h / 8;
        auto t0 = Clock::now();
        ModelInstance result;
        GenerationContext& ctx = result.ctx;
        ctx.model_type = cfg.type;
        const int numThreads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
        Logger::info("=== loadModels ===");
        Logger::info("Model dir: " + modelDir);
        Logger::info("Model type: " + std::string(cfg.type == ModelType::SDXL ? "SDXL" : "SD 1.5"));
        Logger::info("CPU threads: " + std::to_string(numThreads));
        ctx.session_opts.SetIntraOpNumThreads(numThreads);
        ctx.cpu_session_opts.SetIntraOpNumThreads(numThreads);

#if defined(USE_DML)
        try {
            // DML requires these flags to avoid E_INVALIDARG on reshape nodes.
            // ORT_ENABLE_BASIC prevents graph-level rewrites that produce Reshape
            // patterns DML can't execute; DisableMemPattern + ORT_SEQUENTIAL stop
            // dynamic memory reuse patterns that also trigger the same error.
            ctx.session_opts.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
            ctx.session_opts.DisableMemPattern();
            ctx.session_opts.SetExecutionMode(ORT_SEQUENTIAL);
            Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(ctx.session_opts, 0));
            Logger::info("EP: DirectML (unet=GPU, vae=CPU, text_encoders=CPU)");
        } catch (const Ort::Exception& e) {
            Logger::info(std::string("DirectML EP unavailable, falling back to CPU: ") + e.what());
        }
#elif defined(USE_CUDA)
        try {
            OrtCUDAProviderOptions cuda_options;
            ctx.session_opts.AppendExecutionProvider_CUDA(cuda_options);
            Logger::info("EP: CUDA (unet=GPU, vae=GPU, text_encoders=GPU)");
        } catch (const Ort::Exception& e) {
            Logger::info(std::string("CUDA EP unavailable, falling back to CPU: ") + e.what());
        }
#else
        Logger::info("EP: CPU");
#endif

        // ── Resolve model bundles ─────────────────────────────────────────────────
        // resolveBundle() verifies each .onnx file exists and detects whether a
        // .onnx.data companion is present.  It throws immediately on a missing
        // file — before any ORT session is created — with a clear error message.
        Logger::info("Resolving model bundles...");
        namespace fs = std::filesystem;
        const fs::path mdir(modelDir);
        auto teBundle   = resolveBundle(mdir / "text_encoder.onnx");
        auto unetBundle = resolveBundle(mdir / "unet.onnx");
        auto vaeBundle  = resolveBundle(mdir / "vae_decoder.onnx");
        for (const auto* p : {"models/vocab.json", "models/merges.txt"})
            Logger::info((fs::exists(p) ? "  [OK] " : "  [MISSING] ") + std::string(p));
        // text_encoder_2 (SDXL only) is resolved at the point of use below.

        // ── Session loader helper ─────────────────────────────────────────────────
        // Takes a filesystem::path so the same code works on Linux (char) and
        // Windows (wchar_t) — path::c_str() returns the native character type
        // which matches ORT's ORTCHAR_T overload on both platforms.
        auto loadSession = [&](const char* label,
                               const fs::path& fspath,
                               Ort::SessionOptions& opts) {
            Logger::info("Loading " + std::string(label) + "...");
            auto ts = Clock::now();
            Ort::Session s(ctx.env, fspath.c_str(), opts);
            Logger::info("  " + std::string(label) + " loaded in " + fmtMs(ts));
            return s;
        };

        // ── Execution-provider session-option routing ─────────────────────────────
        // DML: SDXL UNet falls back to CPU (unsupported Reshape nodes at runtime).
        //      VAE and text encoders always use CPU under DML.
        // CUDA: all models on GPU.
        // CPU: UNet uses session_opts (no EP), aux models use cpu_session_opts.
#ifdef USE_DML
        auto& unetOpts = (cfg.type == ModelType::SDXL) ? ctx.cpu_session_opts : ctx.session_opts;
        auto& auxOpts  = ctx.cpu_session_opts;
#elif defined(USE_CUDA)
        auto& unetOpts = ctx.session_opts;
        auto& auxOpts  = ctx.session_opts;
#else
        auto& unetOpts = ctx.session_opts;
        auto& auxOpts  = ctx.cpu_session_opts;
#endif

        if (loras.empty()) {
            // ── No-LoRA path ──────────────────────────────────────────────────────
            // Load directly from the .onnx path so ORT can memory-map the file and
            // automatically resolve any .onnx.data companion from the same directory.
            // No bytes are buffered in our process for this path.
            Logger::info("No LoRA adapters — loading models directly from disk");

            ctx.text_encoder = loadSession("text_encoder", teBundle.onnxPath,  auxOpts);
            ctx.unet         = loadSession("unet",         unetBundle.onnxPath, unetOpts);

            // cpu_unet is the DML GPU→CPU fallback.  Skip under CUDA: loading a
            // second copy of a multi-GB UNet would OOM.  For DML/CPU builds, wrap
            // in try-catch so a low-RAM machine degrades gracefully.
#ifndef USE_CUDA
            try {
                ctx.cpu_unet = loadSession("cpu_unet", unetBundle.onnxPath, ctx.cpu_session_opts);
                ctx.cpuFallbackAvailable = true;
            } catch (const std::exception& e) {
                Logger::info(std::string("cpu_unet load failed — GPU fallback disabled: ") + e.what());
            }
#endif
            ctx.vae_decoder = loadSession("vae_decoder", vaeBundle.onnxPath, auxOpts);

            if (cfg.type == ModelType::SDXL) {
                auto te2Bundle = resolveBundle(mdir / "text_encoder_2.onnx");
                ctx.text_encoder_2 = loadSession("text_encoder_2", te2Bundle.onnxPath, auxOpts);
            }

        } else {
            // ── LoRA path ─────────────────────────────────────────────────────────
            // Uses ORT-native external data loading + AddExternalInitializers.
            //
            // For each model component:
            //   1. parseExternalIndex(): parse .onnx to get {normName → offset/length} map
            //   2. buildExternalSuffixIndex(): build O(1) suffix lookup
            //   3. buildLoraOverrides(): read matched base weights from .onnx.data,
            //      apply LoRA deltas in fp32, convert back to model dtype
            //   4. SessionOptions::AddExternalInitializers(): inject patched tensors
            //   5. Ort::Session(env, path, opts): ORT loads non-patched weights natively
            Logger::info("LoRA mode: " + std::to_string(loras.size()) + " adapter(s)");

            auto makeLoraSession = [&](const char* label,
                                       const OnnxModelBundle& bundle,
                                       Ort::SessionOptions& baseOpts) -> Ort::Session {
                Logger::info("  Preparing " + std::string(label) + " with LoRA...");
                const OnnxExternalIndex&  extIdx    = cachedExternalIndex(bundle);
                const OnnxExternalSuffixIndex extSufIdx = buildExternalSuffixIndex(extIdx);
                LoraOverrides overrides = buildLoraOverrides(bundle, extIdx, extSufIdx, loras);

                if (overrides.empty()) {
                    // No LoRA layers matched this model component — load natively.
                    Logger::info("  No LoRA matches for " + std::string(label)
                                 + " — loading from path");
                    return loadSession(label, bundle.onnxPath, baseOpts);
                }

                // Clone session options so AddExternalInitializers doesn't mutate the
                // shared opts (which is reused across components and runs).
                Ort::SessionOptions cloned = baseOpts.Clone();
                cloned.AddExternalInitializers(overrides.names, overrides.values);
                // overrides (backing buffers) must stay alive until session is constructed.
                Logger::info("  Creating " + std::string(label) + " session with "
                             + std::to_string(overrides.names.size()) + " override(s)...");
                auto ts = Clock::now();
                Ort::Session s(ctx.env, bundle.onnxPath.c_str(), cloned);
                Logger::info("  " + std::string(label) + " loaded in " + fmtMs(ts));
                return s;
            };

            ctx.text_encoder = makeLoraSession("text_encoder", teBundle,   auxOpts);
            ctx.unet         = makeLoraSession("unet",         unetBundle, unetOpts);
#ifndef USE_CUDA
            try {
                ctx.cpu_unet = makeLoraSession("cpu_unet", unetBundle, ctx.cpu_session_opts);
                ctx.cpuFallbackAvailable = true;
            } catch (const std::exception& e) {
                Logger::info(std::string("cpu_unet load failed — GPU fallback disabled: ") + e.what());
            }
#endif
            // VAE has no LoRA layers in any known kohya adapter — load natively.
            ctx.vae_decoder = loadSession("vae_decoder", vaeBundle.onnxPath, auxOpts);

            if (cfg.type == ModelType::SDXL) {
                auto te2Bundle = resolveBundle(mdir / "text_encoder_2.onnx");
                ctx.text_encoder_2 = makeLoraSession("text_encoder_2", te2Bundle, auxOpts);
            }
        }

        logModelIO("text_encoder", ctx.text_encoder, ctx.allocator);
        logModelIO("unet",         ctx.unet,         ctx.allocator);
        logModelIO("vae_decoder",  ctx.vae_decoder,  ctx.allocator);
        if (cfg.type == ModelType::SDXL)
            logModelIO("text_encoder_2", ctx.text_encoder_2, ctx.allocator);

        // Cache I/O names — GetInputNameAllocated returns a temporary, copy to std::string.
        ctx.te_input  = ctx.text_encoder.GetInputNameAllocated(0,  ctx.allocator).get();
        ctx.te_output = ctx.text_encoder.GetOutputNameAllocated(0, ctx.allocator).get();
        ctx.unet_in0  = ctx.unet.GetInputNameAllocated(0,  ctx.allocator).get();
        ctx.unet_in1  = ctx.unet.GetInputNameAllocated(1,  ctx.allocator).get();
        ctx.unet_in2  = ctx.unet.GetInputNameAllocated(2,  ctx.allocator).get();
        ctx.unet_out0 = ctx.unet.GetOutputNameAllocated(0, ctx.allocator).get();
        ctx.vae_in    = ctx.vae_decoder.GetInputNameAllocated(0,  ctx.allocator).get();
        ctx.vae_out   = ctx.vae_decoder.GetOutputNameAllocated(0, ctx.allocator).get();

        if (cfg.type == ModelType::SDXL) {
            ctx.te2_input  = ctx.text_encoder_2.GetInputNameAllocated(0,  ctx.allocator).get();
            ctx.te2_output = ctx.text_encoder_2.GetOutputNameAllocated(0, ctx.allocator).get();
            ctx.te2_pooled = ctx.text_encoder_2.GetOutputNameAllocated(1, ctx.allocator).get();
            ctx.unet_in3   = ctx.unet.GetInputNameAllocated(3, ctx.allocator).get();
            ctx.unet_in4   = ctx.unet.GetInputNameAllocated(4, ctx.allocator).get();
        }

        ctx.unetExpectsFp32 =
            (ctx.unet.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType()
             == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
        ctx.vaeExpectsFp32 =
            (ctx.vae_decoder.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType()
             == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

        Logger::info("UNet input dtype : " + std::string(ctx.unetExpectsFp32 ? "float32" : "float16"));
        Logger::info("VAE  input dtype : " + std::string(ctx.vaeExpectsFp32  ? "float32" : "float16"));
        Logger::info("Latent shape     : [1, 4, " + std::to_string(latent_h) + ", " + std::to_string(latent_w) + "]");
#ifdef USE_CUDA
        Logger::info("Session EP       : text_encoder=CUDA  unet=CUDA  vae=CUDA");
#elif defined(USE_DML)
        Logger::info("Session EP       : text_encoder=CPU  unet="
                     + std::string((cfg.type == ModelType::SDXL) ? "CPU" : "DML") + "  vae=CPU");
#else
        Logger::info("Session EP       : CPU");
#endif

        ctx.latent_shape = {1, 4, latent_h, latent_w};
        ctx.latent_size  = 1 * 4 * latent_h * latent_w;

        Logger::info("All models loaded in " + fmtMs(t0));
        Logger::info("=================");
        return result;
    }

}