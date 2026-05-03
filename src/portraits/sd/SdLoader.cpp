#include "SdLoader.hpp"
#include "SdUtils.hpp"
#include "SdSafetensors.hpp"
#include "SdOnnxPatcher.hpp"
#include "LoraInjector.hpp"
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
    // Persistent per-component LoRA injectors keyed by .onnx path.
    // Survives across loadModels() calls so base weights are not reloaded from
    // disk on every LoRA config change — only the merge computation is redone.
    static std::unordered_map<std::string, LoraInjector> s_injectors;

    // ── Model I/O logging ─────────────────────────────────────────────────────────

    static const char* kOrtTypeNames[] = {
        "undefined", "float", "uint8", "int8", "uint16", "int16", "int32", "int64",
        "string", "bool", "float16", "double", "uint32", "uint64",
        "complex64", "complex128", "bfloat16"
    };

    // Runs one dummy forward pass to trigger ORT JIT / kernel compilation.
    // Queries input shapes at runtime; replaces dynamic dims (-1) with 1.
    // Non-fatal: logs and returns on any failure.
    static void warmupSession(const char*                       label,
                              Ort::Session&                     session,
                              Ort::MemoryInfo&                  memInfo,
                              Ort::AllocatorWithDefaultOptions& allocator) {
        auto ts = Clock::now();
        try {
            const size_t nIn  = session.GetInputCount();
            const size_t nOut = session.GetOutputCount();

            // Build zero-filled backing buffers, one per input.
            struct Buf {
                std::vector<int64_t> shape;
                std::vector<uint8_t> raw;     // sizeof(elem) * nElems bytes, all zero
                ONNXTensorElementDataType dtype;
            };
            std::vector<Buf>        bufs(nIn);
            std::vector<Ort::Value> inputs;
            inputs.reserve(nIn);

            for (size_t i = 0; i < nIn; ++i) {
                auto& buf        = bufs[i];
                auto  typeInfo   = session.GetInputTypeInfo(i);
                auto  shapeInfo  = typeInfo.GetTensorTypeAndShapeInfo();
                buf.dtype        = shapeInfo.GetElementType();
                buf.shape        = shapeInfo.GetShape();
                for (auto& d : buf.shape) if (d < 0) d = 1;

                int64_t nElems = 1;
                for (auto d : buf.shape) nElems *= d;

                size_t elemBytes = 4;
                if (buf.dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)  elemBytes = 2;
                else if (buf.dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) elemBytes = 8;
                else if (buf.dtype != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                    Logger::info(std::string(label) + " warmup skipped (unsupported dtype)");
                    return;
                }
                buf.raw.assign(static_cast<size_t>(nElems) * elemBytes, 0u);

                if (buf.dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
                    inputs.push_back(Ort::Value::CreateTensor(
                        memInfo,
                        reinterpret_cast<Ort::Float16_t*>(buf.raw.data()),
                        static_cast<size_t>(nElems),
                        buf.shape.data(), buf.shape.size()));
                } else if (buf.dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                    inputs.push_back(Ort::Value::CreateTensor<float>(
                        memInfo,
                        reinterpret_cast<float*>(buf.raw.data()),
                        static_cast<size_t>(nElems),
                        buf.shape.data(), buf.shape.size()));
                } else {
                    inputs.push_back(Ort::Value::CreateTensor<int64_t>(
                        memInfo,
                        reinterpret_cast<int64_t*>(buf.raw.data()),
                        static_cast<size_t>(nElems),
                        buf.shape.data(), buf.shape.size()));
                }
            }

            std::vector<std::string> inNames, outNames;
            for (size_t i = 0; i < nIn;  ++i)
                inNames.push_back(session.GetInputNameAllocated(i,  allocator).get());
            for (size_t i = 0; i < nOut; ++i)
                outNames.push_back(session.GetOutputNameAllocated(i, allocator).get());

            std::vector<const char*> inPtrs, outPtrs;
            for (const auto& n : inNames)  inPtrs.push_back(n.c_str());
            for (const auto& n : outNames) outPtrs.push_back(n.c_str());

            session.Run(Ort::RunOptions{nullptr},
                        inPtrs.data(),  inputs.data(), nIn,
                        outPtrs.data(), nOut);

            Logger::info(std::string(label) + " warmup in " + fmtMs(ts));
        } catch (const std::exception& e) {
            Logger::info(std::string(label) + " warmup skipped: " + std::string(e.what()));
        }
    }

    // ── Model I/O logging ─────────────────────────────────────────────────────────

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

    static void validateManifest(const std::string& modelDir, const nlohmann::json& j) {
        if (!j.contains("components")) return;
        Logger::info("Validating model manifest...");
        for (const auto& [compName, comp] : j["components"].items()) {
            for (const char* key : {"onnx", "weights"}) {
                if (!comp.contains(key)) continue;
                const std::string file = comp[key].get<std::string>();
                const std::string full = modelDir + "/" + file;
                Logger::info((std::filesystem::exists(full) ? "  [OK]      " : "  [MISSING] ")
                             + file);
            }
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
            validateManifest(modelDir, j);
            const std::string type = j.value("type", "sd15");
            if (type == "sdxl") {
                cfg.type    = ModelType::SDXL;
                cfg.image_w = 1024;
                cfg.image_h = 1024;
                cfg.vae_scaling_factor = 0.13025f;
            }
            cfg.vae_scaling_factor = j.value("vae_scaling_factor", cfg.vae_scaling_factor);
            if (j.contains("capabilities")) {
                const auto& caps = j.at("capabilities");
                cfg.vaeEncoderAvailable = caps.value("vae_encoder_available", cfg.vaeEncoderAvailable);
                cfg.loraCompatible      = caps.value("lora_compatible",        cfg.loraCompatible);
            }
            Logger::info("model.json: type=" + type
                         + "  resolution=" + std::to_string(cfg.image_w)
                         + "x" + std::to_string(cfg.image_h)
                         + "  vae_scaling_factor=" + std::to_string(cfg.vae_scaling_factor)
                         + "  vae_encoder=" + (cfg.vaeEncoderAvailable ? "yes" : "no")
                         + "  lora=" + (cfg.loraCompatible ? "yes" : "no"));
        } catch (const std::exception& e) {
            Logger::info(std::string("model.json parse error, defaulting to SD 1.5: ") + e.what());
        }
        return cfg;
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
        ctx.vaeScalingFactor = cfg.vae_scaling_factor;
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
        const bool hasVaeEncoder = cfg.vaeEncoderAvailable && fs::exists(mdir / "vae_encoder.onnx");
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
            if (hasVaeEncoder) {
                auto vaeEncBundle = resolveBundle(mdir / "vae_encoder.onnx");
                ctx.vae_encoder = loadSession("vae_encoder", vaeEncBundle.onnxPath, auxOpts);
                ctx.vaeEncoderAvailable = true;
            }

            if (cfg.type == ModelType::SDXL) {
                auto te2Bundle = resolveBundle(mdir / "text_encoder_2.onnx");
                ctx.text_encoder_2 = loadSession("text_encoder_2", te2Bundle.onnxPath, auxOpts);
            }

        } else {
            // ── LoRA path ─────────────────────────────────────────────────────────
            // Uses ORT-native external data loading + AddExternalInitializers.
            //
            // For each LoRA-capable component:
            //   1. LoraInjector::loadModelMetadata() — parse .onnx for initializer metadata
            //   2. LoraInjector::applyLoras()        — load base weights from companion
            //        <stem>_weights.safetensors, compute W_merged = W_base + Σ deltas,
            //        validate, inject via AddExternalInitializers, return backing store
            //   3. Ort::Session(env, path, cloned)   — ORT loads non-patched weights natively
            //   4. backing store (LoraOverrides) destroyed after session is constructed
            Logger::info("LoRA mode: " + std::to_string(loras.size()) + " adapter(s)");

            // Helper: companion weights file path for a given .onnx bundle.
            auto weightsPath = [](const OnnxModelBundle& b) -> std::string {
                return (b.onnxPath.parent_path()
                        / (b.onnxPath.stem().string() + "_weights.safetensors"))
                       .string();
            };

            // Helper: apply LoRA injection and create session.
            // If no overrides matched (missing companion file, no LoRA keys for this
            // component, etc.) falls back to plain loadSession — same result, no injection.
            auto makeLoraSession = [&](const char*            label,
                                       const OnnxModelBundle& bundle,
                                       LoraInjector&          injector,
                                       Ort::SessionOptions&   baseOpts) -> Ort::Session {
                Logger::info("  Preparing " + std::string(label) + " with LoRA...");

                Ort::SessionOptions cloned   = baseOpts.Clone();
                LoraOverrides       overrides =
                    injector.applyLoras(cloned, weightsPath(bundle), loras);

                if (overrides.empty()) {
                    Logger::info("  No LoRA matches for " + std::string(label)
                                 + " — loading from path");
                    return loadSession(label, bundle.onnxPath, baseOpts);
                }

                // overrides (backing buffers) must stay alive until session is constructed.
                auto ts = Clock::now();
                Ort::Session s(ctx.env, bundle.onnxPath.c_str(), cloned);
                Logger::info("  " + std::string(label) + " loaded in " + fmtMs(ts));
                return s;
            };

            // Instantiate one injector per component (owns the metadata + merge cache).
            LoraInjector& teInjector   = s_injectors[teBundle.onnxPath.string()];
            LoraInjector& unetInjector = s_injectors[unetBundle.onnxPath.string()];
            teInjector.loadModelMetadata(teBundle.onnxPath.string());    // idempotent
            unetInjector.loadModelMetadata(unetBundle.onnxPath.string()); // idempotent

            ctx.text_encoder = makeLoraSession("text_encoder", teBundle,   teInjector,   auxOpts);
            ctx.unet         = makeLoraSession("unet",         unetBundle, unetInjector, unetOpts);
#ifndef USE_CUDA
            try {
                // cpu_unet reuses unetInjector — second call is a cache hit.
                ctx.cpu_unet = makeLoraSession("cpu_unet", unetBundle, unetInjector,
                                               ctx.cpu_session_opts);
                ctx.cpuFallbackAvailable = true;
            } catch (const std::exception& e) {
                Logger::info(std::string("cpu_unet load failed — GPU fallback disabled: ")
                             + e.what());
            }
#endif
            // VAE has no LoRA layers in any known kohya adapter — load natively.
            ctx.vae_decoder = loadSession("vae_decoder", vaeBundle.onnxPath, auxOpts);
            if (hasVaeEncoder) {
                auto vaeEncBundle = resolveBundle(mdir / "vae_encoder.onnx");
                ctx.vae_encoder = loadSession("vae_encoder", vaeEncBundle.onnxPath, auxOpts);
                ctx.vaeEncoderAvailable = true;
            }

            if (cfg.type == ModelType::SDXL) {
                auto te2Bundle = resolveBundle(mdir / "text_encoder_2.onnx");
                LoraInjector& te2Injector = s_injectors[te2Bundle.onnxPath.string()];
                te2Injector.loadModelMetadata(te2Bundle.onnxPath.string());
                ctx.text_encoder_2 = makeLoraSession("text_encoder_2", te2Bundle,
                                                      te2Injector, auxOpts);
            }
        }

        logModelIO("text_encoder", ctx.text_encoder, ctx.allocator);
        logModelIO("unet",         ctx.unet,         ctx.allocator);
        logModelIO("vae_decoder",  ctx.vae_decoder,  ctx.allocator);
        if (ctx.vaeEncoderAvailable)
            logModelIO("vae_encoder", ctx.vae_encoder, ctx.allocator);
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
        if (ctx.vaeEncoderAvailable) {
            ctx.vae_enc_in  = ctx.vae_encoder.GetInputNameAllocated(0,  ctx.allocator).get();
            ctx.vae_enc_out = ctx.vae_encoder.GetOutputNameAllocated(0, ctx.allocator).get();
        }

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
        if (ctx.vaeEncoderAvailable) {
            ctx.vaeEncoderExpectsFp32 =
                (ctx.vae_encoder.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType()
                 == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
        }

        Logger::info("UNet input dtype : " + std::string(ctx.unetExpectsFp32 ? "float32" : "float16"));
        Logger::info("VAE  input dtype : " + std::string(ctx.vaeExpectsFp32  ? "float32" : "float16"));
        Logger::info("VAE encoder      : " + std::string(ctx.vaeEncoderAvailable
            ? (ctx.vaeEncoderExpectsFp32 ? "float32" : "float16")
            : "not loaded"));
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

        // Warmup: trigger ORT JIT compilation before first real inference.
        Logger::info("Warming up sessions...");
        warmupSession("text_encoder", ctx.text_encoder,   ctx.memory_info, ctx.allocator);
        warmupSession("unet",         ctx.unet,            ctx.memory_info, ctx.allocator);
        if (ctx.cpuFallbackAvailable)
            warmupSession("cpu_unet", ctx.cpu_unet,        ctx.memory_info, ctx.allocator);
        warmupSession("vae_decoder",  ctx.vae_decoder,     ctx.memory_info, ctx.allocator);
        if (ctx.vaeEncoderAvailable)
            warmupSession("vae_encoder", ctx.vae_encoder,  ctx.memory_info, ctx.allocator);
        if (cfg.type == ModelType::SDXL)
            warmupSession("text_encoder_2", ctx.text_encoder_2, ctx.memory_info, ctx.allocator);

        Logger::info("All models loaded in " + fmtMs(t0));
        Logger::info("=================");
        return result;
    }

}
