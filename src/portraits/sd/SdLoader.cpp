#include "SdLoader.hpp"
#include "SdUtils.hpp"
#include "SdSafetensors.hpp"
#include "SdOnnxPatcher.hpp"
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

    // ── File reader ───────────────────────────────────────────────────────────────

    static std::vector<uint8_t> readFileBytes(const std::string& path) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) throw std::runtime_error("Cannot open: " + path);
        const auto size = static_cast<size_t>(f.tellg());
        f.seekg(0);
        std::vector<uint8_t> buf(size);
        f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(size));
        return buf;
    }

    // ── Model byte cache ──────────────────────────────────────────────────────────
    // Base model bytes (unpatched, immutable) keyed by absolute file path.
    // Populated on the first LoRA run; avoids disk I/O on subsequent runs.
    static std::unordered_map<std::string,
                               std::shared_ptr<const std::vector<uint8_t>>> s_modelBytesCache;

    static std::shared_ptr<const std::vector<uint8_t>>
    cachedReadFileBytes(const std::string& path) {
        auto it = s_modelBytesCache.find(path);
        if (it != s_modelBytesCache.end()) {
            Logger::info("  Byte cache hit: " + std::filesystem::path(path).filename().string());
            return it->second;
        }
        Logger::info("  Byte cache miss (reading from disk): "
                     + std::filesystem::path(path).filename().string());
        auto ptr = std::make_shared<const std::vector<uint8_t>>(readFileBytes(path));
        s_modelBytesCache[path] = ptr;
        return ptr;
    }

    // ── Session loading ───────────────────────────────────────────────────────────

    GenerationContext loadModels(const ModelConfig&            cfg,
                                 const std::string&            modelDir,
                                 const std::vector<LoraEntry>& loras) {
        const int latent_w = cfg.image_w / 8;
        const int latent_h = cfg.image_h / 8;
        auto t0 = Clock::now();
        GenerationContext ctx;
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

        Logger::info("Checking model files...");
        {
            std::vector<std::string> required = {
                modelDir + "/text_encoder.onnx",
                modelDir + "/unet.onnx",
                modelDir + "/vae_decoder.onnx",
                "models/vocab.json",
                "models/merges.txt",
            };
            if (cfg.type == ModelType::SDXL)
                required.push_back(modelDir + "/text_encoder_2.onnx");
            for (const auto& p : required)
                Logger::info((std::filesystem::exists(p) ? "  [OK] " : "  [MISSING] ") + p);
        }

        // Path-based loader — used for VAE (no LoRA layers) and the no-LoRA fast path.
        auto loadSession = [&](const char* label, auto path, Ort::SessionOptions& opts) {
            Logger::info("Loading " + std::string(label) + "...");
            auto ts = Clock::now();
            Ort::Session s(ctx.env, path, opts);
            Logger::info("  " + std::string(label) + " loaded in " + fmtMs(ts));
            return s;
        };

        // Select which session options each model uses.
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

#ifdef _WIN32
        auto toWide = [](const std::string& s) { return std::wstring(s.begin(), s.end()); };
        const std::wstring wModelDir = toWide(modelDir);
#endif

        if (loras.empty()) {
            Logger::info("No LoRA adapters configured — loading models directly from disk");
            // No LoRA: load directly from path so ORT can memory-map the file.
#ifdef _WIN32
            ctx.text_encoder = loadSession("text_encoder", (wModelDir + L"/text_encoder.onnx").c_str(), auxOpts);
            ctx.unet         = loadSession("unet",         (wModelDir + L"/unet.onnx").c_str(),         unetOpts);
            ctx.cpu_unet     = loadSession("cpu_unet",     (wModelDir + L"/unet.onnx").c_str(),         ctx.cpu_session_opts);
            ctx.vae_decoder  = loadSession("vae_decoder",  (wModelDir + L"/vae_decoder.onnx").c_str(),  auxOpts);
            if (cfg.type == ModelType::SDXL)
                ctx.text_encoder_2 = loadSession("text_encoder_2",
                                                 (wModelDir + L"/text_encoder_2.onnx").c_str(), auxOpts);
#else
            ctx.text_encoder = loadSession("text_encoder", (modelDir + "/text_encoder.onnx").c_str(), auxOpts);
            ctx.unet         = loadSession("unet",         (modelDir + "/unet.onnx").c_str(),         unetOpts);
            ctx.cpu_unet     = loadSession("cpu_unet",     (modelDir + "/unet.onnx").c_str(),         ctx.cpu_session_opts);
            ctx.vae_decoder  = loadSession("vae_decoder",  (modelDir + "/vae_decoder.onnx").c_str(),  auxOpts);
            if (cfg.type == ModelType::SDXL)
                ctx.text_encoder_2 = loadSession("text_encoder_2",
                                                 (modelDir + "/text_encoder_2.onnx").c_str(), auxOpts);
#endif
        } else {
            // LoRA path: read each model into memory, apply patches, create session from bytes.
            Logger::info("LoRA mode: " + std::to_string(loras.size()) + " adapter(s) configured.");

            // Helper: read base bytes (from cache), apply all LoRA adapters in sequence,
            // and return a shared_ptr to the final patched buffer.
            auto makePatchedBytes = [&](const std::string& path)
                -> std::shared_ptr<const std::vector<uint8_t>> {
                Logger::info("  Preparing " + std::filesystem::path(path).filename().string() + "...");
                auto base = cachedReadFileBytes(path);   // shared_ptr<const vector> — no disk I/O on hit
                auto idx  = parseTensorIndex(*base);
                auto sidx = buildSuffixIndex(idx);
                int  total = 0;

                std::shared_ptr<const std::vector<uint8_t>> current = base;

                for (const auto& lo : loras) {
                    Logger::info("Applying LoRA: " + lo.path + "  scale=" + std::to_string(lo.scale));
                    try {
                        auto rawLoraMap = loadSafetensors(lo.path);
                        Logger::info("  LoRA tensors loaded: " + std::to_string(rawLoraMap.size()) + " key(s)");

                        auto [patched, n] = applyLoraToBytes(current, sidx, rawLoraMap, lo.scale);
                        current = patched;   // shared_ptr<T> → shared_ptr<const T> (implicit)
                        total  += n;
                        Logger::info("  Total layers patched for this LoRA: " + std::to_string(n));

                    } catch (const std::exception& e) {
                        Logger::info("  LoRA '" + lo.path + "' skipped: " + std::string(e.what()));
                    }
                }

                Logger::info("  Total LoRA patches for this model: " + std::to_string(total));
                return current;
            };

            // Helper: create an ORT session from a shared byte buffer.
            auto sessionFromBytes = [&](const char* label,
                                        const std::vector<uint8_t>& bytes,
                                        Ort::SessionOptions& opts) -> Ort::Session {
                auto ts = Clock::now();
                Ort::Session s(ctx.env, bytes.data(), bytes.size(), opts);
                Logger::info("  " + std::string(label) + " session created in " + fmtMs(ts));
                return s;
            };

            {
                auto bytes = makePatchedBytes(modelDir + "/text_encoder.onnx");
                ctx.text_encoder = sessionFromBytes("text_encoder", *bytes, auxOpts);
            }
            {
                // UNet bytes are shared between unet and cpu_unet to avoid reading
                // and patching the (potentially 2 GB) file twice.
                auto unetBytes = makePatchedBytes(modelDir + "/unet.onnx");
                ctx.unet     = sessionFromBytes("unet",     *unetBytes, unetOpts);
                ctx.cpu_unet = sessionFromBytes("cpu_unet", *unetBytes, ctx.cpu_session_opts);
            }
            // VAE has no LoRA layers in any known kohya-format adapter file.
            // Load from path to avoid buffering an extra ~1 GB in RAM.
#ifdef _WIN32
            ctx.vae_decoder = loadSession("vae_decoder", (wModelDir + L"/vae_decoder.onnx").c_str(), auxOpts);
#else
            ctx.vae_decoder = loadSession("vae_decoder", (modelDir + "/vae_decoder.onnx").c_str(), auxOpts);
#endif
            if (cfg.type == ModelType::SDXL) {
                auto bytes = makePatchedBytes(modelDir + "/text_encoder_2.onnx");
                ctx.text_encoder_2 = sessionFromBytes("text_encoder_2", *bytes, auxOpts);
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
        return ctx;
    }

}