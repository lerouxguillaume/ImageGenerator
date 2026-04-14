#include "SdUNet.hpp"
#include "SdUtils.hpp"
#include "SdScheduler.hpp"
#include "../../managers/Logger.hpp"

namespace sd {

std::vector<float> runUNetSingle(const std::vector<float>& x_t,
                                 int t,
                                 const std::vector<float>& embed,
                                 const std::vector<float>& pooled_embed,
                                 GenerationContext& ctx) {
    const bool isXL = ctx.model_type == ModelType::SDXL;
    const bool fp32 = ctx.unetExpectsFp32;

    std::vector<float>   latent_fp32 = x_t;
    std::vector<float>   embed_fp32  = embed;
    std::vector<float>   ts_fp32     = {static_cast<float>(t)};

    std::vector<int64_t> latent_shape = {1, ctx.latent_shape[1], ctx.latent_shape[2], ctx.latent_shape[3]};
    std::vector<int64_t> ts_shape     = {1};
    std::vector<int64_t> embed_shape  = {1, ctx.embed_shape[1], ctx.embed_shape[2]};

    std::vector<const char*> in_names = {
        ctx.unet_in0.c_str(), ctx.unet_in1.c_str(), ctx.unet_in2.c_str()
    };
    if (isXL) {
        in_names.push_back(ctx.unet_in3.c_str()); // text_embeds (1, 1280)
        in_names.push_back(ctx.unet_in4.c_str()); // time_ids    (1, 6)
    }
    const char* out_names[] = {ctx.unet_out0.c_str()};

    // FP16 copies — only populated when the model expects FP16.
    std::vector<Ort::Float16_t> latent_fp16, ts_fp16, embed_fp16;
    if (!fp32) {
        latent_fp16 = toFp16(latent_fp32);
        ts_fp16     = toFp16(ts_fp32);
        embed_fp16  = toFp16(embed_fp32);
    }

    // Push one of the three main typed inputs (latent / timestep / embedding).
    auto pushMain = [&](std::vector<Ort::Value>& v,
                        std::vector<float>& f32,
                        std::vector<Ort::Float16_t>& f16,
                        std::vector<int64_t>& shape) {
        if (fp32)
            v.push_back(Ort::Value::CreateTensor<float>(
                ctx.memory_info, f32.data(), f32.size(), shape.data(), shape.size()));
        else
            v.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(
                ctx.memory_info, f16.data(), f16.size(), shape.data(), shape.size()));
    };

    // Append SDXL extra inputs (text_embeds + time_ids); no-op for SD 1.5.
    // FP16 buffers are kept alive in these locals until Run() returns.
    std::vector<Ort::Float16_t> pool_fp16_buf, time_fp16_buf;
    auto appendSDXLInputs = [&](std::vector<Ort::Value>& inputs) {
        if (!isXL) return;
        std::vector<int64_t> pool_shape = {1, static_cast<int64_t>(pooled_embed.size())};
        std::vector<int64_t> time_shape = {1, static_cast<int64_t>(ctx.time_ids.size())};
        if (fp32) {
            inputs.push_back(Ort::Value::CreateTensor<float>(
                ctx.memory_info,
                const_cast<float*>(pooled_embed.data()), pooled_embed.size(),
                pool_shape.data(), pool_shape.size()));
            inputs.push_back(Ort::Value::CreateTensor<float>(
                ctx.memory_info,
                ctx.time_ids.data(), ctx.time_ids.size(),
                time_shape.data(), time_shape.size()));
        } else {
            pool_fp16_buf = toFp16(pooled_embed);
            inputs.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(
                ctx.memory_info,
                pool_fp16_buf.data(), pool_fp16_buf.size(),
                pool_shape.data(), pool_shape.size()));
            time_fp16_buf = toFp16(ctx.time_ids);
            inputs.push_back(Ort::Value::CreateTensor<Ort::Float16_t>(
                ctx.memory_info,
                time_fp16_buf.data(), time_fp16_buf.size(),
                time_shape.data(), time_shape.size()));
        }
    };

    auto decodeOutput = [&](std::vector<Ort::Value>& output) {
        auto& out_tensor = output.front();
        auto elem_type   = out_tensor.GetTensorTypeAndShapeInfo().GetElementType();
        std::vector<float> result(ctx.latent_size);
        if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
            const auto* data = out_tensor.GetTensorData<Ort::Float16_t>();
            for (int j = 0; j < ctx.latent_size; ++j)
                result[j] = static_cast<float>(data[j]);
        } else {
            const auto* data = out_tensor.GetTensorData<float>();
            std::copy(data, data + ctx.latent_size, result.begin());
        }
        return result;
    };

    auto runOnCpu = [&]() {
        std::vector<Ort::Value> inputs;
        pushMain(inputs, latent_fp32, latent_fp16, latent_shape);
        pushMain(inputs, ts_fp32,     ts_fp16,     ts_shape);
        pushMain(inputs, embed_fp32,  embed_fp16,  embed_shape);
        appendSDXLInputs(inputs);
        auto output = ctx.cpu_unet.Run(ctx.run_opts,
                                       in_names.data(), inputs.data(), inputs.size(), out_names, 1);
        return decodeOutput(output);
    };

    if (ctx.dmlFailed) {
        if (!ctx.cpuFallbackAvailable)
            throw std::runtime_error("GPU UNet failed and no CPU fallback session is loaded");
        return runOnCpu();
    }

    std::vector<Ort::Value> gpu_inputs;
    pushMain(gpu_inputs, latent_fp32, latent_fp16, latent_shape);
    pushMain(gpu_inputs, ts_fp32,     ts_fp16,     ts_shape);
    pushMain(gpu_inputs, embed_fp32,  embed_fp16,  embed_shape);
    appendSDXLInputs(gpu_inputs);

    try {
        auto output = ctx.unet.Run(ctx.run_opts,
                                   in_names.data(), gpu_inputs.data(), gpu_inputs.size(), out_names, 1);
        return decodeOutput(output);
    } catch (const Ort::Exception& e) {
        ctx.dmlFailed = true;
        if (!ctx.cpuFallbackAvailable) {
            Logger::info(std::string("GPU unet failed (no CPU fallback): ") + e.what());
            throw;
        }
        Logger::info(std::string("GPU unet failed, switching to CPU for remaining steps: ") + e.what());
        return runOnCpu();
    }
}

std::vector<float> runUNetCFG(const std::vector<float>& x_t,
                              float sigma,
                              const std::vector<float>& alphas_cumprod,
                              GenerationContext& ctx) {
    int t = sigmaToTimestep(sigma, alphas_cumprod);
    Logger::info("  UNet t=" + std::to_string(t) + "  (uncond)");
    auto tUncond = Clock::now();
    auto u_eps   = runUNetSingle(x_t, t, ctx.uncond_embed, ctx.uncond_embeds_pool, ctx);
    Logger::info("  uncond: " + fmtMs(tUncond) + "  (cond)");
    auto tCond   = Clock::now();
    auto c_eps   = runUNetSingle(x_t, t, ctx.text_embed,   ctx.text_embeds_pool,   ctx);
    Logger::info("  cond:   " + fmtMs(tCond));

    // CFG blend — optionally use a separate scale for the negative direction.
    // Standard:  eps = u + scale * (c - u)
    // Split:     eps = u * (1 - neg_scale) + pos_scale * c
    // When neg_scale == guidance_scale the two forms are identical.
    std::vector<float> eps(ctx.latent_size);
    for (int j = 0; j < ctx.latent_size; ++j)
        eps[j] = u_eps[j] * (1.0f - ctx.neg_guidance_scale)
               + ctx.guidance_scale * c_eps[j];

    // CFG rescale (Lin et al. 2023): prevents oversaturation at high guidance scales.
    // Rescales the blended eps so its std matches the conditional eps, then blends
    // back with the original by cfg_rescale factor.
    if (ctx.cfg_rescale > 0.0f) {
        auto stddev = [](const std::vector<float>& v) {
            float mean = 0.0f;
            for (float x : v) mean += x;
            mean /= static_cast<float>(v.size());
            float var = 0.0f;
            for (float x : v) { float d = x - mean; var += d * d; }
            return std::sqrt(var / static_cast<float>(v.size()));
        };
        const float std_c   = stddev(c_eps);
        const float std_eps = stddev(eps);
        if (std_eps > 0.0f && std_c > 0.0f) {
            const float phi = std_c / std_eps;
            const float r   = ctx.cfg_rescale;
            for (int j = 0; j < ctx.latent_size; ++j)
                eps[j] = (r * phi + (1.0f - r)) * eps[j];
        }
    }

    float eps_mn = 1e9f, eps_mx = -1e9f;
    for (float v : eps) { eps_mn = std::min(eps_mn, v); eps_mx = std::max(eps_mx, v); }
    Logger::info("  eps range: [" + std::to_string(eps_mn) + ", " + std::to_string(eps_mx) + "]");
    return eps;
}

} // namespace sd