// parseLoraLayers: groups safetensors keys into (down, up, alpha) triplets.
#include "SdOnnxPatcher.hpp"

namespace sd {

static bool endsWith(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

ParsedLora parseLoraLayers(const SafetensorsMap& lora) {
    ParsedLora result;

    for (const auto& [key, tensor] : lora) {
        std::string body;
        // Strip known kohya LoRA prefixes.
        // lora_unet_  → UNet layers
        // lora_te_    → text encoder 1 (SD 1.5 and SDXL encoder-1 / CLIP-L)
        // lora_te2_   → text encoder 2 (SDXL OpenCLIP-G)
        if      (key.rfind("lora_unet_", 0) == 0) body = key.substr(10);
        else if (key.rfind("lora_te2_",  0) == 0) body = key.substr(9);
        else if (key.rfind("lora_te_",   0) == 0) body = key.substr(8);
        else continue;

        if (endsWith(body, ".lora_down.weight"))
            result.layers[body.substr(0, body.size() - 17)].down = &tensor;
        else if (endsWith(body, ".lora_up.weight"))
            result.layers[body.substr(0, body.size() - 15)].up = &tensor;
        else if (endsWith(body, ".alpha") && !tensor.data.empty())
            result.layers[body.substr(0, body.size() - 6)].alpha = tensor.data[0];
    }

    return result;
}

} // namespace sd
