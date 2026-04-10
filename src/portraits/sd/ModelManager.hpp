#pragma once
#include "SdTypes.hpp"
#include "../../config/AppConfig.hpp"
#include <memory>
#include <string>
#include <vector>

namespace sd {

// Owns a single loaded GenerationContext and reuses it when the model+LoRA
// configuration is unchanged between calls.
//
// Cache key: (modelDir, loras[]{path, scale})
// Cache miss: delegates to loadModels(), replaces the cached context.
// On every get() call, run_opts is reset so a previous SetTerminate() state
// from a cancelled run does not carry over.
class ModelManager {
public:
    // Returns a reference to a loaded (and potentially cached) GenerationContext.
    // The returned reference is valid until the next get() call that triggers a reload.
    GenerationContext& get(const ModelConfig&            cfg,
                           const std::string&            modelDir,
                           const std::vector<LoraEntry>& loras);

private:
    struct CacheKey {
        std::string            modelDir;
        std::vector<LoraEntry> loras;
    };

    static bool keysEqual(const CacheKey& a, const CacheKey& b);

    CacheKey                           key_;
    std::unique_ptr<GenerationContext> ctx_;
};

} // namespace sd
