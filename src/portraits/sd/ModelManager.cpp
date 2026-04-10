#include "ModelManager.hpp"
#include "SdLoader.hpp"
#include "../../managers/Logger.hpp"

namespace sd {

bool ModelManager::keysEqual(const CacheKey& a, const CacheKey& b) {
    if (a.modelDir != b.modelDir) return false;
    if (a.loras.size() != b.loras.size()) return false;
    for (size_t i = 0; i < a.loras.size(); ++i)
        if (!(a.loras[i] == b.loras[i])) return false;
    return true;
}

GenerationContext& ModelManager::get(const ModelConfig&            cfg,
                                      const std::string&            modelDir,
                                      const std::vector<LoraEntry>& loras) {
    CacheKey newKey{modelDir, loras};

    if (ctx_ && keysEqual(key_, newKey)) {
        Logger::info("ModelManager: cache hit — reusing loaded sessions.");
    } else {
        Logger::info("ModelManager: cache miss — loading models.");
        ctx_ = std::make_unique<GenerationContext>(loadModels(cfg, modelDir, loras));
        key_ = std::move(newKey);
    }

    // Always reset run_opts so a SetTerminate() from a previous cancelled run
    // does not affect the new run.
    ctx_->run_opts = Ort::RunOptions{};
    return *ctx_;
}

} // namespace sd
