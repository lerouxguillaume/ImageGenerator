#include "GenerationService.hpp"

#include "../managers/Logger.hpp"

namespace {

std::string nthImagePath(const std::string& firstPath, int index) {
    if (index <= 1) return firstPath;
    const auto dot = firstPath.rfind('.');
    if (dot == std::string::npos)
        return firstPath + "_" + std::to_string(index);
    return firstPath.substr(0, dot) + "_" + std::to_string(index) + firstPath.substr(dot);
}

} // namespace

void GenerationService::run(const GenerationJob& job,
                            GenerationProgress progress,
                            GenerationCallbacks callbacks,
                            std::stop_token stopToken) const {
    try {
        GenerationParams effectiveParams = job.params;
        if (!job.loraCompatible && !effectiveParams.loras.empty()) {
            Logger::info("LoRA entries cleared: model does not support LoRA injection");
            effectiveParams.loras.clear();
        }

        PortraitGeneratorAi::generateFromPrompt(
            job.prompt,
            job.negativePrompt,
            job.outputPath,
            effectiveParams,
            job.modelDir,
            progress.step,
            progress.currentImage,
            std::move(stopToken),
            progress.stage);

        GenerationResult result;
        for (int i = 1; i <= effectiveParams.numImages; ++i)
            result.rawPaths.push_back(nthImagePath(job.outputPath, i));

        if (progress.stage) progress.stage->store(GenerationStage::Done);
        if (callbacks.onResult) callbacks.onResult(std::move(result));
    } catch (const std::exception& e) {
        Logger::error("Generation failed: " + std::string(e.what()));
        if (callbacks.onError) callbacks.onError(e.what());
    } catch (...) {
        Logger::error("Generation failed: unknown error");
        if (callbacks.onError) callbacks.onError("Unknown error during generation. See log for details.");
    }
}
