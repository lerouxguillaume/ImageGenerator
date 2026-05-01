#include "GenerationService.hpp"

#include "../assets/GeneratedAssetProcessor.hpp"
#include "../assets/ReferenceNormalizer.hpp"
#include "../managers/Logger.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {

std::string nthImagePath(const std::string& firstPath, int index) {
    if (index <= 1) return firstPath;
    const auto dot = firstPath.rfind('.');
    if (dot == std::string::npos)
        return firstPath + "_" + std::to_string(index);
    return firstPath.substr(0, dot) + "_" + std::to_string(index) + firstPath.substr(dot);
}

bool applyReferenceIfAvailable(GenerationParams& params,
                               const GenerationPostProcessSpec& postProcess) {
    const auto& ref = postProcess.reference;
    if (!postProcess.assetMode
        || !ref.enabled
        || !params.initImagePath.empty()
        || ref.imagePath.empty()
        || !std::filesystem::exists(ref.imagePath)) {
        if (postProcess.assetMode && ref.enabled && !ref.imagePath.empty())
            Logger::info("project asset reference requested but unavailable, falling back to txt2img: " + ref.imagePath);
        return false;
    }

    const cv::Mat normalizedRef =
        ReferenceNormalizer::normalizeToCanvas(ref.imagePath, ref.canvasWidth, ref.canvasHeight);
    const std::filesystem::path refPath = ref.cacheDir / "normalized_reference.png";
    cv::Mat bgraRef;
    cv::cvtColor(normalizedRef, bgraRef, cv::COLOR_RGBA2BGRA);
    cv::imwrite(refPath.string(), bgraRef);

    params.initImagePath = refPath.string();
    params.strength = ref.structureStrength;
    Logger::info("project asset reference img2img enabled: " + ref.imagePath);
    return true;
}

void processGeneratedOutput(const std::string& rawPath,
                            const GenerationPostProcessSpec& postProcess,
                            bool referenceUsed) {
    GeneratedAssetProcessor::Request req;
    req.rawPath = rawPath;
    req.processedDir = postProcess.processedDir;
    req.exportSpec = postProcess.exportSpec;
    req.assetMode = postProcess.assetMode;
    req.requiresTransparency = postProcess.requiresTransparency;
    req.metadata.referenceUsed = referenceUsed;
    req.metadata.referenceImage = postProcess.reference.imagePath;
    req.metadata.structureStrength = referenceUsed ? postProcess.reference.structureStrength : 0.0f;
    GeneratedAssetProcessor::process(req);
}

} // namespace

GenerationResult GenerationService::run(const GenerationJob& job,
                                        std::atomic<int>* progressStep,
                                        std::atomic<int>* currentImage,
                                        std::stop_token stopToken) const {
    GenerationParams effectiveParams = job.params;
    const bool referenceUsed = applyReferenceIfAvailable(effectiveParams, job.postProcess);

    PortraitGeneratorAi::generateFromPrompt(
        job.prompt,
        job.negativePrompt,
        job.outputPath,
        effectiveParams,
        job.modelDir,
        progressStep,
        currentImage,
        std::move(stopToken));

    GenerationResult result;
    result.referenceUsed = referenceUsed;
    for (int i = 1; i <= effectiveParams.numImages; ++i) {
        const std::string rawPath = nthImagePath(job.outputPath, i);
        processGeneratedOutput(rawPath, job.postProcess, referenceUsed);
        result.rawPaths.push_back(rawPath);
    }
    return result;
}
