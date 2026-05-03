#include "GenerationService.hpp"

#include "../assets/GeneratedAssetProcessor.hpp"
#include "../assets/ReferenceNormalizer.hpp"
#include "../managers/Logger.hpp"
#include "CandidateRunPipeline.hpp"

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

void GenerationService::run(const GenerationJob& job,
                            GenerationProgress progress,
                            GenerationCallbacks callbacks,
                            std::stop_token stopToken) const {
    try {
        GenerationParams effectiveParams = job.params;
        const bool referenceUsed = applyReferenceIfAvailable(effectiveParams, job.postProcess);

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
        result.referenceUsed = referenceUsed;
        if (progress.stage) progress.stage->store(GenerationStage::PostProcessing);
        for (int i = 1; i <= effectiveParams.numImages; ++i) {
            const std::string rawPath = nthImagePath(job.outputPath, i);
            processGeneratedOutput(rawPath, job.postProcess, referenceUsed);
            result.rawPaths.push_back(rawPath);
        }
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

void GenerationService::runCandidateRun(const CandidateRunJob& job,
                                        GenerationProgress progress,
                                        CandidateRunCallbacks callbacks,
                                        std::stop_token stopToken) const {
    try {
        CandidateRunPipeline pipeline;
        pipeline.prompt = job.prompt;
        pipeline.negPrompt = job.negativePrompt;
        pipeline.modelDir = job.modelDir;
        pipeline.baseParams = job.baseParams;
        pipeline.runId = job.runId;
        pipeline.patronPath = job.patronPath;
        pipeline.runPath = job.runPath;
        pipeline.exploreRawDir = job.exploreRawDir;
        pipeline.exploreProcessedDir = job.exploreProcessedDir;
        pipeline.refineRawDir = job.refineRawDir;
        pipeline.refineProcessedDir = job.refineProcessedDir;
        pipeline.exploreCount = job.exploreCount;
        pipeline.candidateCount = job.candidateCount;
        pipeline.refineVariants = job.refineVariants;
        pipeline.requiresTransparency = job.requiresTransparency;
        pipeline.exportSpec = job.exportSpec;
        pipeline.spec = job.spec;
        pipeline.assetTypeId = job.assetTypeId;
        pipeline.explorationStrength = job.explorationStrength;
        pipeline.refinementStrength = job.refinementStrength;
        pipeline.scoreThreshold = job.scoreThreshold;
        pipeline.step   = progress.step;
        pipeline.imgNum = progress.currentImage;
        pipeline.stage  = progress.stage;

        auto exploration = pipeline.explore(stopToken);
        auto selected = pipeline.selectCandidates(exploration);
        const auto refinement = pipeline.refine(selected, stopToken);
        pipeline.writeManifest(exploration, refinement);
        if (progress.stage) progress.stage->store(GenerationStage::Done);

        CandidateRunResult result;
        result.runId = job.runId;
        result.explorationCount = static_cast<int>(exploration.size());
        result.selectedCount = static_cast<int>(selected.size());
        result.refinementCount = static_cast<int>(refinement.size());
        if (callbacks.onResult) callbacks.onResult(std::move(result));
    } catch (const std::exception& e) {
        Logger::error("Candidate run failed: " + std::string(e.what()));
        if (callbacks.onError) callbacks.onError(e.what());
    } catch (...) {
        Logger::error("Candidate run failed: unknown error");
        if (callbacks.onError) callbacks.onError("Unknown error during candidate run. See log for details.");
    }
}
