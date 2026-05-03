#include "CandidateRunPipeline.hpp"
#include "../assets/AssetArtifactStore.hpp"
#include "../assets/GeneratedAssetProcessor.hpp"
#include "../managers/Logger.hpp"
#include "../portraits/PortraitGeneratorAi.hpp"
#include <algorithm>
#include <fstream>
#include <nlohmann/json.hpp>

// ── CandidateRunPipeline ──────────────────────────────────────────────────────

std::string CandidateRunPipeline::nthPath(const std::filesystem::path& firstPath, int index) {
    if (index == 1) return firstPath.string();
    return (firstPath.parent_path()
            / (firstPath.stem().string() + "_" + std::to_string(index) + firstPath.extension().string()))
               .string();
}

std::string CandidateRunPipeline::processOutput(const std::string& rawPath,
                                                 const std::filesystem::path& processedDir,
                                                 const std::string& stage,
                                                 bool refinementUsed,
                                                 const std::string& refinementSource) const {
    GeneratedAssetProcessor::Request req;
    req.rawPath = rawPath;
    req.processedDir = processedDir;
    req.exportSpec = exportSpec;
    req.assetMode = true;
    req.requiresTransparency = requiresTransparency;
    req.metadata.candidateRunId = runId;
    req.metadata.candidateStage = stage;
    req.metadata.refinementUsed = refinementUsed;
    req.metadata.refinementSource = refinementSource;
    req.metadata.refinementStrength = refinementUsed ? refinementStrength : 0.0f;

    const auto result = GeneratedAssetProcessor::process(req);
    return result.wroteProcessed ? result.processedPath.string() : std::string{};
}

std::vector<CandidateScore> CandidateRunPipeline::explore(std::stop_token st) {
    if (stage) stage->store(GenerationStage::Exploring);
    const std::filesystem::path exploreFirst = exploreRawDir / "explore.png";
    GenerationParams exploreParams = baseParams;
    exploreParams.numImages = exploreCount;
    if (!patronPath.empty()) {
        exploreParams.initImagePath = patronPath;
        exploreParams.strength      = explorationStrength;
    } else {
        exploreParams.initImagePath.clear();
    }
    PortraitGeneratorAi::generateFromPrompt(
        prompt, negPrompt, exploreFirst.string(), exploreParams, modelDir, step, imgNum, st);

    std::vector<CandidateScore> scores;
    for (int i = 1; i <= exploreCount; ++i) {
        const std::string rawPath       = nthPath(exploreFirst, i);
        const std::string processedPath = processOutput(rawPath, exploreProcessedDir, "explore", false, {});
        if (processedPath.empty()) continue;
        const auto score = CandidateScorer::scoreCandidate(
            processedPath, spec, i - 1, exportSpec.alphaCutout);
        if (score.valid) scores.push_back(score);
    }
    return scores;
}

std::vector<CandidateScore> CandidateRunPipeline::selectCandidates(
    std::vector<CandidateScore> scores) const {
    if (stage) stage->store(GenerationStage::Scoring);
    std::sort(scores.begin(), scores.end(),
              [](const auto& a, const auto& b) { return a.score < b.score; });
    if (static_cast<int>(scores.size()) > candidateCount)
        scores.resize(static_cast<size_t>(candidateCount));
    return scores;
}

std::vector<CandidateScore> CandidateRunPipeline::refine(
    const std::vector<CandidateScore>& candidates, std::stop_token st) {
    if (stage) stage->store(GenerationStage::Refining);
    std::vector<CandidateScore> refined;
    int candidateIndex = 0;
    for (const auto& candidate : candidates) {
        if (st.stop_requested()) break;
        ++candidateIndex;
        const std::filesystem::path refineFirst =
            refineRawDir / ("ref_" + std::to_string(candidateIndex) + ".png");
        GenerationParams refineParams = baseParams;
        refineParams.numImages     = refineVariants;
        refineParams.initImagePath = candidate.rawPath;
        refineParams.strength      = refinementStrength;
        PortraitGeneratorAi::generateFromPrompt(
            prompt, negPrompt, refineFirst.string(), refineParams, modelDir, step, imgNum, st);

        for (int i = 1; i <= refineVariants; ++i) {
            const std::string rawPath       = nthPath(refineFirst, i);
            const std::string processedPath =
                processOutput(rawPath, refineProcessedDir, "refine", true, candidate.rawPath);
            if (processedPath.empty()) continue;
            const auto score = CandidateScorer::scoreCandidate(
                processedPath, spec, static_cast<int>(refined.size()), exportSpec.alphaCutout);
            if (score.valid) refined.push_back(score);
        }
    }
    std::sort(refined.begin(), refined.end(),
              [](const auto& a, const auto& b) { return a.score < b.score; });
    return refined;
}

void CandidateRunPipeline::writeManifest(const std::vector<CandidateScore>& exploration,
                                          const std::vector<CandidateScore>& refinement) const {
    if (stage) stage->store(GenerationStage::WritingManifest);
    nlohmann::json manifest;
    manifest["runId"]          = runId;
    manifest["assetTypeId"]    = assetTypeId;
    manifest["scoreThreshold"] = scoreThreshold;
    manifest["exploration"]    = nlohmann::json::array();
    for (const auto& score : exploration) {
        manifest["exploration"].push_back({
            {"rawPath",          score.rawPath},
            {"processedPath",    score.processedPath},
            {"correctnessScore", score.score},
            {"status",           score.score <= scoreThreshold ? "ok" : "candidate"}
        });
    }
    manifest["proposals"] = nlohmann::json::array();
    for (int i = 0; i < static_cast<int>(refinement.size()); ++i) {
        const auto& score = refinement[static_cast<size_t>(i)];
        manifest["proposals"].push_back({
            {"rawPath",          score.rawPath},
            {"processedPath",    score.processedPath},
            {"correctnessScore", score.score},
            {"status",           i == 0 ? "best" : (score.score <= scoreThreshold ? "ok" : "near")}
        });
    }
    AssetArtifactStore::CandidateRunLayout layout;
    layout.runId = runId;
    layout.runPath = runPath;
    std::ofstream manifestFile(layout.manifestPath());
    manifestFile << manifest.dump(4);
}
