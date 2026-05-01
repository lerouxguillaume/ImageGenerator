#include "CandidateRunPipeline.hpp"
#include "../assets/AssetPostProcessor.hpp"
#include "../managers/Logger.hpp"
#include "../postprocess/AlphaCutout.hpp"
#include "../portraits/PortraitGeneratorAi.hpp"
#include <SFML/Graphics.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>

// ── Path helpers ──────────────────────────────────────────────────────────────

static std::string metadataPathFor(const std::string& imagePath) {
    const auto dot = imagePath.rfind('.');
    if (dot == std::string::npos) return imagePath + ".json";
    return imagePath.substr(0, dot) + ".json";
}

static std::filesystem::path rawPathForProcessed(const std::filesystem::path& p) {
    if (p.parent_path().filename().string() == "processed")
        return p.parent_path().parent_path() / "raw" / p.filename();
    return p;
}

static std::filesystem::path processedPathForRaw(const std::filesystem::path& p) {
    if (p.parent_path().filename().string() == "raw")
        return p.parent_path().parent_path() / "processed" / p.filename();
    return p;
}

// ── Scoring helpers ───────────────────────────────────────────────────────────

static std::optional<OccupiedBounds> computeOpaqueBoundsForScore(const sf::Image& img) {
    const unsigned w = img.getSize().x;
    const unsigned h = img.getSize().y;
    if (w == 0 || h == 0) return std::nullopt;
    bool found = false;
    unsigned minX = w, minY = h, maxX = 0, maxY = 0;
    for (unsigned py = 0; py < h; ++py) {
        for (unsigned px = 0; px < w; ++px) {
            if (img.getPixel(px, py).a < 128) continue;
            found = true;
            minX = std::min(minX, px);
            minY = std::min(minY, py);
            maxX = std::max(maxX, px);
            maxY = std::max(maxY, py);
        }
    }
    if (!found) return std::nullopt;
    return OccupiedBounds{
        static_cast<int>(minX), static_cast<int>(minY),
        static_cast<int>(maxX - minX + 1), static_cast<int>(maxY - minY + 1)
    };
}

static bool hasAnyTransparency(const sf::Image& img) {
    const unsigned w = img.getSize().x;
    const unsigned h = img.getSize().y;
    for (unsigned py = 0; py < h; ++py)
        for (unsigned px = 0; px < w; ++px)
            if (img.getPixel(px, py).a < 255) return true;
    return false;
}

static float computeFillRatioForScore(const sf::Image& img) {
    const unsigned w = img.getSize().x;
    const unsigned h = img.getSize().y;
    if (w == 0 || h == 0) return 0.0f;
    unsigned opaque = 0;
    for (unsigned py = 0; py < h; ++py)
        for (unsigned px = 0; px < w; ++px)
            if (img.getPixel(px, py).a >= 128) ++opaque;
    return static_cast<float>(opaque) / static_cast<float>(w * h);
}

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
    sf::Image sfRaw;
    if (!sfRaw.loadFromFile(rawPath)) return {};
    if (requiresTransparency)
        sfRaw = AlphaCutout::removeBackground(sfRaw);

    std::vector<sf::Uint8> pixels(sfRaw.getPixelsPtr(),
                                  sfRaw.getPixelsPtr() + sfRaw.getSize().x * sfRaw.getSize().y * 4);
    cv::Mat rgba(static_cast<int>(sfRaw.getSize().y), static_cast<int>(sfRaw.getSize().x),
                 CV_8UC4, pixels.data());
    cv::Mat bgra;
    cv::cvtColor(rgba, bgra, cv::COLOR_RGBA2BGRA);

    const AssetProcessResult result = AssetPostProcessor::process(bgra, exportSpec);
    const std::filesystem::path processedPath = processedDir / std::filesystem::path(rawPath).filename();
    cv::imwrite(processedPath.string(), result.image);

    nlohmann::json metaJson = toJson(result, exportSpec);
    metaJson["candidateRunId"]     = runId;
    metaJson["candidateStage"]     = stage;
    metaJson["referenceUsed"]      = false;
    metaJson["referenceImage"]     = std::string{};
    metaJson["structureStrength"]  = 0.0f;
    metaJson["refinementUsed"]     = refinementUsed;
    metaJson["refinementSource"]   = refinementSource;
    metaJson["refinementStrength"] = refinementUsed ? refinementStrength : 0.0f;
    std::ofstream meta(metadataPathFor(processedPath.string()));
    meta << metaJson.dump(4);
    return processedPath.string();
}

CandidateScore CandidateRunPipeline::scoreCandidate(const std::string& imagePath,
                                                     const AssetSpec& spec, int index) {
    CandidateScore candidate;
    candidate.index = index;
    const std::filesystem::path inputPath(imagePath);
    const bool inputIsRaw = inputPath.parent_path().filename().string() == "raw";
    candidate.rawPath       = inputIsRaw ? inputPath.string()
                                         : rawPathForProcessed(inputPath).string();
    candidate.processedPath = inputIsRaw ? processedPathForRaw(inputPath).string()
                                         : inputPath.string();

    sf::Image img;
    if (!img.loadFromFile(candidate.rawPath) && !img.loadFromFile(candidate.processedPath))
        return candidate;

    const unsigned w = img.getSize().x;
    const unsigned h = img.getSize().y;
    if (w == 0 || h == 0) return candidate;

    if (spec.requiresTransparency)
        img = AlphaCutout::removeBackground(img);

    float score = 0.0f;
    if ((spec.canvasWidth > 0 && static_cast<int>(w) != spec.canvasWidth)
        || (spec.canvasHeight > 0 && static_cast<int>(h) != spec.canvasHeight))
        score += 1000.0f;

    if (spec.requiresTransparency && !hasAnyTransparency(img))
        score += 500.0f;

    const auto bounds = computeOpaqueBoundsForScore(img);
    if (!bounds) { candidate.score = score + 1000.0f; return candidate; }

    const float fillRatio = computeFillRatioForScore(img);
    float fillPenalty = 0.f;
    if (fillRatio < spec.minFillRatio)
        fillPenalty = (spec.minFillRatio - fillRatio) / std::max(spec.minFillRatio, 0.01f) * 300.f;
    else if (fillRatio > spec.maxFillRatio)
        fillPenalty = (fillRatio - spec.maxFillRatio) / std::max(1.0f - spec.maxFillRatio, 0.01f) * 300.f;
    else {
        const float range = std::max(spec.maxFillRatio - spec.minFillRatio, 0.01f);
        fillPenalty = std::abs(fillRatio - spec.targetFillRatio) / range * 100.f;
    }
    score += std::min(fillPenalty, 300.f);

    if (spec.expectedBounds.w > 0 && spec.expectedBounds.h > 0) {
        const float boundsError =
            static_cast<float>(std::abs(bounds->x - spec.expectedBounds.x)
            + std::abs(bounds->y - spec.expectedBounds.y)
            + std::abs(bounds->w - spec.expectedBounds.w)
            + std::abs(bounds->h - spec.expectedBounds.h));
        const float maxErr = static_cast<float>(spec.expectedBounds.w + spec.expectedBounds.h);
        score += std::min(boundsError / std::max(maxErr, 1.f) * 300.f, 300.f);
    }

    if (spec.anchor.x != 0 || spec.anchor.y != 0) {
        const int anchorX = bounds->x + bounds->w / 2;
        const int anchorY = bounds->y + bounds->h;
        const float dx = static_cast<float>(anchorX - spec.anchor.x);
        const float dy = static_cast<float>(anchorY - spec.anchor.y);
        const float dist = std::sqrt(dx * dx + dy * dy);
        const float maxDist = static_cast<float>(spec.canvasWidth + spec.canvasHeight) / 4.f;
        score += std::min(dist / std::max(maxDist, 1.f) * 200.f, 200.f);
    }

    candidate.score = score;
    candidate.valid = true;
    return candidate;
}

std::vector<CandidateScore> CandidateRunPipeline::explore(std::stop_token st) {
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
        const auto score = scoreCandidate(processedPath, spec, i - 1);
        if (score.valid) scores.push_back(score);
    }
    return scores;
}

std::vector<CandidateScore> CandidateRunPipeline::selectCandidates(
    std::vector<CandidateScore> scores) const {
    std::sort(scores.begin(), scores.end(),
              [](const auto& a, const auto& b) { return a.score < b.score; });
    if (static_cast<int>(scores.size()) > candidateCount)
        scores.resize(static_cast<size_t>(candidateCount));
    return scores;
}

std::vector<CandidateScore> CandidateRunPipeline::refine(
    const std::vector<CandidateScore>& candidates, std::stop_token st) {
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
            const auto score = scoreCandidate(processedPath, spec,
                                              static_cast<int>(refined.size()));
            if (score.valid) refined.push_back(score);
        }
    }
    std::sort(refined.begin(), refined.end(),
              [](const auto& a, const auto& b) { return a.score < b.score; });
    return refined;
}

void CandidateRunPipeline::writeManifest(const std::vector<CandidateScore>& exploration,
                                          const std::vector<CandidateScore>& refinement) const {
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
    std::ofstream manifestFile(runPath / "manifest.json");
    manifestFile << manifest.dump(4);
}
