#pragma once
#include <atomic>
#include <filesystem>
#include <stop_token>
#include <string>
#include <vector>
#include "../assets/AssetMetadata.hpp"
#include "../assets/CandidateScorer.hpp"
#include "../portraits/PortraitGeneratorAi.hpp"
#include "../projects/Project.hpp"

struct CandidateRunPipeline {
    std::string           prompt;
    std::string           negPrompt;
    std::string           modelDir;
    GenerationParams      baseParams;
    std::string           runId;
    std::string           patronPath;
    std::filesystem::path runPath;
    std::filesystem::path exploreRawDir;
    std::filesystem::path exploreProcessedDir;
    std::filesystem::path refineRawDir;
    std::filesystem::path refineProcessedDir;
    int                   exploreCount         = 0;
    int                   candidateCount       = 0;
    int                   refineVariants       = 0;
    bool                  requiresTransparency = false;
    AssetExportSpec       exportSpec;
    AssetSpec             spec;
    std::string           assetTypeId;
    float                 explorationStrength  = 0.70f;
    float                 refinementStrength   = 0.27f;
    float                 scoreThreshold       = 150.0f;
    std::atomic<int>*     step   = nullptr;
    std::atomic<int>*     imgNum = nullptr;

    std::vector<CandidateScore> explore(std::stop_token st);
    std::vector<CandidateScore> selectCandidates(std::vector<CandidateScore> scores) const;
    std::vector<CandidateScore> refine(const std::vector<CandidateScore>& candidates,
                                       std::stop_token st);
    void writeManifest(const std::vector<CandidateScore>& exploration,
                       const std::vector<CandidateScore>& refinement) const;

private:
    static std::string nthPath(const std::filesystem::path& firstPath, int index);
    std::string        processOutput(const std::string& rawPath,
                                     const std::filesystem::path& processedDir,
                                     const std::string& stage,
                                     bool refinementUsed,
                                     const std::string& refinementSource) const;
};
