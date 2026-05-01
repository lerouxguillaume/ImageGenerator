#pragma once

#include "../projects/Project.hpp"
#include "../portraits/PortraitGeneratorAi.hpp"

#include <atomic>
#include <filesystem>
#include <stop_token>
#include <string>
#include <vector>

struct GenerationReferenceInput {
    bool enabled = false;
    std::string imagePath;
    float structureStrength = 0.0f;
    int canvasWidth = 1;
    int canvasHeight = 1;
    std::filesystem::path cacheDir;
};

struct GenerationPostProcessSpec {
    bool assetMode = false;
    bool requiresTransparency = false;
    std::filesystem::path processedDir;
    AssetExportSpec exportSpec;
    GenerationReferenceInput reference;
};

struct GenerationJob {
    std::string prompt;
    std::string negativePrompt;
    std::string outputPath;
    GenerationParams params;
    std::string modelDir;
    GenerationPostProcessSpec postProcess;
};

struct GenerationProgress {
    std::atomic<int>* step = nullptr;
    std::atomic<int>* currentImage = nullptr;
};

struct GenerationResult {
    std::vector<std::string> rawPaths;
    bool referenceUsed = false;
};

struct CandidateRunJob {
    std::string prompt;
    std::string negativePrompt;
    std::string modelDir;
    GenerationParams baseParams;
    std::string runId;
    std::string patronPath;
    std::filesystem::path runPath;
    std::filesystem::path exploreRawDir;
    std::filesystem::path exploreProcessedDir;
    std::filesystem::path refineRawDir;
    std::filesystem::path refineProcessedDir;
    int exploreCount = 0;
    int candidateCount = 0;
    int refineVariants = 0;
    bool requiresTransparency = false;
    AssetExportSpec exportSpec;
    AssetSpec spec;
    std::string assetTypeId;
    float explorationStrength = 0.70f;
    float refinementStrength = 0.27f;
    float scoreThreshold = 150.0f;
};

struct CandidateRunResult {
    std::string runId;
    int explorationCount = 0;
    int selectedCount = 0;
    int refinementCount = 0;
};

class GenerationService {
public:
    GenerationResult run(const GenerationJob& job,
                         GenerationProgress progress,
                         std::stop_token stopToken) const;

    CandidateRunResult runCandidateRun(const CandidateRunJob& job,
                                       GenerationProgress progress,
                                       std::stop_token stopToken) const;
};
