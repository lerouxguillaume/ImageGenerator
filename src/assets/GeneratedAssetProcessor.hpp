#pragma once

#include "AssetMetadata.hpp"
#include <filesystem>
#include <string>

namespace GeneratedAssetProcessor {

struct Metadata {
    bool referenceUsed = false;
    std::string referenceImage;
    float structureStrength = 0.0f;

    bool refinementUsed = false;
    std::string refinementSource;
    float refinementStrength = 0.0f;

    std::string candidateRunId;
    std::string candidateStage;
};

struct Request {
    std::filesystem::path rawPath;
    std::filesystem::path processedDir;
    AssetExportSpec exportSpec;
    bool assetMode = false;
    bool requiresTransparency = false;
    Metadata metadata;
};

struct Result {
    std::filesystem::path processedPath;
    std::filesystem::path transparentPath;
    bool wroteProcessed = false;
    bool wroteTransparent = false;
};

Result process(const Request& request);

}
