#pragma once

#include "../projects/Project.hpp"
#include <filesystem>
#include <string>
#include <vector>

class AssetArtifactStore {
public:
    struct CandidateRunLayout {
        std::string runId;
        std::filesystem::path runPath;
        std::filesystem::path exploreRawDir;
        std::filesystem::path exploreProcessedDir;
        std::filesystem::path refineRawDir;
        std::filesystem::path refineProcessedDir;

        std::filesystem::path manifestPath() const;
    };

    struct GalleryEntry {
        std::filesystem::path path;
        std::string filename;
        std::filesystem::file_time_type modified;
    };

    AssetArtifactStore(std::filesystem::path outputRoot, std::string outputSubpath = {});
    AssetArtifactStore(std::filesystem::path outputRoot, const ResolvedProjectContext& context);

    bool assetMode() const;
    const std::filesystem::path& baseDir() const;

    std::filesystem::path rawDir() const;
    std::filesystem::path processedDir() const;
    std::filesystem::path referenceCacheDir() const;
    std::filesystem::path patronPath() const;
    std::filesystem::path rawImagePath(const std::string& filename) const;
    std::filesystem::path processedImagePath(const std::string& filename) const;
    std::filesystem::path displayedImagePathForRaw(const std::filesystem::path& rawPath) const;

    void ensureStandardDirs() const;

    std::filesystem::path runsDir() const;
    CandidateRunLayout candidateRunLayout(const std::string& runId) const;
    void ensureCandidateRunDirs(const CandidateRunLayout& layout) const;
    std::filesystem::path latestCandidateRunDir() const;
    std::filesystem::path latestCandidateGalleryDir() const;

    std::filesystem::path galleryDir(GenerationWorkflow workflow) const;
    std::vector<GalleryEntry> listGalleryImages(GenerationWorkflow workflow) const;

    static std::filesystem::path metadataPathFor(const std::filesystem::path& imagePath);
    static std::filesystem::path transparentPathFor(const std::filesystem::path& imagePath);
    static std::filesystem::path rawPathForProcessed(const std::filesystem::path& processedPath);
    static std::filesystem::path processedPathForRaw(const std::filesystem::path& rawPath);
    static bool isTransparentDerivative(const std::filesystem::path& path);
    static bool isGalleryImageFile(const std::filesystem::path& path);

private:
    std::filesystem::path outputRoot_;
    std::string outputSubpath_;
    std::filesystem::path baseDir_;
    bool assetMode_ = false;
};
