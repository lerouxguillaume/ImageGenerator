#include "AssetArtifactStore.hpp"
#include <algorithm>
#include <cctype>
#include <utility>

std::filesystem::path AssetArtifactStore::CandidateRunLayout::manifestPath() const {
    return runPath / "manifest.json";
}

AssetArtifactStore::AssetArtifactStore(std::filesystem::path outputRoot, std::string outputSubpath)
    : outputRoot_(std::move(outputRoot))
    , outputSubpath_(std::move(outputSubpath))
    , baseDir_(outputRoot_)
    , assetMode_(!outputSubpath_.empty())
{
    if (!outputSubpath_.empty())
        baseDir_ /= outputSubpath_;
}

AssetArtifactStore::AssetArtifactStore(std::filesystem::path outputRoot,
                                       const ResolvedProjectContext& context)
    : AssetArtifactStore(std::move(outputRoot), context.outputSubpath)
{
    assetMode_ = !context.empty();
}

bool AssetArtifactStore::assetMode() const {
    return assetMode_;
}

const std::filesystem::path& AssetArtifactStore::baseDir() const {
    return baseDir_;
}

std::filesystem::path AssetArtifactStore::rawDir() const {
    return assetMode() ? baseDir_ / "raw" : baseDir_;
}

std::filesystem::path AssetArtifactStore::processedDir() const {
    return assetMode() ? baseDir_ / "processed" : baseDir_;
}

std::filesystem::path AssetArtifactStore::referenceCacheDir() const {
    return baseDir_ / ".reference_cache";
}

std::filesystem::path AssetArtifactStore::patronPath() const {
    return baseDir_ / "patron.png";
}

std::filesystem::path AssetArtifactStore::rawImagePath(const std::string& filename) const {
    return rawDir() / filename;
}

std::filesystem::path AssetArtifactStore::processedImagePath(const std::string& filename) const {
    return processedDir() / filename;
}

std::filesystem::path AssetArtifactStore::displayedImagePathForRaw(
    const std::filesystem::path& rawPath) const {
    return assetMode() ? processedPathForRaw(rawPath) : rawPath;
}

void AssetArtifactStore::ensureStandardDirs() const {
    std::filesystem::create_directories(rawDir());
    if (assetMode()) {
        std::filesystem::create_directories(processedDir());
        std::filesystem::create_directories(referenceCacheDir());
    }
}

std::filesystem::path AssetArtifactStore::runsDir() const {
    return baseDir_ / "runs";
}

AssetArtifactStore::CandidateRunLayout AssetArtifactStore::candidateRunLayout(
    const std::string& runId) const {
    CandidateRunLayout layout;
    layout.runId = runId;
    layout.runPath = runsDir() / runId;
    layout.exploreRawDir = layout.runPath / "explore" / "raw";
    layout.exploreProcessedDir = layout.runPath / "explore" / "processed";
    layout.refineRawDir = layout.runPath / "refine" / "raw";
    layout.refineProcessedDir = layout.runPath / "refine" / "processed";
    return layout;
}

void AssetArtifactStore::ensureCandidateRunDirs(const CandidateRunLayout& layout) const {
    std::filesystem::create_directories(layout.exploreRawDir);
    std::filesystem::create_directories(layout.exploreProcessedDir);
    std::filesystem::create_directories(layout.refineRawDir);
    std::filesystem::create_directories(layout.refineProcessedDir);
}

std::filesystem::path AssetArtifactStore::latestCandidateRunDir() const {
    const auto dir = runsDir();
    std::error_code ec;
    if (!std::filesystem::exists(dir, ec)) return {};

    std::filesystem::path latest;
    std::filesystem::file_time_type latestTime{};
    for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
        if (!entry.is_directory()) continue;
        const auto t = entry.last_write_time(ec);
        if (latest.empty() || t > latestTime) {
            latest = entry.path();
            latestTime = t;
        }
    }
    return latest;
}

std::filesystem::path AssetArtifactStore::latestCandidateGalleryDir() const {
    const std::filesystem::path latestRun = latestCandidateRunDir();
    if (latestRun.empty()) return {};

    std::error_code ec;
    const std::filesystem::path refineDir = latestRun / "refine" / "processed";
    if (std::filesystem::exists(refineDir, ec)) {
        for (const auto& entry : std::filesystem::directory_iterator(refineDir, ec)) {
            if (entry.is_regular_file()
                && isGalleryImageFile(entry.path())
                && !isTransparentDerivative(entry.path())) {
                return refineDir;
            }
        }
    }
    return latestRun / "explore" / "processed";
}

std::filesystem::path AssetArtifactStore::galleryDir(GenerationWorkflow workflow) const {
    if (workflow == GenerationWorkflow::CandidateRun) {
        const auto candidateDir = latestCandidateGalleryDir();
        if (!candidateDir.empty()) return candidateDir;
    }
    return assetMode() ? processedDir() : baseDir_;
}

std::vector<AssetArtifactStore::GalleryEntry> AssetArtifactStore::listGalleryImages(
    GenerationWorkflow workflow) const {
    std::vector<GalleryEntry> images;
    const std::filesystem::path dir = galleryDir(workflow);
    std::error_code ec;
    if (!std::filesystem::exists(dir, ec)) return images;

    for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
        if (!entry.is_regular_file()) continue;
        if (!isGalleryImageFile(entry.path())) continue;
        if (isTransparentDerivative(entry.path())) continue;
        images.push_back({
            entry.path(),
            entry.path().filename().string(),
            entry.last_write_time(ec)
        });
    }
    return images;
}

std::filesystem::path AssetArtifactStore::metadataPathFor(
    const std::filesystem::path& imagePath) {
    const auto ext = imagePath.extension();
    if (ext.empty()) return imagePath.string() + ".json";
    auto p = imagePath;
    p.replace_extension(".json");
    return p;
}

std::filesystem::path AssetArtifactStore::transparentPathFor(
    const std::filesystem::path& imagePath) {
    const auto ext = imagePath.extension();
    if (ext.empty()) return imagePath.string() + "_t";
    return imagePath.parent_path() / (imagePath.stem().string() + "_t" + ext.string());
}

std::filesystem::path AssetArtifactStore::rawPathForProcessed(
    const std::filesystem::path& processedPath) {
    if (processedPath.parent_path().filename().string() == "processed")
        return processedPath.parent_path().parent_path() / "raw" / processedPath.filename();
    return processedPath;
}

std::filesystem::path AssetArtifactStore::processedPathForRaw(
    const std::filesystem::path& rawPath) {
    if (rawPath.parent_path().filename().string() == "raw")
        return rawPath.parent_path().parent_path() / "processed" / rawPath.filename();
    return rawPath;
}

bool AssetArtifactStore::isTransparentDerivative(const std::filesystem::path& path) {
    const std::string stem = path.stem().string();
    return stem.size() >= 2 && stem.substr(stem.size() - 2) == "_t";
}

bool AssetArtifactStore::isGalleryImageFile(const std::filesystem::path& path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".webp";
}
