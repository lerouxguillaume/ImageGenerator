#include "CandidateScorer.hpp"
#include "AssetArtifactStore.hpp"
#include "../postprocess/AlphaCutout.hpp"
#include <SFML/Graphics/Image.hpp>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <optional>

namespace {

std::optional<OccupiedBounds> computeOpaqueBoundsForScore(const sf::Image& img) {
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

bool hasAnyTransparency(const sf::Image& img) {
    const unsigned w = img.getSize().x;
    const unsigned h = img.getSize().y;
    for (unsigned py = 0; py < h; ++py)
        for (unsigned px = 0; px < w; ++px)
            if (img.getPixel(px, py).a < 255) return true;
    return false;
}

float computeFillRatioForScore(const sf::Image& img) {
    const unsigned w = img.getSize().x;
    const unsigned h = img.getSize().y;
    if (w == 0 || h == 0) return 0.0f;
    unsigned opaque = 0;
    for (unsigned py = 0; py < h; ++py)
        for (unsigned px = 0; px < w; ++px)
            if (img.getPixel(px, py).a >= 128) ++opaque;
    return static_cast<float>(opaque) / static_cast<float>(w * h);
}

AlphaCutout::Options toAlphaCutoutOptions(const AlphaCutoutSpec& spec) {
    AlphaCutout::Options opts;
    opts.tolerance = spec.tolerance;
    opts.featherRadius = spec.featherRadius;
    opts.defringe = spec.defringe;
    return opts;
}

}

namespace CandidateScorer {

CandidateScore scoreCandidate(const std::string& imagePath,
                              const AssetSpec& spec,
                              int index,
                              const AlphaCutoutSpec& cutoutSpec) {
    CandidateScore candidate;
    candidate.index = index;
    const std::filesystem::path inputPath(imagePath);
    const bool inputIsRaw = inputPath.parent_path().filename().string() == "raw";
    candidate.rawPath       = inputIsRaw ? inputPath.string()
                                         : AssetArtifactStore::rawPathForProcessed(inputPath).string();
    candidate.processedPath = inputIsRaw ? AssetArtifactStore::processedPathForRaw(inputPath).string()
                                         : inputPath.string();

    sf::Image img;
    if (!img.loadFromFile(candidate.rawPath) && !img.loadFromFile(candidate.processedPath))
        return candidate;

    const unsigned w = img.getSize().x;
    const unsigned h = img.getSize().y;
    if (w == 0 || h == 0) return candidate;

    if (spec.requiresTransparency)
        img = AlphaCutout::removeBackground(img, toAlphaCutoutOptions(cutoutSpec));

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

}
