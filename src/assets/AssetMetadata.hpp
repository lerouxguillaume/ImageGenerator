#pragma once
#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include "../projects/Project.hpp"

struct AssetProcessResult {
    cv::Mat   image;          // final RGBA image
    cv::Rect  contentBounds;  // detected object bounds on final canvas
    float     fillRatio = 0.0f;
    bool      hasAlpha = false;
};

inline nlohmann::json toJson(const AssetProcessResult& result, const AssetExportSpec& spec) {
    return {
        {"exportWidth", spec.exportWidth},
        {"exportHeight", spec.exportHeight},
        {"contentBounds", {result.contentBounds.x, result.contentBounds.y,
                           result.contentBounds.width, result.contentBounds.height}},
        {"fillRatio", result.fillRatio},
        {"hasAlpha", result.hasAlpha},
        {"fitMode",
            spec.fitMode == AssetFitMode::TileExact ? "TileExact" :
            spec.fitMode == AssetFitMode::NoResize ? "NoResize" : "ObjectFit"}
    };
}
