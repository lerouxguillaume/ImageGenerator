#include "AssetPostProcessor.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace AssetPostProcessor {

namespace {
cv::Mat ensureRgba(const cv::Mat& input) {
    if (input.empty())
        throw std::runtime_error("AssetPostProcessor: empty input image");

    cv::Mat rgba;
    if (input.channels() == 4) {
        rgba = input.clone();
    } else if (input.channels() == 3) {
        cv::cvtColor(input, rgba, cv::COLOR_BGR2BGRA);
    } else {
        throw std::runtime_error("AssetPostProcessor: unsupported channel count");
    }
    return rgba;
}

cv::Rect detectContentBounds(const cv::Mat& rgba) {
    std::vector<cv::Mat> channels;
    cv::split(rgba, channels);
    cv::Mat mask;
    cv::threshold(channels[3], mask, 8, 255, cv::THRESH_BINARY);
    std::vector<cv::Point> points;
    cv::findNonZero(mask, points);
    if (points.empty())
        return {0, 0, rgba.cols, rgba.rows};
    return cv::boundingRect(points);
}

float computeFillRatio(const cv::Rect& bounds, int canvasW, int canvasH) {
    if (canvasW <= 0 || canvasH <= 0) return 0.0f;
    return static_cast<float>(bounds.width * bounds.height)
        / static_cast<float>(canvasW * canvasH);
}
}

AssetProcessResult process(const cv::Mat& input, const AssetExportSpec& spec) {
    const cv::Mat rgba = ensureRgba(input);
    AssetProcessResult result;
    result.hasAlpha = true;

    if (spec.fitMode == AssetFitMode::NoResize) {
        result.image = rgba.clone();
        result.contentBounds = detectContentBounds(result.image);
        result.fillRatio = computeFillRatio(result.contentBounds, result.image.cols, result.image.rows);
        return result;
    }

    if (spec.fitMode == AssetFitMode::TileExact) {
        cv::resize(rgba, result.image, {spec.exportWidth, spec.exportHeight}, 0.0, 0.0, cv::INTER_AREA);
        result.contentBounds = detectContentBounds(result.image);
        result.fillRatio = computeFillRatio(result.contentBounds, spec.exportWidth, spec.exportHeight);
        return result;
    }

    const cv::Rect sourceBounds = detectContentBounds(rgba);
    cv::Mat cropped = rgba(sourceBounds).clone();

    const int paddedMaxW = std::max(1, spec.maxObjectWidth);
    const int paddedMaxH = std::max(1, spec.maxObjectHeight);
    const float scale = std::min(
        static_cast<float>(paddedMaxW) / static_cast<float>(std::max(1, cropped.cols)),
        static_cast<float>(paddedMaxH) / static_cast<float>(std::max(1, cropped.rows)));
    const int resizedW = std::max(1, static_cast<int>(std::round(cropped.cols * scale)));
    const int resizedH = std::max(1, static_cast<int>(std::round(cropped.rows * scale)));

    cv::Mat resized;
    cv::resize(cropped, resized, {resizedW, resizedH}, 0.0, 0.0, scale < 1.0f ? cv::INTER_AREA : cv::INTER_LINEAR);

    result.image = cv::Mat(spec.exportHeight, spec.exportWidth, CV_8UC4, cv::Scalar(0, 0, 0, 0));
    const int minX = std::max(0, spec.paddingPx);
    const int minY = std::max(0, spec.paddingPx);
    const int maxX = std::max(minX, spec.exportWidth - spec.paddingPx - resizedW);
    const int maxY = std::max(minY, spec.exportHeight - spec.paddingPx - resizedH);
    const int x = std::clamp((spec.exportWidth - resizedW) / 2, minX, maxX);
    const int y = std::clamp((spec.exportHeight - resizedH) / 2, minY, maxY);
    resized.copyTo(result.image(cv::Rect(x, y, resizedW, resizedH)));

    result.contentBounds = detectContentBounds(result.image);
    result.fillRatio = computeFillRatio(result.contentBounds, spec.exportWidth, spec.exportHeight);
    return result;
}

}
