#include "PatronGenerator.hpp"
#include <algorithm>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

namespace PatronGenerator {
namespace {

const cv::Scalar kPatronFill(128, 128, 128, 255);

OccupiedBounds resolveBounds(const AssetSpec& spec, int canvasWidth, int canvasHeight) {
    if (spec.expectedBounds.w > 0 && spec.expectedBounds.h > 0)
        return spec.expectedBounds;

    const int marginX = canvasWidth / 4;
    const int marginY = canvasHeight / 4;
    return {marginX, marginY, canvasWidth - marginX * 2, canvasHeight - marginY * 2};
}

int wallSkewFor(const OccupiedBounds& bounds) {
    return std::max(0, std::min(bounds.w / 4, bounds.h / 2));
}

std::vector<cv::Point> polygonFor(const OccupiedBounds& bounds, Orientation orientation) {
    const int bx = bounds.x;
    const int by = bounds.y;
    const int bw = bounds.w;
    const int bh = bounds.h;

    switch (orientation) {
        case Orientation::LeftWall: {
            // Isometric left wall: top and bottom edges rise toward the right.
            const int skew = wallSkewFor(bounds);
            return {
                {bx,      by + skew},
                {bx + bw, by},
                {bx + bw, by + bh - skew},
                {bx,      by + bh},
            };
        }
        case Orientation::RightWall: {
            // Mirrored wall: top and bottom edges fall toward the right.
            const int skew = wallSkewFor(bounds);
            return {
                {bx,      by},
                {bx + bw, by + skew},
                {bx + bw, by + bh},
                {bx,      by + bh - skew},
            };
        }
        case Orientation::FloorTile:
            return {
                {bx + bw / 2, by},
                {bx + bw,     by + bh / 2},
                {bx + bw / 2, by + bh},
                {bx,          by + bh / 2},
            };
        case Orientation::Unset:
        case Orientation::Prop:
        case Orientation::Character:
            return {
                {bx,      by},
                {bx + bw, by},
                {bx + bw, by + bh},
                {bx,      by + bh},
            };
    }

    return {};
}

void drawPatronShape(cv::Mat& patron, const OccupiedBounds& bounds, Orientation orientation) {
    if (orientation == Orientation::Character) {
        const cv::Point center(bounds.x + bounds.w / 2, bounds.y + bounds.h / 2);
        const cv::Size axes(std::max(1, bounds.w / 2), std::max(1, bounds.h / 2));
        cv::ellipse(patron, center, axes, 0.0, 0.0, 360.0, kPatronFill, cv::FILLED);
        return;
    }

    cv::fillPoly(patron,
                 std::vector<std::vector<cv::Point>>{polygonFor(bounds, orientation)},
                 kPatronFill);
}

} // namespace

std::string generate(const AssetSpec& spec, const std::filesystem::path& outputPath) {
    const int W = spec.canvasWidth  > 0 ? spec.canvasWidth  : 512;
    const int H = spec.canvasHeight > 0 ? spec.canvasHeight : 768;

    // Fully transparent BGRA canvas
    cv::Mat patron(H, W, CV_8UC4, cv::Scalar(0, 0, 0, 0));

    // Neutral mid-gray fill — not white (biases toward overexposed) nor dark
    drawPatronShape(patron, resolveBounds(spec, W, H), spec.orientation);

    if (!cv::imwrite(outputPath.string(), patron))
        return {};

    return outputPath.string();
}

} // namespace PatronGenerator
