#include "PatronGenerator.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace PatronGenerator {

std::string generate(const AssetSpec& spec, const std::filesystem::path& outputPath) {
    const int W = spec.canvasWidth  > 0 ? spec.canvasWidth  : 512;
    const int H = spec.canvasHeight > 0 ? spec.canvasHeight : 768;

    // Fully transparent BGRA canvas
    cv::Mat patron(H, W, CV_8UC4, cv::Scalar(0, 0, 0, 0));

    const int bx = spec.expectedBounds.x;
    const int by = spec.expectedBounds.y;
    const int bw = spec.expectedBounds.w;
    const int bh = spec.expectedBounds.h;

    std::vector<cv::Point> pts;

    if (spec.orientation == Orientation::LeftWall && bw > 0 && bh > 0) {
        // Isometric left wall: parallelogram with 1-up-per-4-right slope on top/bottom edges.
        // Top-right corner is highest; bottom-left corner is lowest.
        const int skew = bw / 4;
        pts = {
            {bx,      by + skew},
            {bx + bw, by},
            {bx + bw, by + bh - skew},
            {bx,      by + bh},
        };
    } else if (bw > 0 && bh > 0) {
        pts = {
            {bx,      by},
            {bx + bw, by},
            {bx + bw, by + bh},
            {bx,      by + bh},
        };
    } else {
        const int cx = W / 4, cy = H / 4;
        pts = {
            {cx,     cy},
            {W - cx, cy},
            {W - cx, H - cy},
            {cx,     H - cy},
        };
    }

    // Neutral mid-gray fill — not white (biases toward overexposed) nor dark
    cv::fillPoly(patron,
                 std::vector<std::vector<cv::Point>>{pts},
                 cv::Scalar(128, 128, 128, 255));

    if (!cv::imwrite(outputPath.string(), patron))
        return {};

    return outputPath.string();
}

} // namespace PatronGenerator