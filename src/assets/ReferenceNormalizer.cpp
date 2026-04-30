#include "ReferenceNormalizer.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace ReferenceNormalizer {

cv::Mat normalizeToCanvas(const std::string& imagePath, int targetWidth, int targetHeight) {
    cv::Mat input = cv::imread(imagePath, cv::IMREAD_UNCHANGED);
    if (input.empty())
        throw std::runtime_error("ReferenceNormalizer: could not read '" + imagePath + "'");

    cv::Mat rgba;
    if (input.channels() == 4) {
        cv::cvtColor(input, rgba, cv::COLOR_BGRA2RGBA);
    } else if (input.channels() == 3) {
        cv::cvtColor(input, rgba, cv::COLOR_BGR2RGBA);
    } else if (input.channels() == 1) {
        cv::cvtColor(input, rgba, cv::COLOR_GRAY2RGBA);
    } else {
        throw std::runtime_error("ReferenceNormalizer: unsupported channel count");
    }

    cv::Mat resized;
    cv::resize(rgba, resized, {targetWidth, targetHeight}, 0.0, 0.0, cv::INTER_LINEAR);
    return resized;
}

}
