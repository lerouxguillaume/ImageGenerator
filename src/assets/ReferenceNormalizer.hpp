#pragma once
#include <opencv2/core.hpp>
#include <string>

namespace ReferenceNormalizer {

cv::Mat normalizeToCanvas(const std::string& imagePath, int targetWidth, int targetHeight);

}
