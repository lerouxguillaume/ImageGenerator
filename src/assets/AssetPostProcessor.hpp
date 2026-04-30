#pragma once
#include <opencv2/core.hpp>
#include "AssetMetadata.hpp"

namespace AssetPostProcessor {

AssetProcessResult process(const cv::Mat& input, const AssetExportSpec& spec);

}
