#pragma once

#include "../projects/Project.hpp"
#include <string>

struct CandidateScore {
    int         index = -1;
    std::string processedPath;
    std::string rawPath;
    float       score = 0.0f;
    bool        valid = false;
};

namespace CandidateScorer {

CandidateScore scoreCandidate(const std::string& imagePath,
                              const AssetSpec& spec,
                              int index,
                              const AlphaCutoutSpec& cutoutSpec = {});

}
