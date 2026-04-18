#pragma once
#include "Prompt.hpp"

namespace PromptMerge {
    // Merge patch into base. Rules:
    //   subject  — patch overrides if set
    //   styles   — union, deduplicated (base order preserved, patch appended)
    //   tokens   — deduplicated by value; patch overrides weight on collision
    Prompt merge(const Prompt& base, const Prompt& patch);
}
