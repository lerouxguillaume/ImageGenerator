#pragma once
#include "Prompt.hpp"
#include <string>

namespace PromptParser {
    // Parse raw positive and negative text into the DSL.
    // Heuristic: first comma-token becomes subject.
    // Invariant: compile(parse(x), SDXL) ≈ x
    Prompt parse(const std::string& positiveRaw, const std::string& negativeRaw = {});
}
