#pragma once
#include "Prompt.hpp"
#include "../enum/enums.hpp"
#include <string>

namespace PromptCompiler {
    // Compile positive tokens into a prompt string, model-specific.
    // SDXL: natural phrasing, no boosters.
    // SD15: subject boost (x:1.20), quality boosters appended.
    std::string compile(const Prompt& p, ModelType model);

    // Compile negative tokens. Same format for both models.
    std::string compileNegative(const Prompt& p, ModelType model);
}
