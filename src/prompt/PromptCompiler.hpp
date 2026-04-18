#pragma once
#include "Prompt.hpp"
#include "../enum/enums.hpp"
#include <string>

namespace PromptCompiler {
    // Format tokens into a comma-separated string. subject first, then positive.
    // Weighted tokens rendered as (value:weight). model parameter reserved for
    // future formatting differences; currently unused.
    std::string compile(const Prompt& p, ModelType model);

    // Format negative tokens into a comma-separated string.
    std::string compileNegative(const Prompt& p);
}
