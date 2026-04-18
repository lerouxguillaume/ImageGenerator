#pragma once
#include <optional>
#include <string>
#include <vector>

struct Token {
    std::string value;
    float       weight = 1.0f;
};

struct Prompt {
    std::optional<std::string> subject;
    std::vector<std::string>   styles;
    std::vector<Token>         positive;
    std::vector<Token>         negative;
};
