#pragma once
#include <string>
#include <vector>
#include <SFML/Graphics.hpp>
#include "../projects/Project.hpp"

namespace AssetValidator {

// 0 = pass, 1 = warning, 2 = fail
struct Check {
    std::string name;
    int         status = 0;
    std::string detail;
};

struct Result {
    std::vector<Check> checks;
    bool exportReady() const {
        for (const auto& c : checks)
            if (c.status == 2) return false;
        return true;
    }
};

Result validate(const sf::Image& img, const AssetSpec& spec);

} // namespace AssetValidator
