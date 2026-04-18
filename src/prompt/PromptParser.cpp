#include "PromptParser.hpp"
#include <sstream>
#include <cstdlib>

namespace {

std::string trim(const std::string& s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
}

std::vector<std::string> splitComma(const std::string& s) {
    std::vector<std::string> result;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        auto t = trim(tok);
        if (!t.empty())
            result.push_back(std::move(t));
    }
    return result;
}

// Parse A1111 weight syntax: (text:1.3) → {text, 1.3}, (text) → {text, 1.1}, text → {text, 1.0}
Token parseToken(const std::string& s) {
    if (s.size() >= 3 && s.front() == '(' && s.back() == ')') {
        const std::string inner = s.substr(1, s.size() - 2);
        const auto colon = inner.rfind(':');
        if (colon != std::string::npos) {
            try {
                const float w = std::stof(inner.substr(colon + 1));
                return {inner.substr(0, colon), w};
            } catch (...) {}
        }
        return {inner, 1.1f}; // bare (text) → A1111 convention weight
    }
    return {s, 1.0f};
}

} // namespace

namespace PromptParser {

Prompt parse(const std::string& positiveRaw, const std::string& negativeRaw) {
    Prompt p;

    const auto parts = splitComma(positiveRaw);
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i == 0)
            p.subject = parts[0]; // subject stays verbatim (may include A1111 syntax)
        else
            p.positive.push_back(parseToken(parts[i]));
    }

    for (const auto& s : splitComma(negativeRaw))
        p.negative.push_back(parseToken(s));

    return p;
}

} // namespace PromptParser
