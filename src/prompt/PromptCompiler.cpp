#include "PromptCompiler.hpp"
#include <cstdio>

namespace {

void append(std::string& out, const std::string& part) {
    if (part.empty()) return;
    if (!out.empty()) out += ", ";
    out += part;
}

std::string formatToken(const Token& t) {
    if (std::abs(t.weight - 1.0f) < 0.001f)
        return t.value;
    char buf[512];
    std::snprintf(buf, sizeof(buf), "(%s:%.2f)", t.value.c_str(), t.weight);
    return buf;
}

} // namespace

namespace PromptCompiler {

std::string compile(const Prompt& p, ModelType /*model*/) {
    std::string out;

    if (p.subject)
        append(out, formatToken(*p.subject));

    for (const auto& t : p.positive)
        append(out, formatToken(t));

    return out;
}

std::string compileNegative(const Prompt& p) {
    std::string out;
    for (const auto& t : p.negative)
        append(out, formatToken(t));
    return out;
}

} // namespace PromptCompiler
