#include "PromptCompiler.hpp"
#include "../managers/Logger.hpp"
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

// True when a string is already in A1111 weight format: (text:N.N)
bool hasA1111Weight(const std::string& s) {
    return s.size() >= 4 && s.front() == '(' && s.back() == ')'
           && s.rfind(':') != std::string::npos;
}

} // namespace

namespace PromptCompiler {

std::string compile(const Prompt& p, ModelType model) {
    std::string out;

    if (p.subject) {
        if (model == ModelType::SD15 && !hasA1111Weight(*p.subject))
            append(out, "(" + *p.subject + ":1.20)");
        else
            append(out, *p.subject);
    }

    for (const auto& s : p.styles)
        append(out, s);

    for (const auto& t : p.positive)
        append(out, formatToken(t));

    if (model == ModelType::SD15) {
        append(out, "masterpiece");
        append(out, "best quality");
    }

    const std::string modelStr = (model == ModelType::SDXL) ? "SDXL" : "SD15";
    Logger::info("[PromptCompiler] model=" + modelStr + " output=\"" + out + "\"");

    return out;
}

std::string compileNegative(const Prompt& p, ModelType /*model*/) {
    std::string out;
    for (const auto& t : p.negative)
        append(out, formatToken(t));
    return out;
}

} // namespace PromptCompiler
