#include "GoldenRun.hpp"
#include "../portraits/PortraitGeneratorAi.hpp"
#include "../managers/Logger.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <string_view>

namespace headless {
namespace {

// Return the value following `key` in argv, or `def` if the flag is absent.
const char* argValue(int argc, char** argv, std::string_view key, const char* def = nullptr) {
    for (int i = 1; i + 1 < argc; ++i)
        if (key == argv[i]) return argv[i + 1];
    return def;
}

bool hasFlag(int argc, char** argv, std::string_view key) {
    for (int i = 1; i < argc; ++i)
        if (key == argv[i]) return true;
    return false;
}

} // namespace

std::optional<int> maybeRunHeadless(int argc, char** argv) {
    if (!hasFlag(argc, argv, "--headless-generate"))
        return std::nullopt;

    const char* model  = argValue(argc, argv, "--model", "models");
    const char* prompt = argValue(argc, argv, "--prompt", "");
    const char* neg    = argValue(argc, argv, "--neg", "");
    const char* out    = argValue(argc, argv, "--out", "assets/generated/golden.png");

    // seed must be explicit for a golden run — the default -1 would be random.
    GenerationParams p;
    if (auto v = argValue(argc, argv, "--seed"))        p.seed          = std::strtoll(v, nullptr, 10);
    if (auto v = argValue(argc, argv, "--steps"))       p.numSteps      = std::atoi(v);
    if (auto v = argValue(argc, argv, "--images"))      p.numImages     = std::atoi(v);
    if (auto v = argValue(argc, argv, "--guidance"))    p.guidanceScale = std::strtof(v, nullptr);
    if (auto v = argValue(argc, argv, "--cfg-rescale")) p.cfgRescale    = std::strtof(v, nullptr);
    if (auto v = argValue(argc, argv, "--width"))       p.width         = std::atoi(v);
    if (auto v = argValue(argc, argv, "--height"))      p.height        = std::atoi(v);
    if (auto v = argValue(argc, argv, "--init"))        p.initImagePath = v;
    if (auto v = argValue(argc, argv, "--strength"))    p.strength      = std::strtof(v, nullptr);

    Logger::info("=== headless-generate ===");
    Logger::info(std::string("model=") + model + "  out=" + out
                 + "  seed=" + std::to_string(p.seed)
                 + "  steps=" + std::to_string(p.numSteps)
                 + "  images=" + std::to_string(p.numImages)
                 + "  guidance=" + std::to_string(p.guidanceScale)
                 + (p.initImagePath.empty()
                        ? std::string()
                        : ("  init=" + p.initImagePath + "  strength=" + std::to_string(p.strength))));

    try {
        // No progress/stage/stop plumbing: a golden run is synchronous and
        // uncancelled. This exercises the exact same runPipeline() path the GUI
        // uses; only the callers of the atomics differ (here they are null).
        PortraitGeneratorAi::generateFromPrompt(prompt, neg, out, p, model);
    } catch (const std::exception& e) {
        Logger::error(std::string("headless-generate failed: ") + e.what());
        std::cerr << "headless-generate failed: " << e.what() << "\n";
        return 1;
    }
    std::cout << "headless-generate OK: " << out << "\n";
    return 0;
}

} // namespace headless
