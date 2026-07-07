#include "GoldenRun.hpp"
#include "../portraits/PortraitGeneratorAi.hpp"
#include "../managers/Logger.hpp"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <optional>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>

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

// Run one synchronous generation. When `cancelAt` is set, a watcher thread
// requests cancellation once the pipeline's cumulative step counter reaches it —
// exercising the exact SetTerminate() cancel path the GUI uses. Returns false on
// a genuine error (cancellation is NOT an error: the pipeline swallows it).
bool runOne(const std::string& prompt, const std::string& neg, const std::string& out,
            const GenerationParams& params, const std::string& model,
            std::optional<int> cancelAt) {
    std::atomic<int> progressStep{0};
    std::stop_source src;
    std::jthread watcher;
    if (cancelAt) {
        watcher = std::jthread([&progressStep, &src, n = *cancelAt](std::stop_token wt) {
            while (!wt.stop_requested()) {
                if (progressStep.load() >= n) { src.request_stop(); return; }
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
        });
    }

    bool ok = true;
    try {
        PortraitGeneratorAi::generateFromPrompt(prompt, neg, out, params, model,
                                                &progressStep, nullptr, src.get_token(), nullptr);
    } catch (const std::exception& e) {
        Logger::error(std::string("headless-generate failed: ") + e.what());
        std::cerr << "headless-generate failed: " << e.what() << "\n";
        ok = false;
    }
    if (watcher.joinable()) { watcher.request_stop(); watcher.join(); }
    return ok;
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

    // Hires-fix flags (SD1.5 only; no-ops on non-hires-capable models).
    if (hasFlag(argc, argv, "--hires"))                    p.hires.enabled  = true;
    if (auto v = argValue(argc, argv, "--hires-scale"))    p.hires.scale    = std::strtof(v, nullptr);
    if (auto v = argValue(argc, argv, "--hires-strength")) p.hires.strength = std::strtof(v, nullptr);
    if (auto v = argValue(argc, argv, "--hires-steps"))    p.hires.steps    = std::atoi(v);
    if (auto v = argValue(argc, argv, "--hires-mode"))     // "pixel" (default) | "latent"
        p.hires.mode = (std::string_view(v) == "latent") ? UpscaleMode::Latent : UpscaleMode::Pixel;

    // Cancel-restore harness hook: cancel after N cumulative denoise steps.
    std::optional<int> cancelAfter;
    if (auto v = argValue(argc, argv, "--cancel-after-steps")) cancelAfter = std::atoi(v);

    Logger::info("=== headless-generate ===");
    Logger::info(std::string("model=") + model + "  out=" + out
                 + "  seed=" + std::to_string(p.seed)
                 + "  steps=" + std::to_string(p.numSteps)
                 + "  images=" + std::to_string(p.numImages)
                 + "  guidance=" + std::to_string(p.guidanceScale)
                 + (p.hires.enabled
                        ? ("  hires=on scale=" + std::to_string(p.hires.scale)
                           + " strength=" + std::to_string(p.hires.strength))
                        : std::string())
                 + (cancelAfter ? ("  cancel-after-steps=" + std::to_string(*cancelAfter)) : std::string())
                 + (p.initImagePath.empty()
                        ? std::string()
                        : ("  init=" + p.initImagePath + "  strength=" + std::to_string(p.strength))));

    // Cancel-restore mode: run a first generation that self-cancels mid-pass
    // (pick N inside the hires pass), then — in the SAME process, sharing the
    // static ModelManager cache — a clean native generation to --out. If
    // ScopedLatentResolution failed to restore ctx dims on the cancel path, the
    // second (native) output would be wrong; comparing it to an existing native
    // golden proves the restore.
    if (cancelAfter) {
        const std::string cancelledOut = std::string(out) + ".cancelled.png";
        Logger::info("cancel-restore: pass 1 self-cancels after "
                     + std::to_string(*cancelAfter) + " steps → " + cancelledOut);
        runOne(prompt, neg, cancelledOut, p, model, cancelAfter);   // may abort mid-hires

        GenerationParams nativeParams = p;
        nativeParams.hires.enabled = false;   // pass 2: native, no hires
        Logger::info("cancel-restore: pass 2 native generation → " + std::string(out));
        if (!runOne(prompt, neg, out, nativeParams, model, std::nullopt))
            return 1;
        std::cout << "headless-generate OK (cancel-restore): " << out << "\n";
        return 0;
    }

    if (!runOne(prompt, neg, out, p, model, std::nullopt))
        return 1;
    std::cout << "headless-generate OK: " << out << "\n";
    return 0;
}

} // namespace headless
