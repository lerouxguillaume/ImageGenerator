#pragma once
#include <optional>

// Headless generation entry point — the app's golden-run regression harness.
//
// This is NOT a feature: it changes no generation behavior. It only lets the
// binary drive sd::runPipeline() once with fully-scripted, deterministic
// parameters and exit, so a shell harness can capture golden outputs from one
// build and hash-compare them against another. Used to prove the Phase 1
// pipeline refactor is bit-identical.
namespace headless {

// If argv contains "--headless-generate", run the scripted generation and
// return the process exit code (0 on success, 1 on failure). Otherwise return
// std::nullopt so main() continues into the normal GUI app.
std::optional<int> maybeRunHeadless(int argc, char** argv);

} // namespace headless
