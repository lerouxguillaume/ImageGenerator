#pragma once
#include "Project.hpp"
#include <filesystem>
#include <string>

namespace PatronGenerator {

// Generates a patron PNG for the given AssetSpec — a shape-correct reference
// image on a transparent background that is used as the img2img seed for
// CandidateRun exploration. Returns the output path on success, empty on failure.
std::string generate(const AssetSpec& spec, const std::filesystem::path& outputPath);

} // namespace PatronGenerator