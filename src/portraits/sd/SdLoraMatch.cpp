#include "SdLoraMatch.hpp"
#include "../../managers/Logger.hpp"

namespace sd {

const TensorIndex* matchLoraKey(const OnnxSuffixIndex& suffixIndex,
                                const std::string&     loraBase) {
    // Try weight first (most common), then bias as fallback.
    const std::string weightKey = loraBase + "_weight";
    auto it = suffixIndex.find(weightKey);

    if (it == suffixIndex.end()) {
        const std::string biasKey = loraBase + "_bias";
        it = suffixIndex.find(biasKey);
        if (it == suffixIndex.end())
            return nullptr;
    }

    const auto& candidates = it->second;

    // Fast path: unique match — no ambiguity possible.
    if (candidates.size() == 1)
        return &candidates[0].it->second;

    // Resolve ambiguity: pick the candidate with the longest suffix match.
    const SuffixEntry* best = nullptr;
    for (const auto& entry : candidates) {
        if (!best || entry.suffixLen > best->suffixLen)
            best = &entry;
    }

    // Detect true ambiguity: multiple candidates share the same longest suffix length.
    int tieCount = 0;
    for (const auto& entry : candidates)
        if (entry.suffixLen == best->suffixLen) ++tieCount;

    if (tieCount > 1) {
        Logger::info("LoRA truly ambiguous (tie at suffix length "
                     + std::to_string(best->suffixLen) + "): " + loraBase
                     + " (" + std::to_string(tieCount) + " tied candidates)");
    } else if (candidates.size() > 1) {
        Logger::info("LoRA ambiguous match resolved: " + loraBase
                     + " (" + std::to_string(candidates.size()) + " candidates, picked longest)");
    }

    return best ? &best->it->second : nullptr;
}

} // namespace sd
