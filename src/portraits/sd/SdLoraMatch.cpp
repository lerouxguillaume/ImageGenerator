#include "SdLoraMatch.hpp"
#include "../../managers/Logger.hpp"

// Define SD_LORA_MATCH_DEBUG=1 at compile time to enable per-key candidate
// logging.  Off by default — produces one log line per matched layer which
// is verbose for a 180-layer model.
#ifndef SD_LORA_MATCH_DEBUG
#define SD_LORA_MATCH_DEBUG 0
#endif

namespace sd {

const ExternalTensorMeta* matchExternalLoraKey(const OnnxExternalSuffixIndex& suffixIndex,
                                               const std::string&             loraBase) {
    // Try weight first (most common), then bias as fallback.
    const std::string weightKey = loraBase + "_weight";
    auto it = suffixIndex.find(weightKey);

#if SD_LORA_MATCH_DEBUG
    Logger::info("[LoraMatch] lookup: '" + weightKey + "'  found=" + (it != suffixIndex.end() ? "yes" : "no"));
#endif

    if (it == suffixIndex.end()) {
        const std::string biasKey = loraBase + "_bias";
        it = suffixIndex.find(biasKey);

#if SD_LORA_MATCH_DEBUG
        Logger::info("[LoraMatch] fallback lookup: '" + biasKey + "'  found=" + (it != suffixIndex.end() ? "yes" : "no"));
#endif

        if (it == suffixIndex.end()) return nullptr;
    }

    const auto& candidates = it->second;

    // Fast path: unique match — no ambiguity possible.
    if (candidates.size() == 1) {
#if SD_LORA_MATCH_DEBUG
        Logger::info("[LoraMatch] unique match: " + candidates[0].it->first);
#endif
        return &candidates[0].it->second;
    }

    // Resolve ambiguity: pick the candidate with the longest suffix match.
    const ExternalSuffixEntry* best = nullptr;
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

#if SD_LORA_MATCH_DEBUG
    if (best) {
        Logger::info("[LoraMatch] chosen: " + best->it->first
                     + "  (suffixLen=" + std::to_string(best->suffixLen)
                     + ", " + std::to_string(candidates.size()) + " candidate(s))");
        for (const auto& e : candidates)
            Logger::info("[LoraMatch]   candidate: " + e.it->first
                         + "  suffixLen=" + std::to_string(e.suffixLen));
    }
#endif

    return best ? &best->it->second : nullptr;
}

} // namespace sd
