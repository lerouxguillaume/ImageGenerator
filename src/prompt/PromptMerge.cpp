#include "PromptMerge.hpp"

namespace {

void mergeTokens(std::vector<Token>& base, const std::vector<Token>& patch) {
    for (const auto& pt : patch) {
        bool found = false;
        for (auto& bt : base) {
            if (bt.value == pt.value) { bt.weight = pt.weight; found = true; break; }
        }
        if (!found) base.push_back(pt);
    }
}

} // namespace

namespace PromptMerge {

Prompt merge(const Prompt& base, const Prompt& patch) {
    Prompt result = base;
    if (patch.subject) result.subject = patch.subject;
    mergeTokens(result.positive, patch.positive);
    mergeTokens (result.negative, patch.negative);
    return result;
}

} // namespace PromptMerge
