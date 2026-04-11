// buildSuffixIndex: secondary O(1) lookup table over an OnnxTensorIndex.
#include "SdOnnxPatcher.hpp"

namespace sd {

OnnxSuffixIndex buildSuffixIndex(const OnnxTensorIndex& index) {
    OnnxSuffixIndex result;
    result.reserve(index.size() * 6); // realistic average

    for (auto it = index.cbegin(); it != index.cend(); ++it) {
        const std::string& name = it->first;

        size_t pos = 0;
        while (pos < name.size()) {
            std::string suffix = name.substr(pos);

            // Count segments (avoid useless suffixes like "weight")
            int underscoreCount = 0;
            for (char c : suffix) if (c == '_') ++underscoreCount;

            if (underscoreCount >= 1) { // keep only meaningful suffixes
                result[suffix].push_back({
                    it,
                    suffix.size()
                });
            }

            const size_t next = name.find('_', pos);
            if (next == std::string::npos)
                break;

            pos = next + 1;
        }
    }

    return result;
}

} // namespace sd
