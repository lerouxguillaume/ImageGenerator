#pragma once
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <cstdint>

class ClipTokenizer {
public:
    // vocabPath  : path to vocab.json
    // mergesPath : path to merges.txt
    ClipTokenizer(const std::string& vocabPath, const std::string& mergesPath);

    // Encode text to CLIP token IDs, padded/truncated to maxLen (default 77)
    std::vector<int64_t> encode(const std::string& text, int maxLen = 77) const;

private:
    std::unordered_map<std::string, int32_t>            vocab_;
    std::map<std::pair<std::string,std::string>, int32_t> mergeRank_;
    std::string byteToUnicode_[256]; // byte value → UTF-8 string of mapped codepoint

    static void        buildByteToUnicode(std::string out[256]);
    static std::string encodeUtf8(uint32_t codepoint);

    // Apply BPE to a sequence of unicode-character strings
    std::vector<std::string> bpe(std::vector<std::string> chars) const;

    // Encode one word (with optional leading-space prefix) into token IDs
    std::vector<int32_t> encodeWord(const std::string& word, bool leadingSpace) const;
};