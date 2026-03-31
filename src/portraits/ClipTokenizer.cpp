#include "ClipTokenizer.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <climits>
#include <cctype>
#include <nlohmann/json.hpp>

// ── UTF-8 helper ─────────────────────────────────────────────────────────────

std::string ClipTokenizer::encodeUtf8(uint32_t cp) {
    std::string s;
    if (cp < 0x80) {
        s += static_cast<char>(cp);
    } else if (cp < 0x800) {
        s += static_cast<char>(0xC0 | (cp >> 6));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        s += static_cast<char>(0xE0 | (cp >> 12));
        s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    }
    return s;
}

// ── Byte-to-unicode map (GPT-2 / CLIP standard) ───────────────────────────────
// Printable bytes map to themselves; the rest map to codepoints 256, 257, ...
// This matches Python's bytes_to_unicode() in the original CLIP tokenizer.

void ClipTokenizer::buildByteToUnicode(std::string out[256]) {
    bool mapped[256] = {};

    auto direct = [&](int lo, int hi) {
        for (int b = lo; b <= hi; ++b) {
            out[b] = encodeUtf8(static_cast<uint32_t>(b));
            mapped[b] = true;
        }
    };
    direct(33, 126);   // '!' .. '~'
    direct(161, 172);  // '¡' .. '¬'
    direct(174, 255);  // '®' .. 'ÿ'

    uint32_t next = 256;
    for (int b = 0; b < 256; ++b)
        if (!mapped[b])
            out[b] = encodeUtf8(next++);
}

// ── Constructor ───────────────────────────────────────────────────────────────

ClipTokenizer::ClipTokenizer(const std::string& vocabPath, const std::string& mergesPath) {
    buildByteToUnicode(byteToUnicode_);

    // Load vocab.json
    {
        std::ifstream f(vocabPath);
        if (!f.is_open())
            throw std::runtime_error("ClipTokenizer: cannot open vocab file: " + vocabPath);
        auto j = nlohmann::json::parse(f);
        for (auto& [key, val] : j.items())
            vocab_[key] = val.get<int32_t>();
    }

    // Load merges.txt  (first line is a comment, skip it)
    {
        std::ifstream f(mergesPath);
        if (!f.is_open())
            throw std::runtime_error("ClipTokenizer: cannot open merges file: " + mergesPath);
        std::string line;
        std::getline(f, line); // skip "#version: ..." header
        int rank = 0;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            auto sep = line.find(' ');
            if (sep == std::string::npos) continue;
            mergeRank_[{line.substr(0, sep), line.substr(sep + 1)}] = rank++;
        }
    }
}

// ── BPE ──────────────────────────────────────────────────────────────────────

std::vector<std::string> ClipTokenizer::bpe(std::vector<std::string> chars) const {
    while (chars.size() > 1) {
        // Find the highest-priority (lowest rank) adjacent pair
        int    bestRank = INT_MAX;
        size_t bestI    = SIZE_MAX;
        for (size_t i = 0; i + 1 < chars.size(); ++i) {
            auto it = mergeRank_.find({chars[i], chars[i + 1]});
            if (it != mergeRank_.end() && it->second < bestRank) {
                bestRank = it->second;
                bestI    = i;
            }
        }
        if (bestI == SIZE_MAX) break;

        // Merge ALL occurrences of that pair (left to right)
        const std::string a = chars[bestI];
        const std::string b = chars[bestI + 1];
        std::string merged  = a + b;

        std::vector<std::string> next;
        next.reserve(chars.size());
        for (size_t i = 0; i < chars.size(); ) {
            if (i + 1 < chars.size() && chars[i] == a && chars[i + 1] == b) {
                next.push_back(merged);
                i += 2;
            } else {
                next.push_back(chars[i]);
                ++i;
            }
        }
        chars = std::move(next);
    }
    return chars;
}

// ── Word encoding ─────────────────────────────────────────────────────────────

std::vector<int32_t> ClipTokenizer::encodeWord(const std::string& word, bool leadingSpace) const {
    // Build the unicode-encoded string for this word
    std::string encoded;
    if (leadingSpace)
        encoded += byteToUnicode_[' '];       // byte 32 → Ġ
    for (const unsigned char c : word)
        encoded += byteToUnicode_[c];

    // Split the UTF-8 string into individual unicode-character substrings
    std::vector<std::string> chars;
    for (size_t i = 0; i < encoded.size(); ) {
        const unsigned char c = encoded[i];
        const int len = (c < 0x80) ? 1 : (c < 0xE0) ? 2 : 3;
        chars.push_back(encoded.substr(i, len));
        i += len;
    }

    // Apply BPE merges, then look up IDs
    std::vector<int32_t> ids;
    for (const auto& tok : bpe(std::move(chars))) {
        auto it = vocab_.find(tok);
        if (it != vocab_.end())
            ids.push_back(it->second);
    }
    return ids;
}

// ── Main encode ───────────────────────────────────────────────────────────────

std::vector<int64_t> ClipTokenizer::encode(const std::string& text, int maxLen) const {
    const int32_t BOS = vocab_.count("<|startoftext|>") ? vocab_.at("<|startoftext|>") : 49406;
    const int32_t EOS = vocab_.count("<|endoftext|>")   ? vocab_.at("<|endoftext|>")   : 49407;

    // Lowercase
    std::string lower = text;
    for (char& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    // Split into pieces: runs of alphanumerics, or individual punctuation/symbols
    // Each piece tracks whether it was preceded by whitespace.
    struct Piece { std::string text; bool precededBySpace; };
    std::vector<Piece> pieces;
    bool lastWasSpace = false;
    std::string cur;
    bool curIsAlnum = false;

    auto flush = [&]() {
        if (!cur.empty()) {
            pieces.push_back({cur, lastWasSpace && pieces.empty() ? false : lastWasSpace});
            cur.clear();
        }
    };

    for (size_t i = 0; i < lower.size(); ++i) {
        const unsigned char c = lower[i];
        if (std::isspace(c)) {
            flush();
            lastWasSpace = true;
            curIsAlnum   = false;
        } else if (std::isalnum(c)) {
            if (!curIsAlnum) { flush(); curIsAlnum = true; }
            cur += c;
        } else {
            // Punctuation: always its own piece
            flush();
            curIsAlnum = false;
            pieces.push_back({std::string(1, c), lastWasSpace});
            lastWasSpace = false;
        }
        if (!std::isspace(c)) lastWasSpace = false;
    }
    flush();

    // Build token ID sequence
    std::vector<int64_t> ids;
    ids.push_back(BOS);
    for (size_t pi = 0; pi < pieces.size() && static_cast<int>(ids.size()) < maxLen - 1; ++pi) {
        const bool space = (pi > 0) && pieces[pi].precededBySpace;
        for (const int32_t id : encodeWord(pieces[pi].text, space)) {
            if (static_cast<int>(ids.size()) >= maxLen - 1) break;
            ids.push_back(id);
        }
    }
    ids.push_back(EOS);
    while (static_cast<int>(ids.size()) < maxLen)
        ids.push_back(EOS); // pad

    return ids;
}