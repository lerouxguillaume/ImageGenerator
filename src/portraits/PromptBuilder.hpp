#pragma once
#include <string>
#include <vector>
#include <cmath>
#include <cstdio>

// A single prompt term with an optional A1111-style attention weight.
// weight == 1.0 → plain text; weight != 1.0 → serialised as "(text:1.20)".
struct PromptToken {
    std::string text;
    float weight = 1.0f;
};

// Serialise one token: omits the weight wrapper when weight is effectively 1.0.
inline std::string formatToken(const PromptToken& token) {
    if (std::abs(token.weight - 1.0f) < 0.01f)
        return token.text;
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.2f", token.weight);
    return "(" + token.text + ":" + std::string(buf) + ")";
}

// Fluent builder for A1111-compatible weighted prompts.
// Usage:
//   PromptBuilder pb;
//   pb.add("masterpiece", 1.3f).add("portrait");
//   std::string prompt = pb.build(); // "masterpiece:1.30), portrait"
class PromptBuilder {
public:
    // Append a term. weight defaults to 1.0 (no wrapper).
    void add(const std::string& text, float weight = 1.0f) {
        tokens.push_back({text, weight});
    }

    // Serialise all tokens as a comma-separated string ready to pass to CLIP.
    std::string build() const {
        std::string result;
        for (size_t i = 0; i < tokens.size(); ++i) {
            result += formatToken(tokens[i]);
            if (i + 1 < tokens.size()) result += ", ";
        }
        return result;
    }

private:
    std::vector<PromptToken> tokens;
};

// Build a positive portrait prompt for a given race/gender.
// Token order is intentional: gender anchors come first so the model commits to
// them before reading composition and quality tags.
inline PromptBuilder buildCharacterPrompt(const Race& race, const Gender gender) {
    PromptBuilder pb;

    // Gender first — anchors the model before any composition tokens
    const float genderBoost = (race == Race::Elf) ? 1.5f : 1.4f;
    if (gender == Gender::Male) {
        pb.add("1boy",      genderBoost);
        pb.add("male",      1.2f);
        pb.add("masculine", 1.1f);
    } else {
        pb.add("1girl",  genderBoost);
        pb.add("female", 1.2f);
    }

    // Composition
    pb.add("fantasy art",        1.3f);
    pb.add("portrait",            1.3f);
    pb.add("close-up",            1.2f);
    pb.add("headshot",            1.2f);
    pb.add("upper body",          1.1f);
    pb.add("solo");
    pb.add("fantasy");
    pb.add("cinematic lighting");
    pb.add("highly detailed face", 1.2f);

    // Race
    switch (race) {
        case Race::Elf:   pb.add("elf",   1.2f); break;
        case Race::Dwarf: pb.add("dwarf", 1.2f); break;
        case Race::Orc:   pb.add("orc",   1.2f); break;
        default:          pb.add("human", 1.0f); break;
    }

    // Quality
    pb.add("best quality",   1.3f);
    pb.add("masterpiece",    1.2f);
    pb.add("ultra detailed", 1.2f);

    return pb;
}

// Build the negative prompt to suppress opposite-gender outputs, quality issues,
// and unwanted full-body compositions.
inline PromptBuilder buildNegativePrompt(const Race&, const Gender gender) {
    PromptBuilder neg;

    // Opposite gender first
    if (gender == Gender::Male) {
        neg.add("female", 1.3f);
        neg.add("1girl",  1.3f);
    } else {
        neg.add("male",  1.3f);
        neg.add("1boy",  1.3f);
    }

    // Quality
    neg.add("worst quality",     1.3f);
    neg.add("low quality",       1.2f);
    neg.add("blurry",            1.2f);
    neg.add("artifacts",         1.1f);
    neg.add("ugly",              1.1f);
    neg.add("poorly drawn face", 1.2f);

    // Anatomy — hands and arms are the most common failure mode
    neg.add("bad anatomy",            1.4f);
    neg.add("bad hands",              1.4f);
    neg.add("extra fingers",          1.4f);
    neg.add("missing fingers",        1.3f);
    neg.add("fused fingers",          1.3f);
    neg.add("too many fingers",       1.3f);
    neg.add("mutated hands",          1.3f);
    neg.add("poorly drawn hands",     1.3f);
    neg.add("extra arms",             1.3f);
    neg.add("missing arms",           1.3f);
    neg.add("extra limbs",            1.3f);
    neg.add("missing limbs",          1.2f);
    neg.add("malformed limbs",        1.3f);
    neg.add("disconnected limbs",     1.2f);
    neg.add("floating limbs",         1.2f);
    neg.add("deformed",               1.3f);
    neg.add("mutation",               1.2f);
    neg.add("gross proportions",      1.2f);
    neg.add("long neck",              1.1f);

    // Anti-fullbody (reduces chance of showing hands at all)
    neg.add("full body",       1.2f);
    neg.add("full-body shot",  1.1f);

    return neg;
}
