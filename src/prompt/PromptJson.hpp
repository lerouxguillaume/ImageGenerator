#pragma once
#include "Prompt.hpp"
#include <nlohmann/json.hpp>

// ADL hooks for nlohmann_json — must be in the same namespace as Token/Prompt (global).

inline void to_json(nlohmann::json& j, const Token& t) {
    j = {{"value", t.value}, {"weight", t.weight}};
}

inline void from_json(const nlohmann::json& j, Token& t) {
    t.value  = j.value("value",  std::string{});
    t.weight = j.value("weight", 1.0f);
}

inline void to_json(nlohmann::json& j, const Prompt& p) {
    j = nlohmann::json::object();
    if (p.subject) j["subject"] = *p.subject;
    j["styles"]   = p.styles;
    j["positive"] = p.positive;
    j["negative"] = p.negative;
}

inline void from_json(const nlohmann::json& j, Prompt& p) {
    if (j.contains("subject") && j["subject"].is_string())
        p.subject = j["subject"].get<std::string>();
    if (j.contains("styles"))
        p.styles = j["styles"].get<std::vector<std::string>>();
    if (j.contains("positive"))
        p.positive = j["positive"].get<std::vector<Token>>();
    if (j.contains("negative"))
        p.negative = j["negative"].get<std::vector<Token>>();
}
