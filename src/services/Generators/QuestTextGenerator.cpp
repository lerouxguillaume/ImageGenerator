#include "QuestTextGenerator.hpp"

#include <vector>

std::string QuestTextGenerator::generateQuestName(QuestType type) {
    const std::vector<std::string> verbs = {"Rescue", "Defeat", "Protect", "Retrieve", "Investigate", "Explore"};
    const std::vector<std::string> nouns = {
        "Goblin King", "Ancient Artifact", "Lost Prince", "Haunted Forest", "Dark Labyrinth"
    };
    const std::vector<std::string> adjectives = {"Cursed", "Forgotten", "Mysterious", "Dark", "Hidden"};

    std::string templateStr;
    if (type == QuestType::Dungeon) {
        templateStr = "{verb} the {adjective} {noun}";
    } else {
        // normal quest
        templateStr = "{verb} the {noun}";
    }

    replaceAll(templateStr, "{verb}", verbs[rand() % verbs.size()]);
    replaceAll(templateStr, "{noun}", nouns[rand() % nouns.size()]);
    replaceAll(templateStr, "{adjective}", adjectives[rand() % adjectives.size()]);

    return templateStr;
}

std::string QuestTextGenerator::generateQuestDescription(const PartyRole partyRole) {
    // Some example templates
    const std::vector<std::string> descTemplates = {
        "The town of {town} is in peril, and only a brave {role} can {verb} the {noun}.",
        "Legends speak of the {adjective} {noun} that threatens our lands. Your mission: {verb} it.",
        "{role}s are needed to {verb} the {adjective} {noun} before {danger} occurs."
    };

    const std::vector<std::string> verbs = {"rescue", "defeat", "protect", "retrieve", "investigate"};
    const std::vector<std::string> nouns = {"Goblin King", "Ancient Artifact", "Lost Prince", "Haunted Forest"};
    const std::vector<std::string> adjectives = {"cursed", "forgotten", "mysterious", "dark", "hidden"};
    const std::vector<std::string> towns = {"Ironhold", "Stonehaven", "Ravencrest"};
    const std::vector<std::string> dangers = {"doom", "destruction", "chaos", "darkness"};

    std::string templateStr = descTemplates[rand() % descTemplates.size()];

    // pick the first role in roleRequirements for description flavor
    std::string roleStr = "adventurer";
    roleStr = partyRoleToString(partyRole);

    replaceAll(templateStr, "{role}", roleStr);
    replaceAll(templateStr, "{verb}", verbs[rand() % verbs.size()]);
    replaceAll(templateStr, "{noun}", nouns[rand() % nouns.size()]);
    replaceAll(templateStr, "{adjective}", adjectives[rand() % adjectives.size()]);
    replaceAll(templateStr, "{town}", towns[rand() % towns.size()]);
    replaceAll(templateStr, "{danger}", dangers[rand() % dangers.size()]);

    return templateStr;
}


std::string QuestTextGenerator::partyRoleToString(PartyRole role) {
    switch (role) {
        case PartyRole::Leader: return "Leader";
        case PartyRole::Tank: return "Tank";
        case PartyRole::Healer: return "Healer";
        case PartyRole::Scout: return "Scout";
        case PartyRole::DPS: return "DPS";
        default: return "Adventurer";
    }
}

// Simple string replace helper
void QuestTextGenerator::replaceAll(std::string& str, const std::string& from, const std::string& to) {
    size_t pos = 0;
    while ((pos = str.find(from, pos)) != std::string::npos) {
        str.replace(pos, from.length(), to);
        pos += to.length();
    }
}
