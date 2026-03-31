#include <cstdlib>
#include <ctime>
#include <string>

#include "../../entities/QuestTemplate.hpp"
#include "../../enum/enums.hpp"

class QuestTextGenerator {
public:
    QuestTextGenerator() {
        std::srand(static_cast<unsigned int>(std::time(nullptr))); // seed RNG
    }

    static std::string generateQuestName(QuestType type);
    static std::string generateQuestDescription(PartyRole partyRole);

private:
    static std::string partyRoleToString(PartyRole role);
    static void replaceAll(std::string& str, const std::string& from, const std::string& to);
};
