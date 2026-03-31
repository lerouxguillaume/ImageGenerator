#include "CharacterAttributesService.hpp"

#include "utils.hpp"
#include <memory>

void CharacterAttributesService::createCharacterAttributes(Character& character) {
    auto adventurerAttributes = std::make_unique<AdventurerAttributes>();
    adventurerAttributes->setStrength(rollStat());
    adventurerAttributes->setSpeed(rollStat());
    adventurerAttributes->setAgility(rollStat());
    adventurerAttributes->setReflex(rollStat());
    adventurerAttributes->setStamina(rollStat());
    adventurerAttributes->setVitality(rollStat());
    adventurerAttributes->setBalance(rollStat());
    adventurerAttributes->setIntelligence(rollStat());
    adventurerAttributes->setConcentration(rollStat());
    adventurerAttributes->setWillpower(rollStat());
    adventurerAttributes->setDiscipline(rollStat());
    adventurerAttributes->setCreativity(rollStat());
    adventurerAttributes->setAdaptation(rollStat());
    adventurerAttributes->setSensitivity(rollStat());
    adventurerAttributes->setInsight(rollStat());
    adventurerAttributes->setPerception(rollStat());
    adventurerAttributes->setInstinct(rollStat());
    adventurerAttributes->setAwareness(rollStat());
    adventurerAttributes->setAnticipation(rollStat());
    adventurerAttributes->setFocus(rollStat());
    adventurerAttributesRepository.add(*adventurerAttributes);

    auto staffAttributes = std::make_unique<StaffAttributes>();
    staffAttributes->setNetwork(rollStat());
    staffAttributes->setRateAdventurer(rollStat());
    staffAttributes->setRatePotential(rollStat());
    staffAttributesRepository.add(*staffAttributes);

    character.setAdventurerAttributes(std::move(adventurerAttributes));
    character.setStaffAttributes(std::move(staffAttributes));
}