#include "QuestTemplateService.hpp"

#include "Generators/QuestTextGenerator.hpp"

QuestTemplate QuestTemplateService::createQuestTemplate() {
    QuestTemplate questTemplate;
    questTemplate.setName(QuestTextGenerator::generateQuestName(QuestType::Quest));
    questTemplate.setDescription(QuestTextGenerator::generateQuestDescription(PartyRole::Leader));

    questTemplateRepository.add(questTemplate);

    return questTemplate;
}
