#include "../entities/QuestTemplate.hpp"
#include "../repositories/QuestTemplateRepository.hpp"

class QuestTemplateService {
public:
    QuestTemplate createQuestTemplate();

private:
    QuestTemplateRepository questTemplateRepository;
};

