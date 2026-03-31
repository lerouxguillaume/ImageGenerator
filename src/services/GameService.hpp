#pragma once

#include "CharacterService.hpp"
#include "../entities/Game.hpp"
#include "../repositories/GameRepository.hpp"
#include "../repositories/QuestInstanceRepository.hpp"
#include "../repositories/QuestTemplateRepository.hpp"
#include "GuildService.hpp"
#include "QuestTemplateService.hpp"

class GameService {
public:
    Game createNewGame(const char* guildName);
private:
    GameRepository gameRepository;
    QuestTemplateRepository questTemplateRepository;
    QuestInstanceRepository questInstanceRepository;
    GuildService guildService;
    CharacterService characterService;
    QuestTemplateService questTemplateService;
};
