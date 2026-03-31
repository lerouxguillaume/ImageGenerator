#include "GameService.hpp"

int INITIAL_ADVENTURER_POOL = 100;
int INITIAL_QUEST_POOL = 10;

Game GameService::createNewGame(const char* guildName)
{
    Game game;
    game.setIsCurrent(true);
    gameRepository.add(game);

    guildService.createGuild(guildName, 100, true);

    for (int i = 0; i < INITIAL_ADVENTURER_POOL; i++) {
        characterService.createCharacter(Role::Adventurer);
    }

    for (int i = 0; i < INITIAL_QUEST_POOL; i++) {
        questTemplateService.createQuestTemplate();
    }

    return game;
}