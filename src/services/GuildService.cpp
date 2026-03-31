//
// Created by guillaume on 3/27/26.
//

#include "GuildService.hpp"

Guild GuildService::createGuild(const char* name, int gold, bool isPlayerGuild)
{
    Guild guild(name, gold, isPlayerGuild);
    guildRepository.add(guild);
    return guild;
}