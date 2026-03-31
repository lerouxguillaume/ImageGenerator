#pragma once

#include "../entities/Guild.hpp"
#include "../repositories/GuildRepository.hpp"

class GuildService {
public:
    Guild createGuild(const char* name, int gold, bool isPlayerGuild);
private:
    GuildRepository guildRepository;
};
