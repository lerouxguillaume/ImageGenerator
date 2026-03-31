#pragma once

#include "CharacterAttributesService.hpp"
#include "../entities/Character.hpp"
#include "../repositories/CharacterRepository.hpp"

class CharacterService {
public:
    Character createCharacter(Role role);
    std::vector<std::unique_ptr<Character>> findAvailableAdventurers();
    std::vector<std::unique_ptr<Character>> findAllStaffsAvailable();
private:
    CharacterRepository characterRepository;
    CharacterAttributesService characterAttributesService;
};
