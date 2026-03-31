//
// Created by guillaume on 3/27/26.
//

#include "CharacterService.hpp"

Character CharacterService::createCharacter(Role role)
{
    Character character = (role == Role::Adventurer) ? Character::makeAdventurer() : Character::makeStaff();
    characterAttributesService.createCharacterAttributes(character);

    characterRepository.add(character);

    return character;
}


std::vector<std::unique_ptr<Character>> CharacterService::findAvailableAdventurers() {
    return characterRepository.getAllAdventurersAvailable();
}

std::vector<std::unique_ptr<Character>> CharacterService::findAllStaffsAvailable() {
    return characterRepository.getAllStaffsAvailable();
}