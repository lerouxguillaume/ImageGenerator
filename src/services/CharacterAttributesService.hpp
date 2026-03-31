#pragma once
#include "../entities/Character.hpp"
#include "../repositories/AdventurerAttributesRepository.hpp"
#include "../repositories/StaffAttributesRepository.hpp"


class CharacterAttributesService {
    public:
        void createCharacterAttributes(Character& character);

    private:
        AdventurerAttributesRepository adventurerAttributesRepository;
        StaffAttributesRepository staffAttributesRepository;
};

