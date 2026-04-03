#pragma once
#include <cstdint>
#include <string>

// ─── Enums ────────────────────────────────────────────────────────────────────
enum class ModelType : std::uint8_t { SD15, SDXL };

enum class AdventurerClass : std::uint8_t { None, Warrior, Rogue, Mage, Cleric };

enum class Race : std::uint8_t { Human, Elf, Dwarf, Orc };

enum class Gender : std::uint8_t { Male, Female };

enum class Status : std::uint8_t { Idle, Assigned, Injured };

enum class QuestType : std::uint8_t { Quest, Dungeon };

enum class Phase : std::uint8_t { Planning, Results };

enum class Role : std::uint8_t { Adventurer, Staff };

enum class PartyRole : std::uint8_t {
    Leader,
    Tank,
    Healer,
    Scout,
    DPS
};
