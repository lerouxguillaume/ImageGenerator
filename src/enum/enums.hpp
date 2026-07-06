#pragma once
#include <cstdint>
#include <string>

// ─── Inference ────────────────────────────────────────────────────────────────

// Diffusion model architecture. Drives resolution (512 vs 1024), dual text encoder,
// and SDXL-specific UNet inputs (text_embeds, time_ids).
enum class ModelType : std::uint8_t { SD15, SDXL };

// Stage reported by the generation thread to the UI thread via atomic<GenerationStage>.
// Generation cycles: LoadingModel → EncodingText → (EncodingImage) → Denoising
//   → DecodingImage → Done.
enum class GenerationStage : int {
    Idle = 0,
    LoadingModel,
    EncodingText,
    EncodingImage,    // img2img VAE encode only
    Denoising,        // step counter is meaningful while in this stage
    DecodingImage,
    Done
};

// ─── Game domain ──────────────────────────────────────────────────────────────

// Adventurer archetype; None = unassigned/generic NPC.
enum class AdventurerClass : std::uint8_t { None, Warrior, Rogue, Mage, Cleric };

// Character race — used both for portrait prompt construction and game logic.
enum class Race : std::uint8_t { Human, Elf, Dwarf, Orc };

enum class Gender : std::uint8_t { Male, Female };

// Adventurer availability state.
enum class Status : std::uint8_t { Idle, Assigned, Injured };

enum class QuestType : std::uint8_t { Quest, Dungeon };

// UI phase within a quest/dungeon flow.
enum class Phase : std::uint8_t { Planning, Results };

enum class Role : std::uint8_t { Adventurer, Staff };

// Position within a party formation.
enum class PartyRole : std::uint8_t {
    Leader,
    Tank,
    Healer,
    Scout,
    DPS
};
