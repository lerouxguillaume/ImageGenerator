#pragma once
#include <random>
#include <cstdint>
#include <SFML/Graphics/RenderTarget.hpp>
#include <SFML/Graphics/RenderTexture.hpp>
#include <SFML/Graphics/Sprite.hpp>

enum class Race : std::uint8_t;
enum class Gender : std::uint8_t;

struct PortraitContext {
    std::mt19937 prng;

    // Colors
    sf::Color skin{};
    sf::Color hair{};
    sf::Color iris{};
    sf::Color shadow{};
    sf::Color lip{};
    sf::Color brow{};

    // Morphology
    float fr = 0.f;
    float fh = 0.f;

    // Variants
    int hairStyle = 0;
    int fhType = 0;

    Race race{};
    Gender gender{};
};

class PortraitLayer {
public:
    sf::RenderTexture rt;

    void draw(sf::RenderTarget& target) const {
        target.draw(sf::Sprite(rt.getTexture()));
    }
};
