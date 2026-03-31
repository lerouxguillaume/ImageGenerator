#pragma once
#include <memory>
#include <vector>
#include <random>
#include <SFML/Graphics/Texture.hpp>

#include "PortraitLayer.hpp"

class PortraitGenerator {
public:
    sf::Texture generate(unsigned seed, Race race, Gender gender);

private:
    std::unique_ptr<PortraitLayer> createSkinLayer(PortraitContext context);
    std::unique_ptr<PortraitLayer> createNeckLayer(PortraitContext context);
    std::unique_ptr<PortraitLayer> createHairLayer(PortraitContext context);
    std::unique_ptr<PortraitLayer> createFacialHairLayer(PortraitContext context);
    std::unique_ptr<PortraitLayer> createEyeLayer(PortraitContext context);
    std::unique_ptr<PortraitLayer> createEarLayer(PortraitContext context);
    std::unique_ptr<PortraitLayer> createEyebrowLayer(PortraitContext context);
    std::unique_ptr<PortraitLayer> createNoseLayer(PortraitContext context);
    std::unique_ptr<PortraitLayer> createMouthLayer(PortraitContext context);
    std::unique_ptr<PortraitLayer> createScarLayer(PortraitContext context);

    sf::Texture combineLayers(const std::vector<std::unique_ptr<PortraitLayer>>& layers);
    PortraitContext createContext(unsigned seed, Race race, Gender gender);
    static void ellipse(sf::RenderTexture& rt, float x, float y, float rx, float ry, sf::Color col);

    int ri(std::mt19937& prng, int lo, int hi) {
        return std::uniform_int_distribution<int>(lo, hi)(prng);
    }
    sf::Uint8 uc(float v) {
        return static_cast<sf::Uint8>(std::max(0.f, std::min(255.f, v)));
    }
};
