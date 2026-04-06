#pragma once
#include <cmath>
#include <SFML/Graphics.hpp>
#include "../enum/constants.hpp"

namespace Logo {

// Draws a gold medallion (4-pointed star inside a bordered circle) centred at
// (cx, cy). Works on any sf::RenderTarget — window or off-screen texture.
inline void draw(sf::RenderTarget& target, float cx, float cy, float radius) {
    // Outer circle background
    sf::CircleShape bg(radius);
    bg.setFillColor(Col::Panel);
    bg.setOutlineThickness(radius * 0.07f);
    bg.setOutlineColor(Col::Gold);
    bg.setOrigin(radius, radius);
    bg.setPosition(cx, cy);
    target.draw(bg);

    // Subtle inner ring
    const float ringR = radius * 0.82f;
    sf::CircleShape ring(ringR);
    ring.setFillColor(sf::Color::Transparent);
    ring.setOutlineThickness(radius * 0.025f);
    ring.setOutlineColor(Col::Border);
    ring.setOrigin(ringR, ringR);
    ring.setPosition(cx, cy);
    target.draw(ring);

    // 4-pointed star (8 alternating outer/inner vertices)
    sf::ConvexShape star(8);
    const float outerR = radius * 0.52f;
    const float innerR = radius * 0.20f;
    for (int i = 0; i < 8; ++i) {
        const float angleDeg = static_cast<float>(i) * 45.f - 90.f;
        const float angleRad = angleDeg * 3.14159265f / 180.f;
        const float r = (i % 2 == 0) ? outerR : innerR;
        star.setPoint(i, {cx + r * std::cos(angleRad),
                          cy + r * std::sin(angleRad)});
    }
    star.setFillColor(Col::GoldLt);
    target.draw(star);

    // Centre gem
    const float gemR = radius * 0.12f;
    sf::CircleShape gem(gemR);
    gem.setFillColor(Col::Panel);
    gem.setOutlineThickness(radius * 0.03f);
    gem.setOutlineColor(Col::Gold);
    gem.setOrigin(gemR, gemR);
    gem.setPosition(cx, cy);
    target.draw(gem);
}

// Renders the logo into a square sf::Image at the given pixel size.
// The window must already exist (OpenGL context required) before calling this.
inline sf::Image makeIconImage(unsigned size) {
    sf::RenderTexture rt;
    rt.create(size, size);
    rt.clear(sf::Color::Transparent);
    draw(rt, size / 2.f, size / 2.f, size / 2.f * 0.94f);
    rt.display();
    return rt.getTexture().copyToImage();
}

} // namespace Logo
