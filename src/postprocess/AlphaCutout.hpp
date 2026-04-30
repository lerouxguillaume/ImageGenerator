#pragma once
#include <SFML/Graphics.hpp>

namespace AlphaCutout {

struct Options {
    float tolerance     = 30.f; // max Euclidean color distance from bg color (range 0–441)
    int   featherRadius = 3;    // edge feather width in pixels
    bool  defringe      = true; // erode foreground 1px to remove halo fringe
};

// Remove image background by flood-filling from the four corners.
// Returns a new image with background pixels made transparent.
// If the source already has any non-opaque pixels it is returned unchanged.
sf::Image removeBackground(const sf::Image& src, const Options& opts = {});

} // namespace AlphaCutout
