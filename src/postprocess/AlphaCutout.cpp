#include "AlphaCutout.hpp"
#include <algorithm>
#include <cmath>
#include <queue>
#include <vector>

namespace AlphaCutout {

namespace {

float colorDist(float bgR, float bgG, float bgB, sf::Color c) {
    const float dr = bgR - static_cast<float>(c.r);
    const float dg = bgG - static_cast<float>(c.g);
    const float db = bgB - static_cast<float>(c.b);
    return std::sqrt(dr*dr + dg*dg + db*db);
}

} // namespace

sf::Image removeBackground(const sf::Image& src, const Options& opts) {
    const int W = static_cast<int>(src.getSize().x);
    const int H = static_cast<int>(src.getSize().y);
    if (W <= 0 || H <= 0) return src;

    // Pass through if image already has any transparency.
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            if (src.getPixel(x, y).a < 255) return src;

    // Estimate background color as average of 3×3 regions at each corner.
    float bgR = 0.f, bgG = 0.f, bgB = 0.f;
    int count = 0;
    const int cornerX[4] = {0, W-1, 0,   W-1};
    const int cornerY[4] = {0, 0,   H-1, H-1};
    for (int ci = 0; ci < 4; ++ci) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                const int px = std::clamp(cornerX[ci]+dx, 0, W-1);
                const int py = std::clamp(cornerY[ci]+dy, 0, H-1);
                const sf::Color c = src.getPixel(px, py);
                bgR += c.r; bgG += c.g; bgB += c.b;
                ++count;
            }
        }
    }
    bgR /= count; bgG /= count; bgB /= count;

    // BFS flood fill from all four corners.
    std::vector<uint8_t> alpha(static_cast<size_t>(W * H), 255u);
    std::queue<int> q;

    constexpr int ndx[4] = {1, -1, 0, 0};
    constexpr int ndy[4] = {0, 0, 1, -1};

    auto visit = [&](int x, int y) {
        if (x < 0 || y < 0 || x >= W || y >= H) return;
        const int idx = y * W + x;
        if (alpha[idx] == 0u) return;
        if (colorDist(bgR, bgG, bgB, src.getPixel(x, y)) <= opts.tolerance) {
            alpha[idx] = 0u;
            q.push(idx);
        }
    };

    for (int ci = 0; ci < 4; ++ci)
        visit(cornerX[ci], cornerY[ci]);

    while (!q.empty()) {
        const int idx = q.front(); q.pop();
        const int px = idx % W, py = idx / W;
        for (int d = 0; d < 4; ++d)
            visit(px + ndx[d], py + ndy[d]);
    }

    // Defringe: erode foreground mask by 1px to remove bg-contaminated halo pixels.
    if (opts.defringe) {
        std::vector<bool> erode(static_cast<size_t>(W * H), false);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                if (alpha[y*W+x] == 0u) continue;
                for (int d = 0; d < 4; ++d) {
                    const int nx = x + ndx[d], ny = y + ndy[d];
                    if (nx < 0 || ny < 0 || nx >= W || ny >= H || alpha[ny*W+nx] == 0u) {
                        erode[y*W+x] = true;
                        break;
                    }
                }
            }
        for (size_t i = 0; i < static_cast<size_t>(W*H); ++i)
            if (erode[i]) alpha[i] = 0u;
    }

    // Feather: blur alpha at the fg/bg boundary — only allow alpha to decrease.
    for (int pass = 0; pass < opts.featherRadius; ++pass) {
        std::vector<uint8_t> blurred = alpha;
        for (int y = 1; y < H-1; ++y) {
            for (int x = 1; x < W-1; ++x) {
                if (alpha[y*W+x] == 0u) continue;
                const uint32_t sum =
                    static_cast<uint32_t>(alpha[y*W+x]) * 4u
                    + alpha[(y-1)*W+x]
                    + alpha[(y+1)*W+x]
                    + alpha[y*W+x-1]
                    + alpha[y*W+x+1];
                blurred[y*W+x] = static_cast<uint8_t>(
                    std::min(static_cast<uint32_t>(alpha[y*W+x]), sum / 8u));
            }
        }
        alpha = blurred;
    }

    // Compose output.
    sf::Image result;
    result.create(static_cast<unsigned>(W), static_cast<unsigned>(H), sf::Color::Transparent);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            sf::Color c = src.getPixel(x, y);
            c.a = alpha[y*W+x];
            result.setPixel(x, y, c);
        }
    return result;
}

} // namespace AlphaCutout
