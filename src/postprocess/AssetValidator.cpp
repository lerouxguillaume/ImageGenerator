#include "AssetValidator.hpp"
#include <algorithm>
#include <cmath>
#include <optional>
#include <string>

namespace AssetValidator {

namespace {
std::optional<OccupiedBounds> computeOpaqueBounds(const sf::Image& img) {
    const unsigned w = img.getSize().x;
    const unsigned h = img.getSize().y;
    bool found = false;
    unsigned minX = w;
    unsigned minY = h;
    unsigned maxX = 0;
    unsigned maxY = 0;
    for (unsigned py = 0; py < h; ++py) {
        for (unsigned px = 0; px < w; ++px) {
            if (img.getPixel(px, py).a < 128) continue;
            found = true;
            minX = std::min(minX, px);
            minY = std::min(minY, py);
            maxX = std::max(maxX, px);
            maxY = std::max(maxY, py);
        }
    }
    if (!found) return std::nullopt;
    return OccupiedBounds{
        static_cast<int>(minX),
        static_cast<int>(minY),
        static_cast<int>(maxX - minX + 1),
        static_cast<int>(maxY - minY + 1)
    };
}
}

Result validate(const sf::Image& img, const AssetSpec& spec) {
    Result result;
    const unsigned w = img.getSize().x;
    const unsigned h = img.getSize().y;
    if (w == 0 || h == 0) return result;
    const auto actualBounds = computeOpaqueBounds(img);

    // ── Canvas size ───────────────────────────────────────────────────────────
    if (spec.canvasWidth > 0 || spec.canvasHeight > 0) {
        const int sw = spec.canvasWidth  > 0 ? spec.canvasWidth  : static_cast<int>(w);
        const int sh = spec.canvasHeight > 0 ? spec.canvasHeight : static_cast<int>(h);
        const std::string detail = std::to_string(w) + "×" + std::to_string(h);
        if (static_cast<int>(w) == sw && static_cast<int>(h) == sh) {
            result.checks.push_back({"Canvas", 0, detail});
        } else {
            const int status = spec.validation.enforceCanvasSize ? 2 : 1;
            result.checks.push_back({"Canvas", status,
                detail + " (want " + std::to_string(sw) + "×" + std::to_string(sh) + ")"});
        }
    }

    // ── Alpha presence ────────────────────────────────────────────────────────
    if (spec.requiresTransparency) {
        bool hasTransparent = false;
        for (unsigned py = 0; py < h && !hasTransparent; ++py)
            for (unsigned px = 0; px < w && !hasTransparent; ++px)
                if (img.getPixel(px, py).a < 255) hasTransparent = true;

        const int status = hasTransparent ? 0
                         : (spec.validation.enforceTransparency ? 2 : 1);
        result.checks.push_back({"Alpha", status,
            hasTransparent ? "has transparency" : "no transparent pixels"});
    }

    // ── Fill ratio ────────────────────────────────────────────────────────────
    {
        unsigned opaque = 0;
        for (unsigned py = 0; py < h; ++py)
            for (unsigned px = 0; px < w; ++px)
                if (img.getPixel(px, py).a >= 128) ++opaque;

        const float fill = static_cast<float>(opaque) / static_cast<float>(w * h);
        const std::string detail = std::to_string(static_cast<int>(fill * 100.f + 0.5f)) + "% fill";
        int status = 0;
        if (fill < spec.minFillRatio)       status = 2;
        else if (fill > spec.maxFillRatio)  status = 1;
        result.checks.push_back({"Fill", status, detail});
    }

    // ── Occupied bounds fit ──────────────────────────────────────────────────
    if (spec.expectedBounds.w > 0 && spec.expectedBounds.h > 0) {
        if (!actualBounds) {
            const int status = (spec.shapePolicy == ShapePolicy::Freeform) ? 1 : 2;
            result.checks.push_back({"Bounds", status, "no opaque subject"});
        } else {
            const int tol = std::max(6, static_cast<int>(std::round(std::min(w, h) * 0.04f)));
            const int warnTol = tol * 2;
            const int dx = std::abs(actualBounds->x - spec.expectedBounds.x);
            const int dy = std::abs(actualBounds->y - spec.expectedBounds.y);
            const int dw = std::abs(actualBounds->w - spec.expectedBounds.w);
            const int dh = std::abs(actualBounds->h - spec.expectedBounds.h);
            const int maxDelta = std::max(std::max(dx, dy), std::max(dw, dh));
            int status = 0;
            if (maxDelta > tol) {
                if (spec.shapePolicy == ShapePolicy::Freeform) status = 1;
                else status = (maxDelta > warnTol) ? 2 : 1;
            }
            const std::string detail =
                std::to_string(actualBounds->x) + "," + std::to_string(actualBounds->y) + " "
                + std::to_string(actualBounds->w) + "x" + std::to_string(actualBounds->h)
                + " (want "
                + std::to_string(spec.expectedBounds.x) + "," + std::to_string(spec.expectedBounds.y) + " "
                + std::to_string(spec.expectedBounds.w) + "x" + std::to_string(spec.expectedBounds.h) + ")";
            result.checks.push_back({"Bounds", status, detail});
        }
    }

    // ── Anchor fit ───────────────────────────────────────────────────────────
    if (spec.anchor.x != 0 || spec.anchor.y != 0) {
        if (!actualBounds) {
            const int status = spec.validation.enforceAnchor ? 2 : 1;
            result.checks.push_back({"Anchor", status, "no opaque subject"});
        } else {
            const int actualAnchorX = actualBounds->x + actualBounds->w / 2;
            const int actualAnchorY = actualBounds->y + actualBounds->h;
            const int dx = actualAnchorX - spec.anchor.x;
            const int dy = actualAnchorY - spec.anchor.y;
            const float dist = std::sqrt(static_cast<float>(dx * dx + dy * dy));
            const float tol = std::max(8.f, std::min(static_cast<float>(w), static_cast<float>(h)) * 0.035f);
            const int status = dist <= tol ? 0 : (spec.validation.enforceAnchor ? 2 : 1);
            const std::string detail =
                std::to_string(actualAnchorX) + "," + std::to_string(actualAnchorY)
                + " (want " + std::to_string(spec.anchor.x) + "," + std::to_string(spec.anchor.y) + ")";
            result.checks.push_back({"Anchor", status, detail});
        }
    }

    return result;
}

} // namespace AssetValidator
