#include "AssetValidator.hpp"
#include <string>

namespace AssetValidator {

Result validate(const sf::Image& img, const AssetSpec& spec) {
    Result result;
    const unsigned w = img.getSize().x;
    const unsigned h = img.getSize().y;
    if (w == 0 || h == 0) return result;

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

    return result;
}

} // namespace AssetValidator
