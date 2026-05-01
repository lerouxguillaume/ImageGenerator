#include "GeneratedAssetProcessor.hpp"
#include "AssetArtifactStore.hpp"
#include "AssetPostProcessor.hpp"
#include "../postprocess/AlphaCutout.hpp"
#include <SFML/Graphics/Image.hpp>
#include <fstream>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

namespace GeneratedAssetProcessor {

namespace {

AlphaCutout::Options toAlphaCutoutOptions(const AlphaCutoutSpec& spec) {
    AlphaCutout::Options opts;
    opts.tolerance = spec.tolerance;
    opts.featherRadius = spec.featherRadius;
    opts.defringe = spec.defringe;
    return opts;
}

} // namespace

Result process(const Request& request) {
    Result out;

    sf::Image sfRaw;
    if (!sfRaw.loadFromFile(request.rawPath.string()))
        return out;

    if (request.requiresTransparency)
        sfRaw = AlphaCutout::removeBackground(
            sfRaw, toAlphaCutoutOptions(request.exportSpec.alphaCutout));

    if (!request.assetMode) {
        if (request.requiresTransparency) {
            out.transparentPath = AssetArtifactStore::transparentPathFor(request.rawPath);
            out.wroteTransparent = sfRaw.saveToFile(out.transparentPath.string());
        }
        return out;
    }

    std::vector<sf::Uint8> pixels(sfRaw.getPixelsPtr(),
                                  sfRaw.getPixelsPtr() + sfRaw.getSize().x * sfRaw.getSize().y * 4);
    cv::Mat rgba(static_cast<int>(sfRaw.getSize().y), static_cast<int>(sfRaw.getSize().x),
                 CV_8UC4, pixels.data());
    cv::Mat bgra;
    cv::cvtColor(rgba, bgra, cv::COLOR_RGBA2BGRA);

    const AssetProcessResult result = AssetPostProcessor::process(bgra, request.exportSpec);
    out.processedPath = request.processedDir / request.rawPath.filename();
    out.wroteProcessed = cv::imwrite(out.processedPath.string(), result.image);

    nlohmann::json metaJson = toJson(result, request.exportSpec);
    if (!request.metadata.candidateRunId.empty())
        metaJson["candidateRunId"] = request.metadata.candidateRunId;
    if (!request.metadata.candidateStage.empty())
        metaJson["candidateStage"] = request.metadata.candidateStage;
    metaJson["referenceUsed"] = request.metadata.referenceUsed;
    metaJson["referenceImage"] = request.metadata.referenceUsed
        ? request.metadata.referenceImage : std::string{};
    metaJson["structureStrength"] = request.metadata.referenceUsed
        ? request.metadata.structureStrength : 0.0f;
    metaJson["refinementUsed"] = request.metadata.refinementUsed;
    metaJson["refinementSource"] = request.metadata.refinementSource;
    metaJson["refinementStrength"] = request.metadata.refinementUsed
        ? request.metadata.refinementStrength : 0.0f;

    std::ofstream meta(AssetArtifactStore::metadataPathFor(out.processedPath));
    meta << metaJson.dump(4);
    return out;
}

}
