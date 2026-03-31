#include "PortraitGenerationService.h"
#include "../portraits/PortraitGeneratorAi.hpp"

PortraitGenerationService::PortraitGenerationService() = default;

PortraitGenerationService::~PortraitGenerationService() {
    cleanupGeneration();
}

void PortraitGenerationService::startGeneration(Race race, Gender gender, const GenerationParams& params) {
    // Clean up any previous generation
    cleanupGeneration();
    
    // Reset state
    generationDone_ = false;
    generationStep_ = 0;
    
    // Start generation thread
    generationThread_ = std::thread(&PortraitGenerationService::generationThreadFunc, this, 
                                    race, gender, params);
}

bool PortraitGenerationService::isGenerating() const {
    return generationThread_.joinable() && !generationDone_.load();
}

int PortraitGenerationService::getCurrentStep() const {
    return generationStep_.load();
}

bool PortraitGenerationService::isGenerationComplete() const {
    return generationDone_.load();
}

void PortraitGenerationService::cancelGeneration() {
    // TODO: Implement cancellation
    cleanupGeneration();
}



void PortraitGenerationService::generationThreadFunc(const Race race, Gender gender, const GenerationParams& params) {
    // Convert our params to the format expected by PortraitGeneratorAi
    ::GenerationParams aiParams;
    aiParams.numSteps = params.numSteps;
    aiParams.guidanceScale = params.guidanceScale;

    PortraitGeneratorAi::generatePortrait(race, gender, aiParams, &generationStep_);
}

void PortraitGenerationService::cleanupGeneration() {
    if (generationThread_.joinable()) {
        generationThread_.join();
    }
}