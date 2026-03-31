#include "PortraitGenerationService.h"
#include "../portraits/PortraitGeneratorAi.hpp"

PortraitGenerationService::PortraitGenerationService() = default;

void PortraitGenerationService::emitProgressEvent(int currentStep, int totalSteps) {
    EventSystem::getInstance().emit(PortraitGenerationProgressEvent(currentStep, totalSteps));
}

PortraitGenerationService::~PortraitGenerationService() {
    cleanupGeneration();
}

void PortraitGenerationService::startGeneration(const Character& character, const GenerationParams& params) {
    // Clean up any previous generation
    cleanupGeneration();
    
    // Reset state
    generationDone_ = false;
    generationStep_ = 0;
    
    // Start generation thread
    generationThread_ = std::thread(&PortraitGenerationService::generationThreadFunc, this, 
                                    std::cref(character), params);
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



void PortraitGenerationService::generationThreadFunc(const Character& character, const GenerationParams& params) {
    // Emit start event
    EventSystem::getInstance().emit(PortraitGenerationStartedEvent());
    
    try {
        // Convert our params to the format expected by PortraitGeneratorAi
        ::GenerationParams aiParams;
        aiParams.numSteps = params.numSteps;
        aiParams.guidanceScale = params.guidanceScale;

        PortraitGeneratorAi::generatePortrait(character, aiParams, &generationStep_);
        
        // Generation completed successfully
        generationDone_ = true;
        
        // Emit completion event
        EventSystem::getInstance().emit(PortraitGenerationCompletedEvent(true));
        EventSystem::getInstance().emit(PortraitReloadRequestedEvent());
    } catch (const std::exception& e) {
        generationDone_ = true;
        EventSystem::getInstance().emit(PortraitGenerationCompletedEvent(false));
    }
}

void PortraitGenerationService::cleanupGeneration() {
    if (generationThread_.joinable()) {
        generationThread_.join();
    }
}