#pragma once
#include <atomic>
#include <thread>
#include <functional>
#include "../entities/Character.hpp"
#include "../portraits/PortraitGeneratorAi.hpp"
#include "../events/EventSystem.h"

class PortraitGenerationService {
public:
    struct GenerationParams {
        int numSteps = 20;
        float guidanceScale = 7.5f;
    };

    PortraitGenerationService();
    ~PortraitGenerationService();

    // Start portrait generation
    void startGeneration(const Character& character, const GenerationParams& params);
    
    // Check if generation is in progress
    bool isGenerating() const;
    
    // Get current progress (step count)
    int getCurrentStep() const;
    
    // Check if generation completed
    bool isGenerationComplete() const;
    
    // Cancel generation
    void cancelGeneration();
    


private:
    std::atomic<bool> generationDone_{false};
    std::atomic<int> generationStep_{0};
    std::thread generationThread_;
    
    void generationThreadFunc(const Character& character, const GenerationParams& params);
    void cleanupGeneration();
    void emitProgressEvent(int currentStep, int totalSteps);
};