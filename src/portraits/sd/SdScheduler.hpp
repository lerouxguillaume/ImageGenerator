#pragma once
#include <vector>

namespace sd {

// DDPM scaled_linear alpha_bar schedule matching the diffusers SD v1.5 default.
std::vector<float> buildAlphasCumprod(int T, float beta_start, float beta_end);

// DPM++ 2M Karras sigma schedule. Returns num_steps+1 values; last entry is 0.
std::vector<float> buildKarrasSchedule(const std::vector<float>& alphas_cumprod, int num_steps);

// Map a DPM++ sigma to the nearest DDPM integer timestep.
int sigmaToTimestep(float sigma, const std::vector<float>& alphas_cumprod);

} // namespace sd