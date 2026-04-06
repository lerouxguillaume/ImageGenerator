#include "SdScheduler.hpp"
#include <cmath>

namespace sd {

std::vector<float> buildAlphasCumprod(int T, float beta_start, float beta_end) {
    float sqrt_start = std::sqrt(beta_start);
    float sqrt_end   = std::sqrt(beta_end);
    std::vector<float> alphas(T);
    float cum = 1.0f;
    for (int ti = 0; ti < T; ++ti) {
        float t_frac = static_cast<float>(ti) / static_cast<float>(T - 1);
        float beta   = std::pow(sqrt_start + t_frac * (sqrt_end - sqrt_start), 2.0f);
        cum     *= (1.0f - beta);
        alphas[ti] = cum;
    }
    return alphas;
}

std::vector<float> buildKarrasSchedule(const std::vector<float>& alphas_cumprod, int num_steps) {
    auto alphaBarToSigma = [](float ab) { return std::sqrt((1.0f - ab) / ab); };

    const float rho = 7.0f;
    int T = static_cast<int>(alphas_cumprod.size());
    float sigma_max     = alphaBarToSigma(alphas_cumprod[T - 1]);
    float sigma_min     = alphaBarToSigma(alphas_cumprod[0]);
    float inv_rho_max   = std::pow(sigma_max, 1.0f / rho);
    float inv_rho_min   = std::pow(sigma_min, 1.0f / rho);

    std::vector<float> sigmas(num_steps + 1);
    for (int s = 0; s < num_steps; ++s) {
        float ramp  = static_cast<float>(s) / static_cast<float>(num_steps - 1);
        sigmas[s]   = std::pow(inv_rho_max + ramp * (inv_rho_min - inv_rho_max), rho);
    }
    sigmas[num_steps] = 0.0f;
    return sigmas;
}

int sigmaToTimestep(float sigma, const std::vector<float>& alphas_cumprod) {
    float ab = 1.0f / (1.0f + sigma * sigma);
    int T = static_cast<int>(alphas_cumprod.size());
    int lo = 0, hi = T - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (alphas_cumprod[mid] < ab) hi = mid;
        else lo = mid + 1;
    }
    return lo;
}

} // namespace sd