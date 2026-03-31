#include "utils.hpp"
#include <ctime>
#include "../config/config.hpp"


const std::vector<std::string> ADV_NAMES = {
    "Aldric","Brynn","Cedric","Dara","Ewan","Fiona","Gareth","Hild",
    "Idris","Jora","Kael","Lysa","Morn","Nessa","Orin","Petra",
    "Quinn","Reva","Soren","Talia","Urik","Vena","Wulf","Xara","Yven","Zora"
};

std::mt19937 rng(static_cast<unsigned>(std::time(nullptr)));
std::vector<std::string> namePool;

int randInt(int lo, int hi) {
    return std::uniform_int_distribution<int>(lo, hi)(rng);
}

float randFloat() {
    return std::uniform_real_distribution<float>(0.f, 1.f)(rng);
}

std::string pickName() {
    if (namePool.empty()) namePool = ADV_NAMES;
    int idx = randInt(0, static_cast<int>(namePool.size()) - 1);
    std::string n = namePool[idx];
    namePool.erase(namePool.begin() + idx);
    return n;
}

int rollStat() {
    // bell curve around 8-10, very rare chance of high values up to 20
    int v = randInt(2, 5) + randInt(2, 5) + randInt(1, 4); // range 5-14, mean ~9.5
    if (randFloat() < gBalance.burstChance)
        v += randInt(gBalance.burstMin, gBalance.burstMax); // ~4%: exceptional burst
    return std::min(v, 20);
}