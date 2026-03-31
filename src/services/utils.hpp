#pragma once
#include <random>
#include <string>
#include <vector>

extern std::mt19937 rng;
extern std::vector<std::string> namePool;
extern const std::vector<std::string> ADV_NAMES;

int   randInt(int lo, int hi);
float randFloat();
std::string pickName();

int rollStat();