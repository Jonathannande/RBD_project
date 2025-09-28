//
// Created by Jonathan on 28/09/2025.
//
#pragma once
#include <vector>

struct ParsedData {
    std::vector<double> times;
    std::vector<std::vector<double>> accel;
    std::vector<std::vector<double>> vel;
    std::vector<std::vector<double>> pos;
};