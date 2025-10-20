//
// Created by Jonathan on 28/09/2025.
//
#pragma once
#include <vector>

// this data type is specifically design such that it can hold both n bodies, with n dofs thus the vector in a vector. In the case of a single dof it just works like a normal vector
struct ParsedData {
    std::vector<double> times;
    std::vector<std::vector<double>> accel;
    std::vector<std::vector<double>> vel;
    std::vector<std::vector<double>> pos;
};