//
// Created by Jonathan on 28/09/2025.
//

#pragma once
#include <armadillo>

#include "data_types.h"

void plot_data(arma::mat store_data, ParsedData parsed_thetas, int n, std::string data_type);
void plot_thetas(ParsedData parsed_thetas, int dofs);
void plot_generalized_forces(arma::mat store_generalized_forces, ParsedData parsed_thetas);