//
// Created by Jonathan on 28/09/2025.
//
#pragma once
#include <armadillo>

arma::mat tilde(arma::mat vector);
arma::mat rotation_matrix(arma::mat rotation_vec);
arma::mat rotation_matrix_qua(const arma::mat& rotation_vec);
arma::mat rb_transform(arma::mat rotation_vec, arma::mat position_vec);
arma::mat rb_transform_transpose(arma::mat rbt);
arma::mat tilde_velocity(arma::mat velocity_vector);
arma::mat bar_velocity(arma::mat velocity_vector);
arma::mat gyroscopic_force_z(arma::mat inertial_matrix, arma::mat velocity_vector);
arma::mat coriolis_vector(arma::mat hinge_map, arma::mat velocity_vector, arma::vec generalized_velocities);
arma::mat find_spatial_operator(arma::mat rigid_body_transform_vector, int n);