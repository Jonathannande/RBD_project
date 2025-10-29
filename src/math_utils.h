//
// Created by Jonathan on 28/09/2025.
//
#pragma once
#include <armadillo>

arma::mat::fixed<3,3> tilde(const arma::mat& vector);
arma::mat rotation_matrix(const arma::mat& rotation_vec);
arma::mat::fixed<3,3> rotation_matrix_qua(const arma::mat& rotation_vec);
arma::mat::fixed<6,6>  rb_transform(const arma::mat& rotation_vec, const arma::mat& position_vec);
arma::mat::fixed<6,6> rb_transform_transpose(const arma::mat& rbt);
arma::mat::fixed<6,6> tilde_velocity(const arma::mat& velocity_vector);
arma::mat::fixed<6,6> bar_velocity(const arma::mat& velocity_vector);
arma::vec::fixed<6> gyroscopic_force_z(const arma::mat& inertial_matrix,const arma::mat& velocity_vector);
arma::vec::fixed<6> coriolis_vector(const arma::mat& hinge_map, const arma::mat& velocity_vector, const arma::vec& generalized_velocities);
