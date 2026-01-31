
//
// Created by Jonathan on 04/01/2026.
//

#include "hinge_map.h"
#include "math_utils.h"
// Rotational method definitions.
void Rotational::get_animation(double theta) {}

arma::vec6 Rotational::get_hinge_map() { return hinge_map; }

arma::vec6 Rotational::get_transposed_hinge_map() {
  return transpose_hinge_map;
}

arma::vec6 Rotational::compute_coriolis(const arma::vec &theta_dot,
                                        const arma::vec &velocity_vector) {

  const arma::mat tilde_velocity_result = tilde_velocity_fast(velocity_vector);
  const arma::mat delta_velocity = transpose_hinge_map * theta_dot;
  const arma::mat delta_velocity_bar = bar_velocity_fast(delta_velocity);

  return tilde_velocity_result * delta_velocity -
         delta_velocity_bar * delta_velocity;
}

arma::vec::fixed<6>
Body::coriolis_vector(const arma::vec &generalized_positions,
                      const arma::mat &velocity_vector,
                      const arma::vec &generalized_velocities) {

  const arma::mat tilde_velocity_result = tilde_velocity_fast(velocity_vector);
  const arma::mat delta_velocity =
      get_transposed_hinge_map(generalized_positions) * generalized_velocities;
  const arma::mat delta_velocity_bar = bar_velocity_fast(delta_velocity);

  return tilde_velocity_result * delta_velocity -
         delta_velocity_bar * delta_velocity;
}
// Prismatic method definitions

// Screw method definitions

// Universal method definitions

// Cylindrical method definitoins
