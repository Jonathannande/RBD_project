//
// Created by Jonathan on 28/09/2025.
//

#include "bodies.h"
#include "math_utils.h"
#include <cmath>

// Body class implementations
Body::Body(double mass) : m(mass) {
  state = {0, 0}; // state of position then velocity respectively
  bool is_dependent_hingemap = false; // used for hinge maps with dependence on
}

// Setting methods

void Body::set_position_vec_hinge(const arma::vec &input_vec) {

  hinge_pos = join_vert(arma::zeros(3, 1), input_vec);
}

void Body::set_outboard_position_vec_hinge(const arma::vec &input_vec) {

  out_hinge_pos = join_vert(arma::zeros(3, 1), input_vec);
}

void Body::set_outboard_position_vec_hinge_push(const arma::vec &input_vec) {
  out_hinge_tree.push_back(join_vert(arma::zeros(3, 1), input_vec));
}

void Body::set_hinge_map(const std::string &type,
                         const std::vector<double> &params) {

  if (type == "rotational") {
    hinge_map = {0, 0, 1, 0, 0, 0};
    is_dependent_hinge_map = false;
  } else if (type == "prismatic") {
    hinge_map = arma::mat{0, 0, 0, 0, 1, 0};
    is_dependent_hinge_map = false;
  } else if (type == "cylindrical") {
    hinge_map = arma::zeros(2, 6);
    hinge_map(0, 2) = 1;
    hinge_map(1, 5) = 1;
    is_dependent_hinge_map = false;
  } else if (type == "helical") {
    double pitch = params.size() > 0 ? params[0] : 1.0;
    hinge_map = arma::mat{0, 1, 0, 0, pitch, 0};
  } else if (type == "spherical") {
    hinge_map = arma::zeros(3, 6);
    hinge_map(0, 0) = 1;
    hinge_map(1, 1) = 1;
    hinge_map(2, 2) = 1;
    is_dependent_hinge_map = false;
  } else if (type == "universal") {

    is_dependent_hinge_map = true;
    // Dummy hinge map for dof computations of the system
    hinge_map = arma::zeros(2, 6);

    compute_hinge_map = [params](const arma::vec &theta) {
      arma::mat h = arma::zeros(2, 6);
      h(0, 0) = 1;
      h(1, 1) = cos(theta(0));
      h(1, 2) = sin(theta(0));
      return h;
    };
  }
  if (!is_dependent_hinge_map) {
    transpose_hinge_map = hinge_map.t();
  }
}

arma::mat Body::get_hinge_map(const arma::vec &theta) const {
  return is_dependent_hinge_map ? compute_hinge_map(theta) : hinge_map;
}

arma::mat Body::get_transposed_hinge_map(const arma::vec &theta) const {
  return is_dependent_hinge_map ? compute_hinge_map(theta).t()
                                : transpose_hinge_map;
}

void Body::set_hinge_state(const arma::vec &hinge_state_input) {
  if (2 * hinge_map.n_rows == hinge_state_input.n_rows) {
    state = hinge_state_input;
  }
}

// Inverse specific

void Body::set_position_for_inverse_dyn(const std::string func) {
  inverse_dynamics_funcs.push_back(func);
}

void Body::set_velocity_for_inverse_dyn(const std::string func) {
  inverse_dynamics_funcs.push_back(func);
}

void Body::set_acceleration_for_inverse_dyn(const std::string func) {
  inverse_dynamics_funcs.push_back(func);
}

// The order of which you give the functions matter.
void Body::set_functions_for_inverse_dyn(
    const std::vector<std::string> &funcs) {
  const int n = state.size() / 2;
  for (int i = 0; i < n; ++i) {
    set_position_for_inverse_dyn(funcs[i * 3]);
    set_velocity_for_inverse_dyn(funcs[i * 3 + 1]);
    set_acceleration_for_inverse_dyn(funcs[i * 3 + 2]);
  }
}

// Rectangle class implementations
Rectangle_computed::Rectangle_computed(const double length, const double width,
                                       const double height, const double mass)
    : Body(mass), l(length), b(width), h(height) {}

void Rectangle_computed::compute_inertia_matrix() {
  // l,b, and h refer to the dimensions of length, breadth, and height
  // inertial matrix
  const arma::mat inertial_matrix_rectangle = {
      {m * (1.0 / 12.0) * (std::pow(b, 2) + std::pow(l, 2)), 0, 0},
      {0, m * (1.0 / 12.0) * (std::pow(b, 2) + std::pow(h, 2)), 0},
      {0, 0, m * (1.0 / 12.0) * (std::pow(h, 2) + std::pow(l, 2))}};

  // parallel axis theorem
  const arma::mat position_vec_tilde = tilde(-hinge_pos.rows(3, 5));
  const arma::mat I_corner = inertial_matrix_rectangle +
                             m * (position_vec_tilde.t() * position_vec_tilde);
  const arma::mat I = arma::eye(3, 3);
  const arma::mat position_vec_tilde_non_neg = tilde(hinge_pos.rows(3, 5));

  // join the matrices and construct the inertia matrix
  const arma::mat upper = join_horiz(I_corner, m * position_vec_tilde_non_neg);
  const arma::mat lower = join_horiz(m * position_vec_tilde, m * I);

  inertial_matrix = join_vert(upper, lower);
}
