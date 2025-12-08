//
// Created by Jonathan on 10/1/2025.
//

#ifndef MYPROJECT_SYSTEM_OF_BODIES_H
#define MYPROJECT_SYSTEM_OF_BODIES_H

#include "bodies.h"
#include "data_types.h"
#include "math_utils.h"
#include <armadillo>
#include <cstdint>
#include <memory>
#include <vector>

class SystemOfBodies {
public:
  // Simulation attributes
  const double t0{0.0};
  const double t{4.0};
  const double dt{0.01};
  const double system_gravity{9.81};
  unsigned int n{0};
  unsigned int system_total_dof = {0};
  bool has_dynamic_time_step{false};

  // System attributes
  std::vector<unsigned int> system_dofs_distribution;
  std::vector<std::unique_ptr<Body>> bodies;
  ParsedData parsed_data;
  std::vector<double> system_state;

private:
  // Forward dynamic specific attributes
  struct forward_parameters {
    std::vector<arma::mat> P_plus;
    std::vector<arma::vec> J_fractal_plus;
    std::vector<arma::vec> J_fractal;
    std::vector<arma::mat> tau_bar;
    std::vector<arma::mat> P;
    std::vector<arma::vec> accel;
    std::vector<arma::vec> accel_plus;
    std::vector<arma::vec> body_velocities;
    std::vector<arma::vec> body_forces;
    std::vector<arma::vec> G_fractal;
    std::vector<arma::mat> frac_v;
    arma::mat D;
    arma::vec eta;
    std::vector<double> dydt_out;
    int hidden_index = 0; // For body frame coordinate storage
    std::vector<arma::vec> theta;
    std::vector<arma::vec> theta_dot;
    std::vector<arma::vec> theta_ddot;
    forward_parameters(const int n_, const int system_total_dof_)
        : P_plus(n_ + 1), J_fractal_plus(n_ + 1), tau_bar(n_), P(n_),
          J_fractal(n_), body_velocities(n_ + 1), body_forces(n_ + 1),
          G_fractal(n_ + 1), frac_v(n_), dydt_out(system_total_dof_ * 2, 0.0),
          theta(n_), theta_dot(n_), theta_ddot(n_), accel(n_ + 1),
          accel_plus(n_)

    {
      accel[n_] = arma::vec(6, arma::fill::zeros);
      body_velocities[n_] = arma::vec(6, arma::fill::zeros);
      P_plus[0] = arma::mat(6, 6, arma::fill::zeros);
      J_fractal_plus[0] = arma::vec(6, arma::fill::zeros);
    }
  };

public:
  // Methods

  // Formatting methods
  ParsedData parseResults(const std::vector<double> &results) const;

  void to_arma_vec(const std::vector<double> &y, forward_parameters &p)
      const; // used for conversion of state types

  void
  to_std_vec(std::vector<double> &dydt,
             forward_parameters &p) const; // used for conversion of state types

  // System structure
  void update_system_state();

  void create_body(std::unique_ptr<Body> any_body);

  void set_BWA();

  void set_system_structure();

  void prep_system();

  void set_parent(const int &idx, const int &parent, const bool &is_only_child);

  // System/body transform methods
  arma::mat
  find_spatial_operator_input_vector(const std::vector<arma::vec> &state) const;

  arma::mat find_gravity_matrix_pseudo_acceleration();

  std::vector<arma::mat::fixed<6, 6>>
  find_spatial_operator(const std::vector<arma::vec> &state) const;

  // run simulation methods
  void solve_inverse_dynamics();

  void solve_forward_dynamics();

  // Forward dynamics specific

  void system_of_equations_forward_dynamics(const std::vector<double> &y,
                                            std::vector<double> &dydt,
                                            forward_parameters &p) const;

  void set_stepper_type(const bool &is_dynamic);

  // Inverse dynamics specific
  ParsedData inverse_run_funcs();

  std::vector<double> inverse_run_funcs_2();

  // Methods for tree dynamics

  void EOM__forward_tree(const std::vector<double> &y,
                         std::vector<double> &dydt,
                         forward_parameters &p) const;

  void solve_forward_dynamics_tree();

  std::vector<std::vector<arma::mat::fixed<6, 6>>>
  find_spatial_operator_tree(const std::vector<arma::vec> &state) const;

  std::vector<std::vector<arma::mat66>> find_spatial_operator_input_vector_tree(
      const std::vector<arma::vec> &state) const;

  int get_child_index(const int &k_index) const;

  void compute_p(const int &k, forward_parameters &p,
                 const std::vector<std::vector<arma::mat::fixed<6, 6>>>
                     &spatial_operator_dt) const;

  void compute_J_fractal(const int &k, forward_parameters &p,
                         const std::vector<std::vector<arma::mat::fixed<6, 6>>>
                             &spatial_operator_dt) const;
};

#endif // MYPROJECT_SYSTEM_OF_BODIES_H
