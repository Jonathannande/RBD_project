//
// Created by Jonathan on 28/09/2025.
//

#pragma once
#include <armadillo>
#include <string>

class Body {
public:
  // hinge types
  arma::mat hinge_map = {0, 0, 1, 0, 0, 0}; // Defualt hingemap
  arma::mat transpose_hinge_map = hinge_map.t();
  std::string hinge_type;
  bool is_dependent_hinge_map; // is used for 2 DOF hinge

  // own hinge position, vector position(s) of children
  arma::vec::fixed<6> hinge_pos;     // 6X1 position vector used for inertia
                                     // computations - not used in BWA
  arma::vec::fixed<6> out_hinge_pos; // 6X1 position vector used for transform
                                     // from center of mass

  std::vector<arma::vec::fixed<6>>
      out_hinge_tree; // the different positions children may have on the body.

  // Graph attributes
  int body_ID;
  std::vector<int> children_ID;
  int parent_ID;

  // arma::vec rotation_vec hinge; //should maybe be used the general
  // formulation of the program
  arma::vec state = {
      0, 0}; // Initial conditions of body,uses generalized position/velocity.
             // structure - state = {p_1,p_2,...,p_ndof,v_1,v_2,...,v_ndof)
  arma::mat::fixed<6, 6> inertial_matrix; // Inertial matrix
  std::vector<std::string>
      inverse_dynamics_funcs; // saved in pairs of 3, for each dof of the body
                              // from left to right.
  const double m;             // mass

  Body(double mass);

  virtual void compute_inertia_matrix() = 0; // pure virtual
  std::function<arma::mat(const arma::vec &)> compute_hinge_map;
  std::function<arma::vec(const arma::vec &theta)> compute_hinge_state;
  std::function<arma::vec::fixed<6>(const arma::vec &theta,
                                    const arma::vec &theta_dot)>
      gradient_term;
  // Setting methods
  void set_position_vec_hinge(const arma::vec &input_vec);
  void set_outboard_position_vec_hinge(const arma::vec &input_vec);
  void set_outboard_position_vec_hinge_push(const arma::vec &input_vec);
  void set_hinge_map(const std::string &type,
                     const std::vector<double> &params = {});
  void set_hinge_state(const arma::vec &hinge_state_input);
  void set_functions_for_inverse_dyn(const std::vector<std::string> &funcs);
  void set_position_for_inverse_dyn(std::string func);
  void set_velocity_for_inverse_dyn(std::string func);
  void set_acceleration_for_inverse_dyn(std::string func);

  // gettting methods
  arma::vec::fixed<6> get_hinge_state(const arma::vec &theta) const;
  arma::mat get_hinge_map(const arma::vec &theta) const;
  arma::mat get_transposed_hinge_map(const arma::vec &theta) const;
  arma::vec::fixed<6>
  get_coriolis_vector(const arma::vec &generalized_positions,
                      const arma::mat &velocity_vector,
                      const arma::vec &generalized_velocities);

  arma::vec::fixed<6> coriolis_vector(const arma::vec &generalized_positions,
                                      const arma::mat &velocity_vector,
                                      const arma::vec &generalized_velocities);
};

class Rectangle_computed : public Body {
public:
  const double l, b, h;

  Rectangle_computed(double length, double breadth, double height, double mass);
  void compute_inertia_matrix() override;
};

class Sphere : public Body {
public:
  double r;
  // Add constructor and compute_inertia_matrix when you implement it
};
