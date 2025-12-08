//
// Created by Jonathan on 10/1/2025.
//

#include "system_of_bodies.h"
#include "bodies.h"
#include "data_types.h"
#include "math_utils.h"
#include <armadillo>
#include <boost/math/constants/constants.hpp>
#include <boost/math/quadrature/gauss.hpp>
#include <boost/numeric/odeint.hpp>
#include <iostream>
#include <memory>
#include <vector>

using namespace boost::math::float_constants;
using namespace boost::numeric::odeint;

// Formatting

// conversion of the state vector to arma::vec

void SystemOfBodies::to_arma_vec(const std::vector<double> &y,
                                 forward_parameters &p) const {

  std::vector<double> position_vector(y.begin(), y.end() - system_total_dof);
  std::vector<double> velocity_vector(y.begin() + system_total_dof, y.end());

  int start_idx = 0;
  for (int i = 0; i < n; ++i) {
    p.theta[i] = arma::conv_to<arma::vec>::from(std::vector<double>(
        position_vector.begin() + start_idx,
        position_vector.begin() + start_idx + system_dofs_distribution[i]));
    p.theta_dot[i] = arma::conv_to<arma::vec>::from(std::vector<double>(
        velocity_vector.begin() + start_idx,
        velocity_vector.begin() + start_idx + system_dofs_distribution[i]));
    start_idx += system_dofs_distribution[i];
  }
}

// conversion of the arma::vec representation of the state vector back to
// std::vector
//

void SystemOfBodies::to_std_vec(std::vector<double> &dydt,
                                forward_parameters &p) const {
  std::vector<double> return_vector;
  std::vector<double> temp;
  for (int i = 0; i < n; ++i) {
    temp = arma::conv_to<std::vector<double>>::from(p.theta_dot[i]);
    return_vector.insert(return_vector.end(), temp.begin(), temp.end());
  }
  for (int i = 0; i < n; ++i) {
    temp = arma::conv_to<std::vector<double>>::from(p.theta_ddot[i]);
    return_vector.insert(return_vector.end(), temp.begin(), temp.end());
  }
  for (size_t i = 0; i < system_total_dof; ++i) {
    dydt[i] = return_vector[i];
    dydt[system_total_dof + i] = return_vector[system_total_dof + i];
  }

  p.dydt_out = dydt;
}

// Formats a more readable results struct, which plotting also depends on.
ParsedData
SystemOfBodies::parseResults(const std::vector<double> &results) const {

  const size_t row_size = 1 + 3 * system_total_dof;
  std::cout << row_size << std::endl;
  const size_t steps = results.size() / row_size;
  std::cout << "results.size(): " << results.size() << "steps: " << steps
            << std::endl;

  ParsedData data;
  data.times.resize(steps);
  data.pos.assign(system_total_dof, std::vector<double>(steps));
  data.vel.assign(system_total_dof, std::vector<double>(steps));
  data.accel.assign(system_total_dof, std::vector<double>(steps));

  size_t offset;

  for (size_t s = 0; s < steps; s++) {
    offset = s * row_size;
    data.times[s] = results[offset];

    for (size_t i = 0; i < system_total_dof; i++) {
      data.pos[i][s] = results[offset + 1 + i];
      data.vel[i][s] = results[offset + 1 + system_total_dof + i];
      data.accel[i][s] = results[offset + 1 + 2 * system_total_dof + i];
    }
  }

  return data;
}

// System/body transform methods

// this function receives the generalized positions at each time step, converts
// the dofs of each body and uses it to compute the input vector needed for the
// spatial operator.
arma::mat SystemOfBodies::find_spatial_operator_input_vector(
    const std::vector<arma::vec> &state) const {
  arma::mat return_vector = arma::zeros(n, 6);

  // uses the modified theta
  for (int i = 0; i < n; ++i) {
    return_vector.row(i) = (bodies[i]->hinge_map.t() * state[i] +
                            (bodies[i]->hinge_pos - bodies[i]->out_hinge_pos))
                               .t();
  }
  return return_vector;
}

std::vector<arma::mat::fixed<6, 6>> SystemOfBodies::find_spatial_operator(
    const std::vector<arma::vec> &state) const {

  arma::mat rigid_body_transform_vector =
      find_spatial_operator_input_vector(state);
  arma::mat I = arma::eye(6, 6);
  std::vector<arma::mat::fixed<6, 6>> spatial_operator(n + 2);
  spatial_operator.reserve(n + 2);
  spatial_operator[0] = I;
  spatial_operator[n + 1] = I;

  for (int i = 1; i < n + 1; ++i) {

    spatial_operator[i] =
        rb_transform(rigid_body_transform_vector.row(i - 1).cols(0, 2),
                     rigid_body_transform_vector.row(i - 1).cols(3, 5));
  }

  return spatial_operator;
}

void SystemOfBodies::set_stepper_type(const bool &is_dynamic) {
  has_dynamic_time_step = is_dynamic;
}

// System structure

// method run on body insertion into a given system
void SystemOfBodies::create_body(std::unique_ptr<Body> any_body) {
  // Move the unique_ptr into the container and get a pointer to the inserted
  // object
  const auto it =
      bodies.insert(bodies.begin(), std::move(any_body)); // Move ownership
  const Body *inserted_body = it->get(); // Retrieve the raw pointer

  ++n;

  system_dofs_distribution.insert(system_dofs_distribution.begin(),
                                  inserted_body->hinge_map.n_rows);

  system_total_dof += inserted_body->hinge_map.n_rows;
}
// sets default id's to a chain uses mathmatical indexing (starts at 1 ends at
// n)
void SystemOfBodies::prep_system() {
  for (size_t i = 0; i < n; i++) {
    bodies[i]->body_ID = i + 1;
    bodies[i]->parent_ID = i + 2;
    bodies[i]->children_ID.push_back(i);
  }

  update_system_state();
}

// sets a given bodies uses mathematical indexing (starts at 1 ends at n)
void SystemOfBodies::set_parent(const int &idx, const int &parent,
                                const bool &is_only_child) {
  bodies[idx - 1]->parent_ID = parent;
  if (is_only_child) {
    bodies[parent - 1]->children_ID.resize(1);
    bodies[parent - 1]->children_ID[0] = idx;
  } else {
    bodies[parent - 1]->children_ID.push_back(idx);
  }
}

void SystemOfBodies::update_system_state() {

  std::vector<double> state(2 * system_total_dof, 0.0);
  int offset = 0;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < system_dofs_distribution[i]; ++j) {

      state[offset + j] = bodies[i]->state(j);
      std::cout << "State: " << offset + j
                << " Position set to: " << bodies[i]->state(j)
                << " for body: " << i + 1 << " and dof: " << j + 1 << std::endl;
      state[system_total_dof + offset + j] =
          bodies[i]->state(j + system_dofs_distribution[i]);
      std::cout << "State: " << system_total_dof + offset + j
                << " Velocity set to "
                << bodies[i]->state(j + system_dofs_distribution[i])
                << " for body: " << i + 1 << " and dof: " << j + 1 << std::endl;
    }
    offset += system_dofs_distribution[i];
  }
  system_state = state;
}
