//
// Created by Jonathan on 10/31/2025.
//

#include "math_utils.h"
#include "plotting_utils.h"
#include "system_of_bodies.h"
#include <armadillo>
#include <boost/numeric/odeint/integrate/integrate_adaptive.hpp>
#include <boost/numeric/odeint/integrate/integrate_const.hpp>
#include <boost/numeric/odeint/stepper/controlled_runge_kutta.hpp>
#include <boost/numeric/odeint/stepper/dense_output_runge_kutta.hpp>
#include <boost/numeric/odeint/stepper/runge_kutta_dopri5.hpp>
#include <boost/numeric/odeint/stepper/runge_kutta_fehlberg78.hpp>
#include <chrono>
#include <vector>

using namespace boost::numeric::odeint;

void SystemOfBodies::EOM__forward_tree(const std::vector<double> &y,
                                       std::vector<double> &dydt,
                                       forward_parameters &p) const {

  // method converting the state to the arma::vec format distributed over the
  // dofs of each body
  to_arma_vec(y, p);

  // Sets system gravity, currently hard-coded should ideally be more general.
  p.accel[n](4) = system_gravity;
  // p.accel[n](2) = sin(t);  // add swinging to the base body
  const std::vector<std::vector<arma::mat::fixed<6, 6>>> spatial_operator_dt =
      find_spatial_operator_tree(p.theta);

  // now we start sweeping n times in total, first kinematics which has in part
  // already been done through the spatial operator, but now velocities

  for (int k = n - 1; k > -1; --k) {

    p.body_velocities[k] =
        spatial_operator_dt[bodies[k]->parent_ID - 1][get_child_index(k)].t() *
            p.body_velocities[bodies[k]->parent_ID - 1] +
        bodies[k]->get_transposed_hinge_map(p.theta[k]) * p.theta_dot[k];
  }

  for (int k = 0; k < n; ++k) {

    compute_p(k, p, spatial_operator_dt);

    p.D = bodies[k]->get_hinge_map(p.theta[k]) * p.P[k] *
          bodies[k]->get_transposed_hinge_map(p.theta[k]);

    p.G_fractal[k] = p.P[k] * bodies[k]->get_transposed_hinge_map(p.theta[k]) *
                     arma::inv(trimatu(p.D));

    p.tau_bar[k] =
        arma::eye(6, 6) - p.G_fractal[k] * bodies[k]->get_hinge_map(p.theta[k]);

    p.P_plus[k + 1] = p.tau_bar[k] * p.P[k];

    compute_J_fractal(k, p, spatial_operator_dt);

    p.eta = -bodies[k]->get_hinge_map(p.theta[k]) * p.J_fractal[k];

    p.frac_v[k] = arma::solve(trimatu(p.D), p.eta);

    p.J_fractal_plus[k + 1] = p.J_fractal[k] + p.G_fractal[k] * p.eta;
  }

  for (int k = n - 1; k > -1; --k) {

    p.accel_plus[k] =
        spatial_operator_dt[bodies[k]->parent_ID - 1][get_child_index(k)].t() *
        p.accel[bodies[k]->parent_ID - 1];
    p.theta_ddot[k] = p.frac_v[k] - p.G_fractal[k].t() * p.accel_plus[k];
    p.accel[k] =
        p.accel_plus[k] +
        bodies[k]->get_transposed_hinge_map(p.theta[k]) * p.theta_ddot[k] +
        bodies[k]->get_coriolis_vector(p.theta[k], p.body_velocities[k],
                                       p.theta_dot[k]);
  }

  for (int k = 0; k < n; ++k) {
    p.body_forces[k] =
        p.P_plus[k + 1] * p.accel_plus[k] + p.J_fractal_plus[k + 1];
  }

  to_std_vec(dydt, p);
}

void SystemOfBodies::solve_forward_dynamics_tree() {

  forward_parameters p(n, system_total_dof);

  // single vector to store data
  std::vector<double> results;
  results.reserve((system_total_dof + 1) * t / dt + 50);

  // body frame storage
  arma::mat store_velocities = arma::zeros(n * 6, t / dt + 1);
  arma::mat store_accelerations = arma::zeros(n * 6, t / dt + 1);
  arma::mat store_forces = arma::zeros(n * 6, t / dt + 1);

  // Observer lambda
  auto obs = [&](const std::vector<double> &y, double t) {
    results.push_back(t);

    /*
    for (size_t k = 0; k < n; k++) {
            store_accelerations.col(p.hidden_index).rows(k*6,(k+1)*6-1) =
    p.accel[k]; store_velocities.col(p.hidden_index).rows(k*6,(k+1)*6-1) =
    p.body_velocities[k];
            store_forces.col(p.hidden_index).rows(k*6,(k+1)*6-1) =
    p.body_forces[k];
    }
    */

    // state is save from y which holds positions and velocities - and then
    // accelerations are obtained from a dummy variable as dydt is hidden in
    // solver internal state
    results.insert(results.end(), y.begin(), y.end());
    results.insert(results.end(), p.dydt_out.begin() + system_total_dof,
                   p.dydt_out.end());

    p.hidden_index++;
  };

  auto start = std::chrono::high_resolution_clock::now();

  if (has_dynamic_time_step == false) {

    runge_kutta_fehlberg78<std::vector<double>> stepper;
    integrate_const(
        stepper,
        [&](const std::vector<double> &y, std::vector<double> &dydt, double t) {
          EOM__forward_tree(y, dydt, p);
        },
        system_state, t0, t, dt, obs);
  } else {

    dense_output_runge_kutta<
        controlled_runge_kutta<runge_kutta_dopri5<std::vector<double>>>>
        dense;
    integrate_adaptive(
        dense,
        [&](const std::vector<double> &y, std::vector<double> &dydt, double t) {
          EOM__forward_tree(y, dydt, p);
        },
        system_state, t0, t, dt, obs);
  }

  auto end = std::chrono::high_resolution_clock::now();

  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  std::cout << "Solving time: " << duration.count() << " microseconds\n";
  // format generalized results
  ParsedData formatted_results = parseResults(results);

  // plot generalized results
  plot_thetas(formatted_results, system_total_dof);

  // plot body frame results if system is small
  if (n < 5) {
    plot_data(store_accelerations, formatted_results, n, "accelerations");
    plot_data(store_velocities, formatted_results, n, "velocities");
    plot_data(store_forces, formatted_results, n, "forces");
  }
  animate_tree(formatted_results);
}

// computes the transform vectors of each body. Stored in n+1 vector
// v[n] stores transform from inertial to base body frame
std::vector<std::vector<arma::mat66>>
SystemOfBodies::find_spatial_operator_tree(
    const std::vector<arma::vec> &state) const {

  std::vector<std::vector<arma::mat66>> return_vector(n + 1);

  return_vector[n].resize(1);

  arma::vec6 base =
      bodies[n - 1]->get_transposed_hinge_map(state[n - 1]) * state[n - 1] +
      bodies[n - 1]->hinge_pos;

  arma::vec6 b = base - bodies[n - 1]->out_hinge_tree[0];
  return_vector[n][0] = rb_transform(b.rows(0, 2), b.rows(3, 5));
  // uses the modified theta
  for (size_t i = n - 1; i > 0; --i) {

    if (bodies[i]->children_ID[0] == 0) {
      continue;
    }
    size_t num_out = bodies[i]->children_ID.size();
    return_vector[i].resize(num_out);

    for (int j = 0; j < bodies[i]->children_ID.size(); ++j) {

      arma::vec6 base =
          bodies[bodies[i]->children_ID[j] - 1]->get_transposed_hinge_map(
              state[bodies[i]->children_ID[j] - 1]) *
              state[bodies[i]->children_ID[j] - 1] +
          bodies[bodies[i]->children_ID[j] - 1]->hinge_pos;

      arma::vec6 a =
          base -
          bodies[i]->out_hinge_tree[j]; // If you encounter an error look here!!

      return_vector[bodies[i]->body_ID - 1][j] =
          rb_transform(a.rows(0, 2), a.rows(3, 5));
    }
  }
  return return_vector;
}

// This method gets the index of a child's id as it is stored in the parent's
// outboard vector.
int SystemOfBodies::get_child_index(const int &k_index) const {
  if (k_index >= n - 1) {
    return 0;
  }

  std::vector<int> child_ids =
      bodies[bodies[k_index]->parent_ID - 1]->children_ID;
  int i = 0;
  while (child_ids[i] != bodies[k_index]->body_ID) {
    i++;
  }
  return i;
}

// When having multiple children, it is necessary to sum the influences of each
// child on a body. This is done in the else statement
void SystemOfBodies::compute_p(
    const int &k, forward_parameters &p,
    const std::vector<std::vector<arma::mat::fixed<6, 6>>> &spatial_operator_dt)
    const {

  if (bodies[k]->children_ID[0] == 0) {
    p.P[k] = bodies[k]->inertial_matrix;
  } else if (bodies[k]->children_ID.size() == 1) {
    p.P[k] = spatial_operator_dt[k][0] * p.P_plus[bodies[k]->children_ID[0]] *
                 spatial_operator_dt[k][0].t() +
             bodies[k]->inertial_matrix;
  } else {
    p.P[k] = {arma::zeros<arma::mat>(6, 6)};
    for (int i = 0; i < bodies[k]->children_ID.size(); ++i) {
      p.P[k] += spatial_operator_dt[k][i] *
                p.P_plus[bodies[k]->children_ID[i]] *
                spatial_operator_dt[k][i].t();
    }
    p.P[k] += bodies[k]->inertial_matrix;
  }
}
// Same case as compute_p
void SystemOfBodies::compute_J_fractal(
    const int &k, forward_parameters &p,
    const std::vector<std::vector<arma::mat::fixed<6, 6>>> &spatial_operator_dt)
    const {
  if (bodies[k]->children_ID[0] == 0) {
    p.J_fractal[k] =
        p.P[k] * bodies[k]->get_coriolis_vector(
                     p.theta[k], p.body_velocities[k], p.theta_dot[k]) +
        gyroscopic_force_z(bodies[k]->inertial_matrix, p.body_velocities[k]);

  } else if (bodies[k]->children_ID.size() == 1) {
    p.J_fractal[k] =
        spatial_operator_dt[k][0] *
            p.J_fractal_plus[bodies[k]->children_ID[0]] +
        p.P[k] * bodies[k]->get_coriolis_vector(
                     p.theta[k], p.body_velocities[k], p.theta_dot[k]) +
        gyroscopic_force_z(bodies[k]->inertial_matrix, p.body_velocities[k]);
  } else {

    p.J_fractal[k] = arma::zeros<arma::vec>(6);
    for (size_t i = 0; i < bodies[k]->children_ID.size(); ++i) {
      p.J_fractal[k] += spatial_operator_dt[k][i] *
                        p.J_fractal_plus[bodies[k]->children_ID[i]];
    }
    p.J_fractal[k] +=
        p.P[k] * bodies[k]->get_coriolis_vector(
                     p.theta[k], p.body_velocities[k], p.theta_dot[k]) +
        gyroscopic_force_z(bodies[k]->inertial_matrix, p.body_velocities[k]);
  }
}
