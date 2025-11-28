//
// Created by Jonathan on 10/7/2025.
//

#include "plotting_utils.h"
#include "system_of_bodies.h"
#include <boost/numeric/odeint/integrate/integrate_adaptive.hpp>
#include <boost/numeric/odeint/integrate/integrate_const.hpp>
#include <boost/numeric/odeint/stepper/controlled_runge_kutta.hpp>
#include <boost/numeric/odeint/stepper/dense_output_runge_kutta.hpp>
#include <boost/numeric/odeint/stepper/runge_kutta_dopri5.hpp>
#include <boost/numeric/odeint/stepper/runge_kutta_fehlberg78.hpp>
#include <chrono>
#include <vector>

using namespace boost::numeric::odeint;

// The problem with this implementation is that we cant size theta at
// compilation. Could be known through intermediary state of system prep, or
// define theta as a system attribute

void SystemOfBodies::system_of_equations_forward_dynamics(
    const std::vector<double> &y, std::vector<double> &dydt,
    forward_parameters &p) const {

  to_arma_vec(y, p);

  p.accel[n](4) = system_gravity;

  const std::vector<arma::mat::fixed<6, 6>> spatial_operator_dt =
      find_spatial_operator(p.theta);

  for (int k = n - 1; k > -1; --k) {
    p.body_velocities[k] =
        spatial_operator_dt[k + 1].t() * p.body_velocities[k + 1] +
        bodies[k]->transpose_hinge_map * p.theta_dot[k];
  }

  for (int k = 0; k < n; ++k) {

    p.P[k] = spatial_operator_dt[k] * p.P_plus[k] * spatial_operator_dt[k].t() +
             bodies[k]->inertial_matrix;

    p.D = bodies[k]->hinge_map * p.P[k] * bodies[k]->transpose_hinge_map;

    p.G_fractal[k] =
        p.P[k] * bodies[k]->transpose_hinge_map * arma::inv(trimatu(p.D));

    p.tau_bar[k] = arma::eye(6, 6) - p.G_fractal[k] * bodies[k]->hinge_map;

    p.P_plus[k + 1] = p.tau_bar[k] * p.P[k];

    p.J_fractal[k] =
        spatial_operator_dt[k] * p.J_fractal_plus[k] +
        p.P[k] * coriolis_vector(bodies[k]->transpose_hinge_map,
                                 p.body_velocities[k], p.theta_dot[k]) +
        gyroscopic_force_z(bodies[k]->inertial_matrix, p.body_velocities[k]);

    p.eta = -bodies[k]->hinge_map * p.J_fractal[k];

    p.frac_v[k] = arma::solve(trimatu(p.D), p.eta);

    p.J_fractal_plus[k + 1] = p.J_fractal[k] + p.G_fractal[k] * p.eta;
  }

  for (int k = n - 1; k > -1; --k) {

    p.accel_plus[k] = spatial_operator_dt[k + 1].t() * p.accel[k + 1];
    p.theta_ddot[k] = p.frac_v[k] - p.G_fractal[k].t() * p.accel_plus[k];
    p.accel[k] = p.accel_plus[k] +
                 bodies[k]->transpose_hinge_map * p.theta_ddot[k] +
                 coriolis_vector(bodies[k]->transpose_hinge_map,
                                 p.body_velocities[k], p.theta_dot[k]);
  }

  for (int k = 0; k < n; ++k) {
    p.body_forces[k] =
        p.P_plus[k + 1] * p.accel_plus[k] + p.J_fractal_plus[k + 1];
  }

  to_std_vec(dydt, p);
}

void SystemOfBodies::solve_forward_dynamics() {

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

    for (size_t k = 0; k < n; k++) {
      // store_accelerations.col(p.hidden_index).rows(k*6,(k+1)*6-1) =
      // p.accel.col(k);
      // store_velocities.col(p.hidden_index).rows(k*6,(k+1)*6-1) =
      // p.body_velocities.col(k);
      // store_forces.col(p.hidden_index).rows(k*6,(k+1)*6-1) =
      // p.body_forces.col(k);
    }

    // state is save from y which holds positions and velocities - and then
    // accelerations are obtained from a dummy variable as dydt is hidden in
    // solver internal state
    results.insert(results.end(), y.begin(), y.end());
    results.insert(results.end(), p.dydt_out.begin() + system_total_dof,
                   p.dydt_out.end());

    p.hidden_index++;
  };

  if (has_dynamic_time_step == false) {

    runge_kutta_fehlberg78<std::vector<double>> stepper;
    integrate_const(
        stepper,
        [&](const std::vector<double> &y, std::vector<double> &dydt, double t) {
          system_of_equations_forward_dynamics(y, dydt, p);
        },
        system_state, t0, t, dt, obs);
  } else {

    dense_output_runge_kutta<
        controlled_runge_kutta<runge_kutta_dopri5<std::vector<double>>>>
        dense;

    std::chrono::time_point start = std::chrono::high_resolution_clock::now();

    integrate_adaptive(
        dense,
        [&](const std::vector<double> &y, std::vector<double> &dydt, double t) {
          system_of_equations_forward_dynamics(y, dydt, p);
        },
        system_state, t0, t, dt, obs);

    auto end = std::chrono::high_resolution_clock::now();

    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Time: " << duration.count() << " microseconds\n";
  }
  // format generalized results
  ParsedData formatted_results = parseResults(results);
  parsed_data = formatted_results;

  // plot generalized results
  plot_thetas(formatted_results, system_total_dof);

  // plot body frame results if system is small
  if (n < 5) {
    plot_data(store_accelerations, formatted_results, n, "accelerations");
    plot_data(store_velocities, formatted_results, n, "velocities");
    plot_data(store_forces, formatted_results, n, "forces");
  }
}
