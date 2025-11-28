
void SystemOfBodies::system_of_equations_forward_dynamics(
    const std::vector<double> &y, std::vector<double> &dydt,
    forward_parameters &p) const {

  // method converting the state to the arma::vec format distributed over the
  // dofs of each body
  to_arma_vec(y, p);

  p.accel(4, n) = system_gravity; // should be reworked
  // p.accel(5, n) = system_gravity;  // should be reworked

  // Calling the methods
  const std::vector<arma::mat::fixed<6, 6>> spatial_operator_dt =
      find_spatial_operator(p.theta);

  // now we start sweeping n times in total, first kinematics which has in part
  // already been done through the spatial operator, but now velocities

  for (int k = n - 1; k > -1; --k) {
    p.body_velocities.col(k) =
        spatial_operator_dt[k + 1].t() * p.body_velocities.col(k + 1) +
        bodies[k]->transpose_hinge_map * p.theta_dot[k];
  }

  for (int k = 0; k < n; ++k) {

    p.P(span_k1_[k], span_k1_[k]) = spatial_operator_dt[k] *
                                        p.P_plus(span_k1_[k], span_k1_[k]) *
                                        spatial_operator_dt[k].t() +
                                    bodies[k]->inertial_matrix;

    p.D = bodies[k]->hinge_map * p.P(span_k1_[k], span_k1_[k]) *
          bodies[k]->transpose_hinge_map;

    p.G_fractal[k] = p.P(span_k1_[k], span_k1_[k]) *
                     bodies[k]->transpose_hinge_map * arma::inv(trimatu(p.D));

    p.tau_bar(span_k1_[k], span_k1_[k]) =
        arma::eye(6, 6) - p.G_fractal[k] * bodies[k]->hinge_map;

    p.P_plus(span_k2_[k], span_k2_[k]) =
        p.tau_bar(span_k1_[k], span_k1_[k]) * p.P(span_k1_[k], span_k1_[k]);

    p.J_fractal.col(k) =
        spatial_operator_dt[k] * p.J_fractal_plus.col(k) +
        p.P(span_k1_[k], span_k1_[k]) *
            coriolis_vector(bodies[k]->transpose_hinge_map,
                            p.body_velocities.col(k), p.theta_dot[k]) +
        gyroscopic_force_z(bodies[k]->inertial_matrix,
                           p.body_velocities.col(k));

    p.eta = -bodies[k]->hinge_map * p.J_fractal.col(k);

    p.frac_v[k] = arma::solve(trimatu(p.D), p.eta);

    p.J_fractal_plus.col(k + 1) = p.J_fractal.col(k) + p.G_fractal[k] * p.eta;
  }

  for (int k = n - 1; k > -1; --k) {

    p.accel_plus.col(k) = spatial_operator_dt[k + 1].t() * p.accel.col(k + 1);
    p.theta_ddot[k] = p.frac_v[k] - p.G_fractal[k].t() * p.accel_plus.col(k);
    p.accel.col(k) = p.accel_plus.col(k) +
                     bodies[k]->transpose_hinge_map * p.theta_ddot[k] +
                     coriolis_vector(bodies[k]->transpose_hinge_map,
                                     p.body_velocities.col(k), p.theta_dot[k]);
  }

  for (int k = 0; k < n; ++k) {
    p.body_forces.col(k) =
        p.P_plus(span_k2_[k], span_k2_[k]) * p.accel_plus.col(k) +
        p.J_fractal_plus.col(k + 1);
  }

  to_std_vec(dydt, p);
}

// initialize needed spans once
if (!spans_initialized) {
  span_k1_.reserve(n);
  span_k2_.reserve(n);

  for (int k = 0; k < n; ++k) {
    span_k1_.emplace_back(6 * k, 6 * (k + 1) - 1);
    span_k2_.emplace_back(6 * (k + 1), 6 * (k + 2) - 1);
  }
  spans_initialized = true;
}

// Sorward dynamic specific attributes
struct forward_parameters {
  arma::mat P_plus;
  arma::mat J_fractal_plus;
  arma::mat tau_bar;
  arma::mat P;
  arma::mat J_fractal;
  arma::mat accel;
  arma::mat accel_plus;
  arma::mat body_velocities;
  arma::mat body_forces;
  std::vector<arma::mat> G_fractal;
  std::vector<arma::mat> frac_v;
  arma::mat D;
  arma::mat eta;
  std::vector<double> dydt_out;
  std::vector<arma::vec> theta;
  std::vector<arma::vec> theta_dot;
  std::vector<arma::vec> theta_ddot;
  int hidden_index = 0;

  forward_parameters(int n_, int system_total_dof_)
      : P_plus(arma::zeros(6 * (n_ + 1), 6 * (n_ + 1))),
        J_fractal_plus(arma::zeros(6, n_ + 1)),
        tau_bar(arma::zeros(6 * n_, 6 * (n_ + 1))),
        P(arma::zeros(6 * n_, 6 * (n_ + 1))), J_fractal(arma::zeros(6, n_ + 1)),
        accel(arma::zeros(6, n_ + 1)), accel_plus(arma::zeros(6, n_ + 1)),
        body_velocities(arma::zeros(6, n_ + 1)),
        body_forces(arma::zeros(6, n_ + 1)), G_fractal(n_), frac_v(n_),
        theta(n_), theta_dot(n_), theta_ddot(n_),
        dydt_out(system_total_dof_ * 2, 0.0)

  {}
};
