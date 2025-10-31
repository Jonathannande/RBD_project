//
// Created by Jonathan on 10/31/2025.
//

#import "system_of_bodies.h"
#include "plotting_utils.h"
#include <vector>
#include <boost/numeric/odeint/integrate/integrate_adaptive.hpp>
#include <boost/numeric/odeint/stepper/runge_kutta_fehlberg78.hpp>
#include <boost/numeric/odeint/integrate/integrate_const.hpp>
#include <boost/numeric/odeint/stepper/runge_kutta_dopri5.hpp>
#include <boost/numeric/odeint/stepper/controlled_runge_kutta.hpp>
#include <boost/numeric/odeint/stepper/dense_output_runge_kutta.hpp>
#include <chrono>

using namespace boost::numeric::odeint;

void SystemOfBodies::EOM__forward_tree(const std::vector<double> &y, std::vector<double>& dydt, forward_parameters &p) const {

	//method converting the state to the arma::vec format distributed over the dofs of each body
	to_arma_vec(y,p);

	p.accel(4, n) = system_gravity;  // should be reworked

    // Calling the methods
	const std::vector<arma::mat::fixed<6,6>> spatial_operator_dt = find_spatial_operator(p.theta);

	//now we start sweeping n times in total, first kinematics which has in part already been done through the spatial operator, but now velocities

	for (int k = n-1; k > -1; --k) {
		p.body_velocities.col(k) = spatial_operator_dt[k+1].t()*p.body_velocities.col(k+1)+bodies[k]->transpose_hinge_map*p.theta_dot[k];
	}


	for (int k = 0; k < n; ++k) {


		p.P(span_k1_[k],span_k1_[k]) = spatial_operator_dt[k]*p.P_plus(span_k1_[k],span_k1_[k])*spatial_operator_dt[k].t()+bodies[k]->inertial_matrix;

		p.D = bodies[k]->hinge_map*p.P(span_k1_[k],span_k1_[k])*bodies[k]->transpose_hinge_map;

		p.G_fractal[k] =p.P(span_k1_[k],span_k1_[k])*bodies[k]->transpose_hinge_map*arma::inv(trimatu(p.D));

		p.tau_bar(span_k1_[k],span_k1_[k]) = arma::eye(6,6)-p.G_fractal[k]*bodies[k]->hinge_map;

		p.P_plus(span_k2_[k],span_k2_[k]) = p.tau_bar(span_k1_[k],span_k1_[k])*p.P(span_k1_[k],span_k1_[k]);

		p.J_fractal.col(k) = spatial_operator_dt[k]*p.J_fractal_plus.col(k)+p.P(span_k1_[k],span_k1_[k])*coriolis_vector(bodies[k]->transpose_hinge_map,p.body_velocities.col(k),p.theta_dot[k])+gyroscopic_force_z(bodies[k]->inertial_matrix,p.body_velocities.col(k));

		p.eta = -bodies[k]->hinge_map*p.J_fractal.col(k);

		p.frac_v[k] = arma::solve(trimatu(p.D),p.eta);

		p.J_fractal_plus.col(k+1) = p.J_fractal.col(k)+p.G_fractal[k]*p.eta;

	}


	for (int k = n-1; k > -1; --k) {

		p.accel_plus.col(k) = spatial_operator_dt[k+1].t()*p.accel.col(k+1);
		p.theta_ddot[k] = p.frac_v[k] - p.G_fractal[k].t()*p.accel_plus.col(k);
		p.accel.col(k) = p.accel_plus.col(k)+bodies[k]->transpose_hinge_map*p.theta_ddot[k]+coriolis_vector(bodies[k]->transpose_hinge_map,p.body_velocities.col(k),p.theta_dot[k]);
	}

	for (int k = 0; k < n; ++k) {
		p.body_forces.col(k) =p.P_plus(span_k2_[k],span_k2_[k])*p.accel_plus.col(k)+p.J_fractal_plus.col(k+1);
	}

	to_std_vec(dydt,p);
}

void SystemOfBodies::solve_forward_dynamics_tree() {

	//initialize needed spans once
	if (!spans_initialized) {
		span_k1_.reserve(n);
		span_k2_.reserve(n);

		for (int k=0; k < n; ++k) {
			span_k1_.emplace_back(6*k,6*(k+1)-1);
			span_k2_.emplace_back(6*(k+1),6*(k+2)-1);
		}
		spans_initialized = true;
	}

	forward_parameters p(n,system_total_dof);
	forward_parameters_2 p_2(n,system_total_dof);


	//single vector to store data
	std::vector<double> results;
	results.reserve((system_total_dof+1)*t/dt+50);

	// body frame storage
	arma::mat store_velocities = arma::zeros(n*6,t/dt+1);
	arma::mat store_accelerations = arma::zeros(n*6,t/dt+1);
	arma::mat store_forces = arma::zeros(n*6,t/dt+1);




	// Observer lambda
	auto obs = [&](const std::vector<double>& y, double t) {
		results.push_back(t);

		/*
		for (size_t k = 0; k < n; k++) {
			store_accelerations.col(p.hidden_index).rows(k*6,(k+1)*6-1) = p.accel.col(k);
			store_velocities.col(p.hidden_index).rows(k*6,(k+1)*6-1) = p.body_velocities.col(k);
			store_forces.col(p.hidden_index).rows(k*6,(k+1)*6-1) = p.body_forces.col(k);
		}
		*/

		//state is save from y which holds positions and velocities - and then accelerations are obtained from a dummy variable as dydt is hidden in solver internal state
		results.insert(results.end(), y.begin() , y.end());
		results.insert(results.end(), p.dydt_out.begin() + system_total_dof, p.dydt_out.end());

		p.hidden_index++;
	};


	auto start = std::chrono::high_resolution_clock::now();

	if (has_dynamic_time_step == false) {

		runge_kutta_fehlberg78<std::vector<double>> stepper;
		integrate_const(
			stepper,
			[&](const std::vector<double>& y, std::vector<double>& dydt, double t) {
				system_of_equations_forward_dynamics(y, dydt,p);
			},
			system_state, t0, t, dt, obs
		);
	}
	else {


		dense_output_runge_kutta< controlled_runge_kutta< runge_kutta_dopri5< std::vector<double> > > > dense;
		integrate_adaptive(
			dense,
			[&](const std::vector<double>& y, std::vector<double>& dydt, double t) {
				system_of_equations_forward_dynamics(y, dydt,p);
			},
			system_state, t0, t, dt, obs
		);
	}



	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

	std::cout << "Time: " << duration.count() << " microseconds\n";
	// format generalized results
	ParsedData formatted_results = parseResults(results);

	// plot generalized results
	plot_thetas(formatted_results, system_total_dof);

	// plot body frame results if system is small
	if (n<5) {
		plot_data(store_accelerations,formatted_results,n,"accelerations");
		plot_data(store_velocities,formatted_results,n,"velocities");
		plot_data(store_forces,formatted_results,n,"forces");
	}


	}