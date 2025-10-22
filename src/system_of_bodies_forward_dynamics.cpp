//
// Created by Jonathan on 10/7/2025.
//


#include "system_of_bodies.h"
#include "math_utils.h"
#include "plotting_utils.h"
#include <array>
#include <vector>
#include <boost/numeric/odeint/stepper/runge_kutta_fehlberg78.hpp>
#include <boost/numeric/odeint/integrate/integrate_const.hpp>


using namespace boost::numeric::odeint;

void SystemOfBodies::system_of_equations_forward_dynamics(const std::vector<double> &y, std::vector<double> &dydt, double t,arma::mat P_plus, arma::mat J_fractal_plus, arma::mat tau_bar, arma::mat P, arma::mat J_fractal,
                                                          arma::mat &accel, arma::mat accel_plus, arma::mat &body_velocities, std::vector<arma::mat> G_fractal,std::vector<arma::mat> frac_v,arma::mat eta,arma::mat D,arma::mat &body_forces,std::vector<double>&dydt_out) {

	//indexing for the for loops, k_idx's are used to define spans, the spans are then saved for repeated use
	std::array<int, 2> k_idx1, k_idx2;
	std::vector<arma::span> spans_k1(n), spans_k2(n);


	for (int k = 0; k < n; ++k) {
		k_idx1 = {6*k,6*(k+1)-1};
		k_idx2 = {6*(k+1),6*(k+2)-1};
		spans_k1[k] = arma::span(k_idx1[0], k_idx1[1]);
		spans_k2[k] = arma::span(k_idx2[0], k_idx2[1]);
	}


	//takes current state of the solver and creates the generalized coordinates
	std::vector<double> theta_standard_form(y.begin() , y.end()-system_total_dof);
	std::vector<double> theta_dot_standard_form(y.begin()+system_total_dof, y.end());

	std::vector<arma::vec> theta = to_arma_vec(theta_standard_form);
	std::vector<arma::vec> theta_dot = to_arma_vec(theta_dot_standard_form);
	std::vector<arma::vec> theta_ddot(n);

	accel(4, n) = system_gravity;  // Equivalent to MATLAB's `accel(5,n+1) = -g;`



    //we really need to do something about this
    arma::mat transform_matrix = find_spatial_operator_input_vector(theta);
    arma::mat spatial_operator_dt = find_spatial_operator(transform_matrix, n);

	//now we start sweeping n times in total, first kinematics which has in part already been done through the spatial operator, but now velocities

	for (int k = n-1; k > -1; --k) {
		body_velocities.col(k) = spatial_operator_dt(spans_k2[k],spans_k2[k]).t()*body_velocities.col(k+1)+bodies[k]->transpose_hinge_map*theta_dot[k];
	}




	for (int k = 0; k < n; ++k) {

		P(spans_k1[k],spans_k1[k]) = spatial_operator_dt(spans_k1[k],spans_k1[k])*P_plus(spans_k1[k],spans_k1[k])*spatial_operator_dt(spans_k1[k],spans_k1[k]).t()+bodies[k]->inertial_matrix;

		D = bodies[k]->hinge_map*P(spans_k1[k],spans_k1[k])*bodies[k]->transpose_hinge_map;

		G_fractal[k] = P(spans_k1[k],spans_k1[k])*bodies[k]->transpose_hinge_map*arma::inv(arma::mat{D});

		tau_bar(spans_k1[k],spans_k1[k]) = arma::eye(6,6)-G_fractal[k]*bodies[k]->hinge_map;

		P_plus(spans_k2[k],spans_k2[k]) = tau_bar(spans_k1[k],spans_k1[k])*P(spans_k1[k],spans_k1[k]);

		J_fractal.col(k) = spatial_operator_dt(spans_k1[k],spans_k1[k])*J_fractal_plus.col(k)+P(spans_k1[k],spans_k1[k])*coriolis_vector(bodies[k]->transpose_hinge_map,body_velocities.col(k),theta_dot[k])+gyroscopic_force_z(bodies[k]->inertial_matrix,body_velocities.col(k));

		eta = -bodies[k]->hinge_map*J_fractal.col(k);

		frac_v[k] = arma::solve(trimatu(D),eta);

		J_fractal_plus.col(k+1) = J_fractal.col(k)+G_fractal[k]*eta;
	}


	for (int k = n-1; k > -1; --k) {

		accel_plus.col(k) = spatial_operator_dt(spans_k2[k],spans_k2[k]).t()*accel.col(k+1);
		theta_ddot[k] = frac_v[k] - G_fractal[k].t()*accel_plus.col(k);
		accel.col(k) = accel_plus.col(k)+bodies[k]->transpose_hinge_map*theta_ddot[k]+coriolis_vector(bodies[k]->transpose_hinge_map,body_velocities.col(k),theta_dot[k]);
	}

	for (int k = 0; k < n; ++k) {
		k_idx2 = {6*(k+1),6*(k+2)-1};
		body_forces.col(k) =P_plus(spans_k2[k],spans_k2[k])*accel_plus.col(k)+J_fractal_plus.col(k+1);
	}

	std::vector<double> theta_dot_as_std_vec = to_std_vec(theta_dot);
	std::vector<double> theta_ddot_as_std_vec = to_std_vec(theta_ddot);

    for(size_t i = 0; i < system_total_dof; ++i){
        dydt[i] = theta_dot_as_std_vec[i];
    	dydt[system_total_dof + i] = theta_ddot_as_std_vec[i];
    }

	dydt_out =  dydt;


}


	void SystemOfBodies::solve_forward_dynamics() {

	// body frame storage
	arma::mat store_velocities = arma::zeros(n*6,t/dt+1);
	arma::mat store_accelerations = arma::zeros(n*6,t/dt+1);
	arma::mat store_forces = arma::zeros(n*6,t/dt+1);

	// parameters used in the forward comutation
	arma::mat P_plus = arma::zeros(6*(n+1), 6*(n+1));
	arma::mat J_fractal_plus = arma::zeros(6, n+1);
	arma::mat tau_bar = arma::zeros(6*n, 6*(n+1));
	arma::mat P = arma::zeros(6*n, 6*(n+1));
	arma::mat J_fractal = arma::zeros(6, n+1);
	arma::mat accel = arma::zeros(6, n+1);
	arma::mat accel_plus = arma::zeros(6, n+1);
	arma::mat body_velocities = arma::zeros(6, n+1);
	arma::mat body_forces = arma::zeros(6, n+1);
	std::vector<arma::mat> G_fractal(n);
	std::vector<arma::mat> frac_v(n);
	arma::mat D;
	arma::mat eta;
	std::vector<double> dydt_out(system_total_dof*2, 0.0);

	//single vector to store data
	std::vector<double> results;
	results.reserve((system_total_dof+1)*t/dt+50);






	// Observer lambda
	auto obs = [&](const std::vector<double>& y, double t) {


		results.push_back(t);

		for (size_t k = 0; k < n; k++) {
			store_accelerations.col(t/dt).rows(k*6,(k+1)*6-1) = accel.col(k);
			store_velocities.col(t/dt).rows(k*6,(k+1)*6-1) = body_velocities.col(k);
			store_forces.col(t/dt).rows(k*6,(k+1)*6-1) = body_forces.col(k);
		}

		//state is save from y which holds positions and velocities - and then accelerations are obtained from a dummy variable as dydt is hidden in solver internal state
		results.insert(results.end(), y.begin() , y.end());
		results.insert(results.end(), dydt_out.begin() + system_total_dof, dydt_out.end());



	};

	if (has_dynamic_time_step == false) {
		runge_kutta_fehlberg78<std::vector<double>> stepper;
		integrate_const(
			stepper,
			[&](const std::vector<double>& y, std::vector<double>& dydt, double t) {
				system_of_equations_forward_dynamics(y, dydt, t, P_plus, J_fractal_plus, tau_bar, P, J_fractal,
			accel, accel_plus, body_velocities, G_fractal, frac_v,eta,D,body_forces, dydt_out);
			},
			system_state, t0, t, dt, obs
		);
	}
	else {

	}
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