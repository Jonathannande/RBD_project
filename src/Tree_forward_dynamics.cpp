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
	const std::vector<std::vector<arma::mat::fixed<6,6>>> spatial_operator_dt = find_spatial_operator_tree(p.theta);

	//now we start sweeping n times in total, first kinematics which has in part already been done through the spatial operator, but now velocities

	// the current form is pretty nasty - should consider a rewrite eventually
	for (int k = n-1; k > -1; --k) {
			p.body_velocities.col(k) = spatial_operator_dt[bodies[k]->parent_ID+1][child_index(bodies[bodies[k]->parent_ID]->children_ID,k)].t()*p.body_velocities.col(bodies[k]->parent_ID)		+bodies[k]->transpose_hinge_map*p.theta_dot[k];
		}


	for (int k = 0; k < n; ++k) {


		compute_p(k,p,spatial_operator_dt);

		p.D = bodies[k]->hinge_map*p.P(span_k1_[k],span_k1_[k])*bodies[k]->transpose_hinge_map;

		p.G_fractal[k] =p.P(span_k1_[k],span_k1_[k])*bodies[k]->transpose_hinge_map*arma::inv(trimatu(p.D));

		p.tau_bar(span_k1_[k],span_k1_[k]) = arma::eye(6,6)-p.G_fractal[k]*bodies[k]->hinge_map;

		p.P_plus(span_k2_[k],span_k2_[k]) = p.tau_bar(span_k1_[k],span_k1_[k])*p.P(span_k1_[k],span_k1_[k]);

		compute_J_fractal(k,p,spatial_operator_dt);

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




	//Assuming the list is sorted from 1 through n
	std::vector<std::vector<arma::vec6>> SystemOfBodies::find_spatial_operator_input_vector_tree(const std::vector<arma::vec>& state) const{

		std::vector<std::vector<arma::vec6>> return_vector(n);

		//uses the modified theta
		for (int i = 0; i<n; ++i) {
			size_t num_out = bodies[i]->out_hinge_tree.size();
			return_vector[i].reserve(num_out);

			arma::vec6 base = bodies[i]->hinge_map.t() * state[i] +bodies[i]->hinge_pos;
			for (int j = 0; j < num_out; ++j) {
				return_vector[i][j] = base - bodies[i]->out_hinge_tree[j];
			}
		}
		return return_vector;
	}


	std::vector<std::vector<arma::mat::fixed<6,6>>> SystemOfBodies::find_spatial_operator_tree(const std::vector<arma::vec>& state) const {


		std::vector<std::vector<arma::vec6>> rigid_body_transform_vector = find_spatial_operator_input_vector_tree(state);
		const arma::mat I = arma::eye(6,6);
		std::vector<std::vector<arma::mat::fixed<6,6>>> spatial_operator(n+2);
		spatial_operator.reserve(n+2);
		spatial_operator[0][0] = I; spatial_operator[n+1][0] = I;


		for (int i = 1; i < n+1; ++i) {
			const size_t num_out = bodies[i]->out_hinge_tree.size();
			spatial_operator[i].reserve(num_out);

			for (int j = 0; j < num_out; ++j) {
				{

					spatial_operator[i][j] = rb_transform(rigid_body_transform_vector[i-1][j].cols(0, 2),rigid_body_transform_vector[i-1][j].cols(3, 5));
				}
			}
		}
		return spatial_operator;
	}

	int SystemOfBodies::child_index(const std::vector<int>& child_ids, const int& k_index) const {
		int k = 0;
		while (child_ids[k] != k_index+1) {
			k++;
		}
		return k;
	}


	void SystemOfBodies::compute_J_fractal(const int& k, forward_parameters& p,const std::vector<std::vector<arma::mat>>& spatial_operator_dt) const{

		if (bodies[k]->children_ID.size() == 1) {
			p.J_fractal.col(k) = spatial_operator_dt[k][0]*p.J_fractal_plus.col(k)+p.P(span_k1_[k],span_k1_[k])*coriolis_vector(bodies[k]->transpose_hinge_map,p.body_velocities.col(k),p.theta_dot[k])+gyroscopic_force_z(bodies[k]->inertial_matrix,p.body_velocities.col(k));
		}
		else {
			p.J_fractal.col(k) = arma::zeros<arma::vec>(6);
			for (size_t i = 0; i < bodies[k]->children_ID.size(); ++i) {
				p.J_fractal.col(k) += spatial_operator_dt[k][i]*p.J_fractal_plus.col(k);
			}
			p.J_fractal.col(k) += p.P(span_k1_[k],span_k1_[k])*coriolis_vector(bodies[k]->transpose_hinge_map,p.body_velocities.col(k),p.theta_dot[k])+gyroscopic_force_z(bodies[k]->inertial_matrix,p.body_velocities.col(k));
		}
	}


	void SystemOfBodies::compute_p(const int& k, forward_parameters& p,const std::vector<std::vector<arma::mat>>& spatial_operator_dt) const {

		if (bodies[k]->children_ID.size() == 1) {
			p.P(span_k1_[k],span_k1_[k]) = spatial_operator_dt[k][0]*p.P_plus(span_k1_[k],span_k1_[k])*spatial_operator_dt[k][0].t()+bodies[k]->inertial_matrix;
		}
		else {
			p.P(span_k1_[k],span_k1_[k]) = arma::zeros<arma::mat>(6,6);
			for (int i = 0; i < bodies[k]->children_ID.size(); ++i) {
				p.P(span_k1_[k],span_k1_[k]) += spatial_operator_dt[k][i]*p.P_plus(span_k1_[k],span_k1_[k])*spatial_operator_dt[k][i].t();
			}
			p.P(span_k1_[k],span_k1_[k]) += bodies[k]->inertial_matrix;
		}
	}
