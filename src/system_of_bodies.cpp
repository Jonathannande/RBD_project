//
// Created by Jonathan on 10/1/2025.
//

#include "system_of_bodies.h"

#include <boost/numeric/odeint.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/quadrature/gauss.hpp>
#include <armadillo>
#include <iostream>
#include <cmath>
#include <vector>
#include "../matplotlibcpp.h"
#include <memory>
#include "data_types.h"
#include "bodies.h"


namespace plt = matplotlibcpp;
using namespace boost::math::float_constants;
using namespace boost::numeric::odeint;



//we need to use some pointer for this thing


	//modified parsed result to fit arbitrary hinge_map
	ParsedData SystemOfBodies::parseResults(const std::vector<double>& results)
	{

	    const size_t row_size = 1 + 3*system_total_dof;
	    std::cout << row_size << std::endl;
	    const size_t steps = results.size() / row_size;
	    std::cout << "results.size(): " << results.size() << "steps: " <<steps << std::endl;

	    ParsedData data;
	    data.times.resize(steps);
	    data.accel.assign(system_total_dof, std::vector<double>(steps));
	    data.vel.assign(system_total_dof, std::vector<double>(steps));
	    data.pos.assign(system_total_dof, std::vector<double>(steps));

	    for (size_t s = 0; s < steps; s++)
	    {
	        size_t offset = s * (row_size); //seemingly has some kind of a bug +2 for the 2 dof single pendulum thing
	        data.times[s] = results[offset];
	        //std::cout << data.times[s] << std::endl;
	        // accelerations
	        for (size_t i = 0; i < system_total_dof; i++) {
	            data.accel[i][s] = results[offset + 1 + i];
	            data.vel[i][s] = results[offset + 1 + system_total_dof + i];
	            data.pos[i][s] = results[offset + 1 + 2*system_total_dof + i];
	        }
	    }

	    return data;
	}

	void SystemOfBodies::update_system_state() {
		std::cout << std::endl << "new body added" << std::endl;
		std::vector<double> state(2*system_total_dof,0.0);
		int offset = 0;
		for (int i = 0; i<n; i++) {
			for (int j = 0; j<system_dofs_distribution[i]; ++j) {

				state[offset+j] = bodies[i]->state(j);
				std::cout << offset+j << " is now set to " << bodies[i]->state(j) << " for "<< system_dofs_distribution[i] << std::endl;
				state[system_total_dof+offset+j] = bodies[i]->state(j+system_dofs_distribution[i]);
				std::cout << system_total_dof+offset+j << " is now set to " << bodies[i]->state(j+system_dofs_distribution[i]) << std::endl;

			}
			offset += system_dofs_distribution[i];
		}
		system_state = state;


	}



	//this function receives the generalized positions at each time step, converts the dofs of each body and uses it to compute the input vector needed for the spatial operator.
	arma::mat SystemOfBodies::find_spatial_operator_input_vector(std::vector<arma::vec>& DoFs) {
		arma::mat return_vector = arma::zeros(n,6);

		//uses the modified theta
		for (int i = 0; i<n; ++i) {
			return_vector.row(i) = (bodies[i]->hinge_map.t()*DoFs[i] + (bodies[i]->position_vec_hinge_big-bodies[i]->outboard_position_vec_hinge_big)).t();
		}
		return return_vector;
	}

	std::vector<arma::vec> SystemOfBodies::to_arma_vec(std::vector<double> DoFs) {
		//std::cout << "yo";
		std::vector<arma::vec> return_vector;

		int start_idx = 0;
		//matbe shouldn't set it to false i dont know
		for (int i = 0; i<n; ++i) {

			return_vector.push_back(arma::vec(DoFs.data() + start_idx, system_dofs_distribution[i], true));
			start_idx += system_dofs_distribution[i];
		}
		return return_vector;
	}

	std::vector<double> SystemOfBodies::to_std_vec(std::vector<arma::vec>& DoFs) {
		std::vector<double> return_vector;

		for (int i = 0; i < n; ++i)
		{
			for (int j = 0; j < system_dofs_distribution[i]; ++j)
			{
				return_vector.push_back(DoFs[i](j));
			}
		}
		return return_vector;
	}

	//should only be necessary in inverse kinematics
	arma::mat SystemOfBodies::find_gravity_matrix_pseudo_acceleration() {
		arma::mat zero_config = arma::zeros(6,n);
		return zero_config;
	}
	/*
	void compute_body_forward_dynamics(ParsedData parsed_thetas) {

		//temporary placeholders for computation
		arma::mat body_velocities = arma::zeros(6,n+1);
		arma::mat body_accelerations = arma::zeros(6,n+1);
		arma::mat body_forces = arma::zeros(6,n+1);
		arma::mat generalized_forces = arma::zeros(1,n);

		//storage
		arma::mat store_velocities = arma::zeros(n*6,t/dt);
		arma::mat store_accelerations = arma::zeros(n*6,t/dt);
		arma::mat store_forces = arma::zeros(n*6,t/dt);
		arma::mat store_generalized_forces = arma::zeros(n,t/dt);

		//matrix initialization

	    arma::mat P_plus = arma::zeros(6*(n+1), 6*(n+1));
	    arma::mat J_fractal_plus = arma::zeros(6, n+1);
	    arma::mat tau_bar = arma::zeros(6*n, 6*(n+1));
	    arma::mat P = arma::zeros(6*n, 6*(n+1));
	    arma::mat D = arma::zeros(n, n);
	    arma::mat G_fractal = arma::zeros(6, n+1);
	    arma::mat J_fractal = arma::zeros(6, n+1);
	    arma::mat eta = arma::zeros(1, n+1);
	    arma::mat frac_v = arma::zeros(1, n);
	    arma::mat accel = arma::zeros(6, n+1);
	    accel(4, n) = system_gravity;  // Equivalent to MATLAB's `accel(5,n+1) = -g;`
	    arma::mat accel_plus = arma::zeros(6, n+1);

	    arma::mat inertia_matrix;

	    std::array<int, 2> k_idx1;
	    std::array<int, 2> k_idx2;

		for (int i = 0; i < t/dt; ++i)
		{
			//this should be implemented as a system property, current implementation is very much NOT general
			arma::mat transform_matix = {{0,0,parsed_thetas.pos[0][i],0,-l,0},{0,0,parsed_thetas.pos[1][i],0,-l,0},{0,0,parsed_thetas.pos[2][i],0,-l,0}};

			//this is implmented well for general system though it might not be computationally efficient
			arma::mat spatial_operator_dt = find_spatial_operator(transform_matix,n);

			//should be implemented in general and ideally as acting on the force instead further testing is necessary for that
			arma::mat gravity_vector = {{0,0,0},{0,0,0},{0,0,0},{0,0,g*sin(parsed_thetas.pos[2][i])},{0,0,g*cos(parsed_thetas.pos[2][i])},{0,0,0}};


			// scatter sweep
			for (int k = n-1; k >= 0; --k)
			{
				//might not be computationally efficient though it has general implementation
				body_velocities.col(k) = spatial_operator_dt(arma::span(6*(k+1),6*(k+2)-1),arma::span(6*(k+1),6*(k+2)-1)).t()*body_velocities.col(k+1)+hinge_map_1_transpose*parsed_thetas.vel[k][i];
			}

			for (int k = 0; k <= n-1; ++k) {

				k_idx1 = {6*(k),6*(k+1)-1};
				k_idx2 = {6*(k+1),6*(k+2)-1};

				inertia_matrix = bodies[k]->inertia_matrix();
				P(arma::span(k_idx1[0],k_idx1[1]),arma::span(k_idx1[0],k_idx1[1])) = spatial_operator_dt(arma::span(k_idx1[0],k_idx1[1]),arma::span(k_idx1[0],k_idx1[1]))*P_plus(arma::span(k_idx1[0],k_idx1[1]),arma::span(k_idx1[0],k_idx1[1]))*spatial_operator_dt(arma::span(k_idx1[0],k_idx1[1]),arma::span(k_idx1[0],k_idx1[1])).t()+inertia_matrix;
				D(k,k) = arma::as_scalar(hinge_map_1*P(arma::span(k_idx1[0],k_idx1[1]),arma::span(k_idx1[0],k_idx1[1]))*hinge_map_1_transpose);
				G_fractal.col(k) = P(arma::span(k_idx1[0], k_idx1[1]),arma::span(k_idx1[0], k_idx1[1]))*hinge_map_1_transpose*arma::inv(arma::mat{D(k,k)});
				tau_bar(arma::span(k_idx1[0],k_idx1[1]),arma::span(k_idx1[0],k_idx1[1])) = arma::eye(6,6)-G_fractal.col(k)*hinge_map_1;
				P_plus(arma::span(k_idx2[0],k_idx2[1]),arma::span(k_idx2[0],k_idx2[1])) = tau_bar(arma::span(k_idx1[0],k_idx1[1]),arma::span(k_idx1[0],k_idx1[1]))*P(arma::span(k_idx1[0],k_idx1[1]),arma::span(k_idx1[0],k_idx1[1]));
				J_fractal.col(k) = spatial_operator_dt(arma::span(k_idx1[0],k_idx1[1]),arma::span(k_idx1[0],k_idx1[1]))*J_fractal_plus.col(k)+P(arma::span(k_idx1[0],k_idx1[1]),arma::span(k_idx1[0],k_idx1[1]))*coriolis_vector(hinge_map_1_transpose,body_velocities.col(k),parsed_thetas.vel[k][i])+gyroscopic_force_z(inertia_matrix,body_velocities.col(k));
				eta.col(k) = -hinge_map_1*J_fractal.col(k);
				frac_v(k) = arma::as_scalar(arma::solve(D(arma::span(k, k), arma::span(k, k)),eta.col(k)));
				J_fractal_plus.col(k+1) = J_fractal.col(k)+G_fractal.col(k)*eta.col(k);

			}

			for (int k = n-1; k >= 0; --k) {
				k_idx1 = {6*(k),6*(k+1)-1};
				k_idx2 = {6*(k+1),6*(k+2)-1};
				accel_plus.col(k) = spatial_operator_dt(arma::span(k_idx2[0],k_idx2[1]),arma::span(k_idx2[0],k_idx2[1])).t()*accel.col(k+1);
				accel.col(k) = accel_plus.col(k)+hinge_map_1_transpose*parsed_thetas.accel[k][i]+coriolis_vector(hinge_map_1_transpose,body_velocities.col(k),parsed_thetas.vel[k][i]);
			}

			//gather sweep
			for (int k = 0; k < n; ++k)
			{
				k_idx1 = {6*(k),6*(k+1)-1};
				k_idx2 = {6*(k+1),6*(k+2)-1};
				//is not efficient, you only need to call inertia once per body, inertia matrix should be a per body computation in case boddies are different,
				body_forces.col(k+1) = P_plus(arma::span(k_idx2[0],k_idx2[1]),arma::span(k_idx2[0],k_idx2[1]))*accel_plus.col(k)+J_fractal_plus.col(k+1);
			}



			//formatting
			for (int k = 0; k < n; ++k)
			{
				//std::cout << "hello";
				store_velocities.col(i).rows(k*(6),(k+1)*(6)-1) = body_velocities.col(k);
				store_accelerations.col(i).rows(k*(6),(k+1)*(6)-1) = accel.col(k);
				store_forces.col(i).rows(k*(6),(k+1)*(6)-1) = body_forces.col(k+1);
				//store_generalized_forces.col(i) = generalized_forces.t();
			}


		}

		//plot_velocities(store_velocities, parsed_thetas);
		//plot_accelerations(store_accelerations,parsed_thetas);
		//plot_forces(store_forces,parsed_thetas);
		//plot_generalized_forces(store_generalized_forces,parsed_thetas);
		//plot_thetas(parsed_thetas);

	}
	*/
	void SystemOfBodies::test_method() {
		//takes current state of the solver and creates the generalized coordinates

	    std::vector<double> theta_standard_form(system_state.begin() , system_state.end()-system_state.size()/2);

	    std::vector<double> theta_dot_standard_form(system_state.begin()+system_state.size()/2, system_state.end());

	    std::vector<arma::vec> theta = to_arma_vec(theta_standard_form);

	    std::vector<arma::vec> theta_dot = to_arma_vec(theta_dot_standard_form);
	    for (int i = 0; i < theta.size(); ++i)

	    {
	    	//std::cout << "The postional state of theta " << i << " has the value of " << theta[i]<< std::endl;
	    	//std::cout << "The postional state of theta_dot " << i << " has the value of " << theta_dot[i]<< std::endl;
	    }

	}

	void SystemOfBodies::solve_hybrid_dynamics() {

	}

	void SystemOfBodies::get_states_forward_dynamics() {

	}

	//method that adds body to system.
	void SystemOfBodies::create_body(std::unique_ptr<Body> any_body) {
		// Move the unique_ptr into the container and get a pointer to the inserted object
		auto it = bodies.insert(bodies.begin(), std::move(any_body)); // Move ownership
		Body* inserted_body = it->get(); // Retrieve the raw pointer

		// Now use inserted_body to access methods and properties
		++n;

		system_dofs_distribution.insert(system_dofs_distribution.begin(),inserted_body->hinge_map.n_rows);

		system_total_dof += inserted_body->hinge_map.n_rows;
/*

		for (int j = 0; j < bodies.size(); ++j){
			idx1 = {6*j,6*(j+1)-1};
			idx2 = {6*(k+1),6*(k+2)-1};
		}
		*/
    update_system_state();

	}

