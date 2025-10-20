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
	ParsedData SystemOfBodies::parseResults(std::vector<double>& results)
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


	/*
	std::vector<arma::vec> SystemOfBodies::to_arma_vec(std::vector<double> DoFs) {

		std::vector<arma::vec> return_vector;

		int start_idx{0};

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

*/

	std::vector<arma::vec> SystemOfBodies::to_arma_vec(const std::vector<double>& DoFs) {
		std::vector<arma::vec> return_vector;
		return_vector.reserve(n);

		int start_idx = 0;
		for (int i = 0; i < n; ++i) {
			arma::vec slice = arma::conv_to<arma::vec>::from(
				std::vector<double>(DoFs.begin() + start_idx,
								  DoFs.begin() + start_idx + system_dofs_distribution[i])
			);
			return_vector.push_back(std::move(slice));
			start_idx += system_dofs_distribution[i];
		}
		return return_vector;
	}

	std::vector<double> SystemOfBodies::to_std_vec(const std::vector<arma::vec>& DoFs) {
		std::vector<double> return_vector;

		for (int i = 0; i < n; ++i) {
			std::vector<double> temp = arma::conv_to<std::vector<double>>::from(DoFs[i]);
			return_vector.insert(return_vector.end(), temp.begin(), temp.end());
		}
		return return_vector;
	}



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

