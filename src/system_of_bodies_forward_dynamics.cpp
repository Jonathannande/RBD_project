//
// Created by Jonathan on 10/7/2025.
//


#include "system_of_bodies.h"
#include "math_utils.h"
#include "plotting_utils.h"
#include <array>
#include <vector>
#include <boost/numeric/odeint/stepper/runge_kutta_cash_karp54.hpp>
#include <boost/numeric/odeint/integrate/integrate_const.hpp>

using namespace boost::numeric::odeint;

void SystemOfBodies::system_of_equations_forward_dynamics(const std::vector<double> &y, std::vector<double> &dydt, double t,arma::mat P_plus, arma::mat J_fractal_plus, arma::mat tau_bar, arma::mat P, arma::mat J_fractal,
                                                          arma::mat accel, arma::mat accel_plus, arma::mat body_velocities, std::vector<arma::mat> G_fractal,std::vector<arma::mat> frac_v,arma::mat eta,arma::mat D) {



		//takes current state of the solver and creates the generalized coordinates
		std::vector<double> theta_standard_form(y.begin() , y.end()-y.size()/2);
		std::vector<double> theta_dot_standard_form(y.begin()+y.size()/2, y.end());
		std::vector<arma::vec> theta = to_arma_vec(theta_standard_form);
		std::vector<arma::vec> theta_dot = to_arma_vec(theta_dot_standard_form);


		accel(4, n) = system_gravity;  // Equivalent to MATLAB's `accel(5,n+1) = -g;`



	    //we really need to do something about this
	    arma::mat transform_matrix = find_spatial_operator_input_vector(theta);
	    arma::mat spatial_operator_dt = find_spatial_operator(transform_matrix, n);

		//now we start sweeping n times in total, first kinematics which has in part already been done through the spatial operator, but now velocities

		for (int k = n-1; k >= 0; --k) {
			body_velocities.col(k) = spatial_operator_dt(arma::span(6*(k+1),6*(k+2)-1),arma::span(6*(k+1),6*(k+2)-1)).t()*body_velocities.col(k+1)+bodies[k]->transpose_hinge_map*theta_dot[k];
		}


		//these are arrays storing the kth interval index, just so that you dont have to do the annoying indexing all the time
		std::array<int, 2> k_idx1;
		std::array<int, 2> k_idx2;

		for (int k = 0; k <= n-1; ++k) {

			k_idx1 = {6*k,6*(k+1)-1};
			k_idx2 = {6*(k+1),6*(k+2)-1};

			P(arma::span(k_idx1[0],k_idx1[1]),arma::span(k_idx1[0],k_idx1[1])) = spatial_operator_dt(arma::span(k_idx1[0],k_idx1[1]),arma::span(k_idx1[0],k_idx1[1]))*P_plus(arma::span(k_idx1[0],k_idx1[1]),arma::span(k_idx1[0],k_idx1[1]))*spatial_operator_dt(arma::span(k_idx1[0],k_idx1[1]),arma::span(k_idx1[0],k_idx1[1])).t()+bodies[k]->inertial_matrix;

			D = bodies[k]->hinge_map*P(arma::span(k_idx1[0],k_idx1[1]),arma::span(k_idx1[0],k_idx1[1]))*bodies[k]->transpose_hinge_map;

			G_fractal[k] = P(arma::span(k_idx1[0], k_idx1[1]),arma::span(k_idx1[0], k_idx1[1]))*bodies[k]->transpose_hinge_map*arma::inv(arma::mat{D});

			tau_bar(arma::span(k_idx1[0],k_idx1[1]),arma::span(k_idx1[0],k_idx1[1])) = arma::eye(6,6)-G_fractal[k]*bodies[k]->hinge_map;

			P_plus(arma::span(k_idx2[0],k_idx2[1]),arma::span(k_idx2[0],k_idx2[1])) = tau_bar(arma::span(k_idx1[0],k_idx1[1]),arma::span(k_idx1[0],k_idx1[1]))*P(arma::span(k_idx1[0],k_idx1[1]),arma::span(k_idx1[0],k_idx1[1]));

			J_fractal.col(k) = spatial_operator_dt(arma::span(k_idx1[0],k_idx1[1]),arma::span(k_idx1[0],k_idx1[1]))*J_fractal_plus.col(k)+P(arma::span(k_idx1[0],k_idx1[1]),arma::span(k_idx1[0],k_idx1[1]))*coriolis_vector(bodies[k]->transpose_hinge_map,body_velocities.col(k),theta_dot[k])+gyroscopic_force_z(bodies[k]->inertial_matrix,body_velocities.col(k));

			eta = -bodies[k]->hinge_map*J_fractal.col(k);

			frac_v[k] = arma::solve(D,eta);

			J_fractal_plus.col(k+1) = J_fractal.col(k)+G_fractal[k]*eta;


		}

		std::vector<arma::vec> theta_ddot(n);
		for (int k = n-1; k >= 0; --k) {
			k_idx1 = {6*k,6*(k+1)-1};
			k_idx2 = {6*(k+1),6*(k+2)-1};
			accel_plus.col(k) = spatial_operator_dt(arma::span(k_idx2[0],k_idx2[1]),arma::span(k_idx2[0],k_idx2[1])).t()*accel.col(k+1);
			theta_ddot[k] = frac_v[k] - G_fractal[k].t()*accel_plus.col(k);
			accel.col(k) = accel_plus.col(k)+bodies[k]->transpose_hinge_map*theta_ddot[k]+coriolis_vector(bodies[k]->transpose_hinge_map,body_velocities.col(k),theta_dot[k]);
		}


		    // dv/dt = a(t)
		std::vector<double> theta_as_std_vec = to_std_vec(theta_dot);
		std::vector<double> theta_ddot_as_std_vec = to_std_vec(theta_ddot);

	    for(size_t i = 0; i < system_total_dof; ++i){
	        dydt[i] = theta_as_std_vec[i];
	    }

	    // dx/dt = v
	    for(size_t i = 0; i < system_total_dof; ++i){
	        dydt[system_total_dof + i] = theta_ddot_as_std_vec[i];

	    }

	    //formatting
		for (int k = 0; k < n; ++k)
		{

			//store_velocities.col(col_index).rows(k*(6),(k+1)*(6)-1) = body_velocities.col(k);
			//store_accelerations.col(col_index).rows(k*(6),(k+1)*(6)-1) = accel.col(k);
			//store_forces.col(col_index).rows(k*(6),(k+1)*(6)-1) = body_forces.col(k+1);
			//store_generalized_forces.col(i) = generalized_forces.t();
		}


	}


	void SystemOfBodies::solve_forward_dynamics() {
		//storage

		/* This a should hold the results of the body relative states for later plotting and insspection of results
		arma::mat store_velocities = arma::zeros(n*6,t/dt);
		arma::mat store_accelerations = arma::zeros(n*6,t/dt);
		arma::mat store_forces = arma::zeros(n*6,t/dt);
		arma::mat store_generalized_forces = arma::zeros(n,t/dt);
		*/
		arma::mat P_plus = arma::zeros(6*(n+1), 6*(n+1));
		arma::mat J_fractal_plus = arma::zeros(6, n+1);
		arma::mat tau_bar = arma::zeros(6*n, 6*(n+1));
		arma::mat P = arma::zeros(6*n, 6*(n+1));
		arma::mat J_fractal = arma::zeros(6, n+1);
		arma::mat accel = arma::zeros(6, n+1);
		arma::mat accel_plus = arma::zeros(6, n+1);
		arma::mat body_velocities = arma::zeros(6, n+1);
		std::vector<arma::mat> G_fractal(n);
		std::vector<arma::mat> frac_v(n);
		arma::mat D;
		arma::mat eta;

	    // Single vector to store data
	    std::vector<double> results;

	    //This is a dummy parameter which would count the iteration, for have a storage index for the body relative states
	    //int col_index = 0;

	    // Observer lambda - OBS does't work for the current state model
		auto obs = [&](const std::vector<double>& y, double t) {
		    // Create a temporary vector to hold the derivative.
		    std::vector<double> dydt(y.size(), 0.0);

		    // Compute the derivative at the current state and time.
			system_of_equations_forward_dynamics(y, dydt, t,P_plus, J_fractal_plus, tau_bar, P, J_fractal,
			accel, accel_plus, body_velocities, G_fractal, frac_v,eta,D);

		    results.push_back(t);

		    //has to separate for loops because of the pushback
		    for (size_t i = system_total_dof; i < 2*system_total_dof; i++) {
		         results.push_back(dydt[i]);
		    }
			for (size_t i = system_total_dof; i < 2*system_total_dof; i++) {
				results.push_back(y[i]);
			}

		    for (size_t i = 0; i < system_total_dof; i++) {
		         results.push_back(y[i]);
		    }
		    //col_index++;
		};

		//starting time

	    runge_kutta_cash_karp54<std::vector<double>> stepper;
	    integrate_const(
		    stepper,
		    [&](const std::vector<double>& y, std::vector<double>& dydt, double t) {
		    	system_of_equations_forward_dynamics(y, dydt, t, P_plus, J_fractal_plus, tau_bar, P, J_fractal,
			accel, accel_plus, body_velocities, G_fractal, frac_v,eta,D);
		    },
		    system_state, t0, t, dt, obs
		);

	    ParsedData formatted_results = parseResults(results);
	    std::cout << "done1" << std::endl;
	    plot_thetas(formatted_results, system_total_dof);
	    std::cout << "done2" << std::endl;
	    //compute_body_forward_dynamics(formatted_results);

	}