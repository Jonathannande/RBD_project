//
// Created by Jonathan on 10/7/2025.
//

#import "system_of_bodies.h"
#import "bodies.h"
#import "muparser.h"
#import "math_utils.h"
#import "plotting_utils.h"

	ParsedData SystemOfBodies::inverse_run_funcs() {

    ParsedData data;

	double var_t = t0;
	mu::Parser p;
	p.DefineVar("t", &var_t);
	p.DefineConst("pi", M_PI);


    // Evaluate at each time step
    for (; var_t < t; var_t += dt) {
        data.times.push_back(var_t);

        // Evaluate all bodies at this time
        for (int body_idx = 0; body_idx < n; ++body_idx) {
        	Body* body = bodies[body_idx].get();
            int n_dofs = system_dofs_distribution[body_idx];
        	std::vector<std::string>& funcs = body->inverse_dynamics_funcs;

            std::vector<double> pos_vals(n_dofs);
            std::vector<double> vel_vals(n_dofs);
            std::vector<double> acc_vals(n_dofs);

            for (int dof = 0; dof < n_dofs; ++dof) {
            	p.SetExpr(funcs[dof*3]);
            	pos_vals[dof] = p.Eval();

            	p.SetExpr(funcs[dof*3+1]);
            	vel_vals[dof] = p.Eval();

            	p.SetExpr(funcs[dof*3+2]);
                acc_vals[dof] = p.Eval();
            }

            // Store in ParsedData (assuming it has vectors for storage)
            data.pos.push_back(pos_vals);
            data.vel.push_back(vel_vals);
            data.accel.push_back(acc_vals);
        }
    }
    return data;
}

std::vector<double> SystemOfBodies::inverse_run_funcs_2() {

		std::vector<double> data;

		double var_t = t0;
		mu::Parser p;
		p.DefineVar("t", &var_t);
		p.DefineConst("pi", M_PI);


		// Evaluate at each time step
		for (; var_t < t; var_t += dt) {


			// Evaluate all bodies at this time
			data.push_back(var_t);
			for (int k = 0; k < n; ++k) {
				std::vector<std::string>& funcs = bodies[k]->inverse_dynamics_funcs;
				p.SetExpr(funcs[2]);
				double acc_vals = p.Eval();
				data.push_back(acc_vals);
			}

			for (int k = 0; k < n; ++k) {
				std::vector<std::string>& funcs = bodies[k]->inverse_dynamics_funcs;
				p.SetExpr(funcs[1]);
				double vel_vals = p.Eval();
				data.push_back(vel_vals);
			}

			for (int k = 0; k < n; ++k) {
				std::vector<std::string>& funcs = bodies[k]->inverse_dynamics_funcs;
				p.SetExpr(funcs[0]);
				double pos_vals = p.Eval();
				data.push_back(pos_vals);
			}




		}
		return data;
	}






	void SystemOfBodies::solve_inverse_dynamics() {
		ParsedData thetas = inverse_run_funcs();
		std::vector<double> to_be_parsed = inverse_run_funcs_2();


	//not very happy with this implementation but will stay for now


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



	for (int i = 0; i < t/dt; ++i)
	{
		//this should be implemented as a system property, current implementation is very much NOT general


		std::vector<arma::vec> theta_position = to_arma_vec(thetas.pos[3*i]);
		std::vector<arma::vec> theta_velocity = to_arma_vec(thetas.vel[3*i]);
		std::vector<arma::vec> theta_acceleration = to_arma_vec(thetas.accel[3*i]);

		arma::mat transform_matrix = find_spatial_operator_input_vector(theta_position);
		arma::mat spatial_operator_dt = find_spatial_operator(transform_matrix, n);




		//should be implemented in general and ideally as acting on the force instead further testing is necessary for that


		arma::mat gravity_vector = arma::zeros(6, n);
		gravity_vector(3, n-1) = system_gravity * sin(thetas.pos[i][n-1]);
		gravity_vector(4, n-1) = system_gravity * cos(thetas.pos[i][n-1]);

		// scatter sweep
		for (int k = n-1; k >= 0; --k)
		{
			//std::vector<arma::vec> theta_position = to_arma_vec(thetas.pos[k]);


			//might not be computationally efficient though it has general implementation
			body_velocities.col(k) = spatial_operator_dt(arma::span(6*(k+1),6*(k+2)-1),arma::span(6*(k+1),6*(k+2)-1)).t()*body_velocities.col(k+1)+bodies[k]->transpose_hinge_map*theta_velocity[k];

			//might not be computationally efficient though it has general implementation - gravity should be removed eventually
			body_accelerations.col(k) = spatial_operator_dt(arma::span(6*(k+1),6*(k+2)-1),arma::span(6*(k+1),6*(k+2)-1)).t()*body_accelerations.col(k+1)+bodies[k]->transpose_hinge_map*theta_acceleration[k]+coriolis_vector(bodies[k]->transpose_hinge_map,body_velocities.col(k),theta_velocity[k])+gravity_vector.col(k);
		}

		//gather sweep
		for (int k = 0; k < n; ++k)
		{
			//is not efficient, you only need to call inertia once per body, inertia matrix should be a per body computation in case boddies are different,
			body_forces.col(k+1) = spatial_operator_dt(arma::span(6*(k),6*(k+1)-1),arma::span(6*(k),6*(k+1)-1))*body_forces.col(k)+bodies[k]->inertial_matrix*(body_accelerations.col(k))+gyroscopic_force_z(bodies[k]->inertial_matrix,body_velocities.col(k));
			//not much to add here
			generalized_forces.col(k) = bodies[k]->hinge_map*body_forces.col(k+1);
			//std::cout << k << std::endl;

		}
		//std::cout << body_forces << std::endl;
		for (int k = 0; k < n; ++k)
		{
			//std::cout << "hello";
			store_velocities.col(i).rows(k*6,(k+1)*6-1) = body_velocities.col(k);
			store_accelerations.col(i).rows(k*6,(k+1)*6-1) = body_accelerations.col(k);
			store_forces.col(i).rows(k*6,(k+1)*6-1) = body_forces.col(k+1);
			store_generalized_forces.col(i) = generalized_forces.t();


		}

	}

		plot_data(store_velocities, thetas, n,"velocities");
		plot_data(store_accelerations, thetas, n,"accelerations");
		plot_data(store_forces, thetas, n,"forces");

		ParsedData parsed_thetas = parseResults(to_be_parsed);
		plot_thetas(parsed_thetas,n);


	}