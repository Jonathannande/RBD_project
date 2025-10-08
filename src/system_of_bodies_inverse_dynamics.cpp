//
// Created by Jonathan on 10/7/2025.
//

#import "system_of_bodies.h"

	ParsedData SystemOfBodies::inverse_run_funcs() {

		/*
    typedef exprtk::symbol_table<double> symbol_table_t;
    typedef exprtk::expression<double> expression_t;
    typedef exprtk::parser<double> parser_t;

    ParsedData data;

    // Compile trajectory functions for all bodies
    std::vector<std::vector<double>> time_vars(n);
    std::vector<std::vector<expression_t>> pos_expressions(n);
    std::vector<std::vector<expression_t>> vel_expressions(n);
    std::vector<std::vector<expression_t>> acc_expressions(n);

    for (int body_idx = 0; body_idx < n; ++body_idx) {
        Body* body = bodies[body_idx].get();
        int n_dofs = system_dofs_distribution[body_idx];
        std::vector<std::string>& funcs = body->inverse_dynamics_funcs;

        // Expected: [pos1, vel1, acc1, pos2, vel2, acc2, ...] for n_dofs
        if (funcs.size() != 3 * n_dofs) {
            std::cerr << "Error: Body " << body_idx << " expected " << 3 * n_dofs
                      << " functions but got " << funcs.size() << std::endl;
            throw std::runtime_error("Invalid number of trajectory functions");
        }

        // Create time variables and symbol tables for this body
        time_vars[body_idx].resize(3 * n_dofs, 0.0);

        // Compile position, velocity, and acceleration expressions
        for (int dof = 0; dof < n_dofs; ++dof) {
            parser_t parser;

            // Position function
            expression_t pos_expr;
            symbol_table_t pos_symtab;
            pos_symtab.add_variable("t", time_vars[body_idx][3*dof]);
            pos_symtab.add_constants();
            pos_expr.register_symbol_table(pos_symtab);

            if (!parser.compile(funcs[3*dof], pos_expr)) {
                std::cerr << "Error compiling position function for body " << body_idx
                          << ", DOF " << dof << ": " << funcs[3*dof] << std::endl;
                std::cerr << "Parser error: " << parser.error() << std::endl;
                throw std::runtime_error("Function compilation failed");
            }
            pos_expressions[body_idx].push_back(pos_expr);

            // Velocity function
            expression_t vel_expr;
            symbol_table_t vel_symtab;
            vel_symtab.add_variable("t", time_vars[body_idx][3*dof + 1]);
            vel_symtab.add_constants();
            vel_expr.register_symbol_table(vel_symtab);

            if (!parser.compile(funcs[3*dof + 1], vel_expr)) {
                std::cerr << "Error compiling velocity function for body " << body_idx
                          << ", DOF " << dof << ": " << funcs[3*dof + 1] << std::endl;
                std::cerr << "Parser error: " << parser.error() << std::endl;
                throw std::runtime_error("Function compilation failed");
            }
            vel_expressions[body_idx].push_back(vel_expr);

            // Acceleration function
            expression_t acc_expr;
            symbol_table_t acc_symtab;
            acc_symtab.add_variable("t", time_vars[body_idx][3*dof + 2]);
            acc_symtab.add_constants();
            acc_expr.register_symbol_table(acc_symtab);

            if (!parser.compile(funcs[3*dof + 2], acc_expr)) {
                std::cerr << "Error compiling acceleration function for body " << body_idx
                          << ", DOF " << dof << ": " << funcs[3*dof + 2] << std::endl;
                std::cerr << "Parser error: " << parser.error() << std::endl;
                throw std::runtime_error("Function compilation failed");
            }
            acc_expressions[body_idx].push_back(acc_expr);
        }
    }

    // Evaluate at each time step
    for (double time = t0; time < t; time += dt) {
        data.times.push_back(time);

        // Evaluate all bodies at this time
        for (int body_idx = 0; body_idx < n; ++body_idx) {
            int n_dofs = system_dofs_distribution[body_idx];
            std::vector<double> pos_vals(n_dofs);
            std::vector<double> vel_vals(n_dofs);
            std::vector<double> acc_vals(n_dofs);

            for (int dof = 0; dof < n_dofs; ++dof) {
                // Update time variables
                time_vars[body_idx][3*dof] = time;
                time_vars[body_idx][3*dof + 1] = time;
                time_vars[body_idx][3*dof + 2] = time;

                // Evaluate expressions
                pos_vals[dof] = pos_expressions[body_idx][dof].value();
                vel_vals[dof] = vel_expressions[body_idx][dof].value();
                acc_vals[dof] = acc_expressions[body_idx][dof].value();
            }

            // Store in ParsedData (assuming it has vectors for storage)
            data.pos.push_back(pos_vals);
            data.vel.push_back(vel_vals);
            data.accel.push_back(acc_vals);
        }
    }

    return data;
		*/
}

	void SystemOfBodies::solve_inverse_dynamics() {
		//ParsedData thetas = inverse_run_funcs();

	//not very happy with this implementation but will stay for now

/*
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

			//might not be computationally efficient though it has general implementation - gravity should be removed eventually
			body_accelerations.col(k) = spatial_operator_dt(arma::span(6*(k+1),6*(k+2)-1),arma::span(6*(k+1),6*(k+2)-1)).t()*body_accelerations.col(k+1)+hinge_map_1_transpose*parsed_thetas.accel[k][i]+coriolis_vector(hinge_map_1_transpose,body_velocities.col(k),parsed_thetas.vel[k][i])+gravity_vector.col(k);
		}

		//gather sweep
		for (int k = 0; k < n; ++k)
		{
			//is not efficient, you only need to call inertia once per body, inertia matrix should be a per body computation in case boddies are different,
			body_forces.col(k+1) = spatial_operator_dt(arma::span(6*(k),6*(k+1)-1),arma::span(6*(k),6*(k+1)-1))*body_forces.col(k)+inertia_matrix_rect_z(mass,l,b,h,position_p)*(body_accelerations.col(k))+gyroscopic_force_z(inertia_matrix_rect_z(mass,l,b,h,position_p),body_velocities.col(k));
			//not much to add here
			generalized_forces.col(k) = hinge_map_1*body_forces.col(k+1);
			//std::cout << k << std::endl;

		}
		//std::cout << body_forces << std::endl;
		for (int k = 0; k < n; ++k)
		{
			//std::cout << "hello";
			store_velocities.col(i).rows(k*(6),(k+1)*(6)-1) = body_velocities.col(k);
			store_accelerations.col(i).rows(k*(6),(k+1)*(6)-1) = body_accelerations.col(k);
			store_forces.col(i).rows(k*(6),(k+1)*(6)-1) = body_forces.col(k+1);
			store_generalized_forces.col(i) = generalized_forces.t();
		}


	}
		*/
	}