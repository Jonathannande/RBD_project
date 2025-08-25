#include <boost/numeric/odeint.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/quadrature/gauss.hpp>
#include <armadillo>
#include <iostream>
#include <cmath>
#include <vector>
#include <SFML/Graphics.hpp>
#include "matplotlibcpp.h"
#include <boost/numeric/odeint/stepper/runge_kutta_cash_karp54.hpp>
#include <memory>


namespace plt = matplotlibcpp;
using namespace boost::math::float_constants;
using namespace boost::numeric::odeint;




arma::mat tilde(arma::mat vector) {
	arma::mat tilde_matrix = {
		{0,-vector(2),vector(1)},
		{vector(2),0,-vector(0)},
		{-vector(1),vector(0),0}
	};
	//std::cout << tilde_matrix << std::endl;
	return tilde_matrix;
}

arma::mat rotation_matrix(arma::mat rotation_vec) {
	//basic euler 

	double theta = rotation_vec(0);
	double phi = rotation_vec(1);
	double gamma = rotation_vec(2);

	arma::mat R_x = {{1,0         ,0		  },
					 {0,cos(theta),-sin(theta)},
					 {0,sin(theta), cos(theta)},
					
	};
	arma::mat R_y = {{cos(phi) ,0,sin(phi)},
					 {0        ,1,0       },
					 {-sin(phi),0,cos(phi)}
};
	arma::mat R_z = {{cos(gamma),-sin(gamma),0},
					 {sin(gamma), cos(gamma),0},
					 {0         ,0          ,1}
};	
	arma::mat R_xyz = R_x*R_y*R_z;
	return R_xyz;
}

arma::mat rb_transform(arma::mat rotation_vec, arma::mat position_vec) {
	// define rotation matrix and tilde position matrix
	arma::mat rotations = rotation_matrix(rotation_vec); 
	arma::mat position_vec_tilde = tilde(position_vec.t());

	// rotate positions to the new frame
	arma::mat rotated_positions = position_vec_tilde*rotations;
	

	// join the matrices and construct the rigid body transform
	arma::mat upper = join_horiz(rotations,rotated_positions);
	arma::mat lower = join_horiz(arma::zeros(3,3),rotations);
	arma::mat rbt = join_vert(upper,lower);

	return rbt;
}

arma::mat rb_transform_transpose(arma::mat rbt) {
	arma::mat rbt_transpose = rbt.t();
	return rbt_transpose;
}

//6x6 matrix used in various computations
arma::mat tilde_velocity(arma::mat velocity_vector) {
	
	// do the tildes
	arma::mat angular_tilde = tilde(velocity_vector.rows(0,2));
	arma::mat translational_tilde = tilde(velocity_vector.rows(3,5));
	
	// assemble the upper and lower halfs
	arma::mat upper = join_horiz(angular_tilde,arma::zeros<arma::mat>(3, 3));
	arma::mat lower = join_horiz(translational_tilde,angular_tilde);

	//assemble the matrix
	arma::mat tilde_velocity = join_vert(upper,lower);

	return tilde_velocity;
}

//transpose of tilde_velocity
arma::mat bar_velocity(arma::mat velocity_vector) {
	// do the tildes
	arma::mat angular_tilde = tilde(velocity_vector.rows(0,2));
	arma::mat translational_tilde = tilde(velocity_vector.rows(3,5));
	
	// assemble the upper and lower halfs
	arma::mat upper = join_horiz(angular_tilde,translational_tilde);
	arma::mat lower = join_horiz(arma::zeros<arma::mat>(3, 3),angular_tilde);

	//assemble the matrix
	arma::mat tilde_velocity = join_vert(upper,lower);

	return tilde_velocity;
}

// gyroscopic force for body frame, eq. 2.28
arma::mat gyroscopic_force_z(arma::mat inertial_matrix, arma::mat velocity_vector) {

	arma::mat velocity_bar = bar_velocity(velocity_vector);
	arma::mat gyroscopic_force_z = velocity_bar*inertial_matrix*velocity_vector;
	return gyroscopic_force_z;
}

// coriolis term for body frame, eq. 5.11 - without the differential hinge map part
arma::mat coriolis_vector(arma::mat hinge_map,arma::mat velocity_vector,arma::vec generalized_velocities) {
	arma::mat tilde_velocity_result = tilde_velocity(velocity_vector);
	arma::mat delta_velocity = hinge_map*generalized_velocities;
	arma::mat delta_velocity_bar = bar_velocity(delta_velocity);

	arma::mat term_1 = tilde_velocity_result*delta_velocity;
	arma::mat term_2 = delta_velocity_bar*delta_velocity;

	arma::mat coriolis_vector = term_1-term_2;
	return coriolis_vector;
}
// consists of all the rigid body transforms of the system
arma::mat find_spatial_operator(arma::mat rigid_body_transform_vector, int n) {
	arma::mat I = arma::eye(6,6);
	arma::mat spatial_operator = arma::zeros(6*(n+2),6*(n+2));
	spatial_operator(arma::span(0,5),arma::span(0,5)) = I;
	spatial_operator(arma::span(6*(n+1),6*(n+2)-1),arma::span(6*(n+1),6*(n+2)-1)) = I;

	for (int i = 1; i < n+1; ++i)
	{
		
		spatial_operator(arma::span(6*i,6*(i+1)-1),arma::span(6*(i),6*(i+1)-1)) = rb_transform(rigid_body_transform_vector.row(i-1).cols(0, 2),rigid_body_transform_vector.row(i-1).cols(3, 5));
	}
	//std::cout << spatial_operator << std::endl;

	return spatial_operator;
}

struct ParsedData {
    std::vector<double> times;
    std::vector<std::vector<double>> accel;
    std::vector<std::vector<double>> vel;
    std::vector<std::vector<double>> pos;
};

//plots velocities - OBS: takes the ParsedData type - set up for specifically 3 bodies (matplotlib does not work well in cpp)
void plot_thetas(ParsedData parsed_thetas) {

	// Set the size of output image to 1200x780 pixels
    plt::figure_size(1200, 780);
    // Convert store_velocities.row(0) to std::vector<double>
	
	//std::cout << store_velocities.col(1) ;
	// Convert parsed_thetas.times to std::vector<double> (if not already a vector)
	std::vector<double> time_vector(parsed_thetas.times.begin(), parsed_thetas.times.end());
	for (int i = 0; i <= 2; ++i)
	{
		std::vector<double> position_theta(parsed_thetas.pos[i].begin(), parsed_thetas.pos[i].end());
		plt::plot(time_vector, position_theta);
		plt::grid(true);
	}

	std::string title = "positional theta plot";  // Set a single title string
	plt::title(title);  // Now, this will work
    plt::save("output_folder/position_thetas.png");
    plt::clf();

	for (int i = 0; i <= 2; ++i)
	{
		std::vector<double> velocity_theta(parsed_thetas.vel[i].begin(), parsed_thetas.vel[i].end());
		plt::plot(time_vector, velocity_theta);
		plt::grid(true);
	}

	title = "velocity theta plot";  // Set a single title string
	plt::title(title);  // Now, this will work
    plt::save("output_folder/velocity_thetas.png");
    plt::clf();

    for (int i = 0; i <= 2; ++i)
	{
		std::vector<double> acceleration_theta(parsed_thetas.accel[i].begin(), parsed_thetas.accel[i].end());
		plt::plot(time_vector, acceleration_theta);
		plt::grid(true);
	}

	title = "acceleration theta plot";  // Set a single title string
	plt::title(title);  // Now, this will work
    plt::save("output_folder/acceleration_thetas.png");
    plt::clf();


}

//plots velocities - OBS: takes the ParsedData type - set up for specifically 3 bodies (matplotlib does not work well in cpp)
void plot_velocities(arma::mat store_velocities,ParsedData parsed_thetas) {

	// Set the size of output image to 1200x780 pixels
    plt::figure_size(1200, 780);
    // Convert store_velocities.row(0) to std::vector<double>
	
	//std::cout << store_velocities.col(1) ;
	// Convert parsed_thetas.times to std::vector<double> (if not already a vector)
	std::vector<double> time_vector(parsed_thetas.times.begin(), parsed_thetas.times.end() - 1);
	for (int i = 12; i <= 17; ++i)
	{
		std::vector<double> velocity_row3(store_velocities.row(i).begin(), store_velocities.row(i).end());
		plt::plot(time_vector, velocity_row3);
		plt::grid(true);

		//std::cout << store_velocities.row(i) << std::endl;
	}

	std::string title3 = "vel_body_3";  // Set a single title string
	plt::title(title3);  // Now, this will work
    plt::save("output_folder/velocity_body_3.png");
    plt::clf();

	for (int i = 6; i <= 11; ++i)
	{
		std::vector<double> velocity_row2(store_velocities.row(i).begin(), store_velocities.row(i).end());
		plt::plot(time_vector, velocity_row2);
		plt::grid(true);

	}
	std::string title2 = "vel_body_2";  // Set a single title string
	plt::title(title2);  // Now, this will work
    plt::save("output_folder/velocity_body_2.png");
	plt::clf();

	for (int i = 0; i <= 5; ++i)
	{
		std::vector<double> velocity_row1(store_velocities.row(i).begin(), store_velocities.row(i).end());
		plt::plot(time_vector, velocity_row1);
		plt::grid(true);

		//std::cout << store_velocities.row(i) << std::endl;
	}
	std::string title1 = "vel_body_1";  // Set a single title string
	plt::title(title1);  // Now, this will work
    plt::save("output_folder/velocity_body_1.png");
    plt::clf();

}
//plots accelerations - OBS: takes the ParsedData type
void plot_accelerations(arma::mat store_accelerations,ParsedData parsed_thetas) {

	// Set the size of output image to 1200x780 pixels
    plt::figure_size(1200, 780);
    
	
	
	// Convert parsed_thetas.times to std::vector<double>
	std::vector<double> time_vector(parsed_thetas.times.begin(), parsed_thetas.times.end() - 1);
	for (int i = 12; i <= 17; ++i)
	{
		std::vector<double> acceleration_row3(store_accelerations.row(i).begin(), store_accelerations.row(i).end());
		plt::plot(time_vector, acceleration_row3);
		plt::grid(true);

		//std::cout << store_velocities.row(i) << std::endl;
	}

	std::string title3 = "acc_body_3";
	plt::title(title3); 
    plt::save("output_folder/acceleration_body_3.png");
    plt::clf();

	for (int i = 6; i <= 11; ++i)
	{
		std::vector<double> acceleration_row2(store_accelerations.row(i).begin(), store_accelerations.row(i).end());
		plt::plot(time_vector, acceleration_row2);
		plt::grid(true);

	}
	std::string title2 = "acc_body_2"; 
	plt::title(title2);  // Now, this will work
    plt::save("output_folder/acceleration_body_2.png");
	plt::clf();

	for (int i = 0; i <= 5; ++i)
	{
		std::vector<double> acceleration_row1(store_accelerations.row(i).begin(), store_accelerations.row(i).end());
		plt::plot(time_vector, acceleration_row1);
		plt::grid(true);
		//std::cout << store_velocities.row(i) << std::endl;
	}
	std::string title1 = "acc_body_1";
	plt::title(title1);  
    plt::save("output_folder/acceleration_body_1.png");
    plt::clf();

}

//plots forces - OBS: takes the ParsedData type
void plot_forces(arma::mat store_forces,ParsedData parsed_thetas) {

	// Set the size of output image to 1200x780 pixels
    plt::figure_size(1200, 780);
    // Convert store_velocities.row(0) to std::vector<double>
	
	//std::cout << store_velocities.col(1) ;
	// Convert parsed_thetas.times to std::vector<double> (if not already a vector)
	std::vector<double> time_vector(parsed_thetas.times.begin(), parsed_thetas.times.end() - 1);
	for (int i = 12; i <= 17; ++i)
	{
		std::vector<double> forces_row3(store_forces.row(i).begin(), store_forces.row(i).end());
		plt::plot(time_vector, forces_row3);
		plt::grid(true);
		//std::cout << store_velocities.row(i) << std::endl;
	}

	std::string title3 = "forces_body_3";  // Set a single title string
	plt::title(title3);  // Now, this will work
    plt::save("output_folder/forces_body_3.png");
    plt::clf();

	for (int i = 6; i <= 11; ++i)
	{
		std::vector<double> forces_row2(store_forces.row(i).begin(), store_forces.row(i).end());
		plt::plot(time_vector, forces_row2);
		plt::grid(true);

	}
	std::string title2 = "forces_body_2";  // Set a single title string
	plt::title(title2);  // Now, this will work
    plt::save("output_folder/forces_body_2.png");
	plt::clf();

	for (int i = 0; i <= 5; ++i)
	{
		std::vector<double> forces_row1(store_forces.row(i).begin(), store_forces.row(i).end());
		plt::plot(time_vector, forces_row1);
		plt::grid(true);

		//std::cout << store_velocities.row(i) << std::endl;
	}
	std::string title1 = "forces_body_1";  // Set a single title string
	plt::title(title1);  // Now, this will work
    plt::save("output_folder/forces_body_1.png");
    plt::clf();

}
//plots generalized forces - OBS: takes the ParsedData type
void plot_generalized_forces(arma::mat store_generalized_forces, ParsedData parsed_thetas) {
	// Set the size of output image to 1200x780 pixels
    plt::figure_size(1200, 780);
    
	// Convert parsed_thetas.times to std::vector<double> (if not already a vector)
	std::vector<double> time_vector(parsed_thetas.times.begin(), parsed_thetas.times.end() - 1);
	for (int i = 0; i <= 2; ++i)
	{
		std::vector<double> generalized_forces(store_generalized_forces.row(i).begin(), store_generalized_forces.row(i).end());
		plt::plot(time_vector, generalized_forces);
		plt::grid(true);
		//std::cout << store_velocities.row(i) << std::endl;
	}
	std::string title = "Generalized_forces";  // Set a single title string
	plt::title(title);  // Now, this will work
    plt::save("output_folder/generalized_forces.png");
    plt::clf();
}










//the generic body type which the system boides stores, this class holds the traits which all bodies share - mass, a hinge-type, hinge postition from center of mass, RBT 
class Body {
public:
	arma::mat hinge_map = {0,0,1,0,0,0};
	arma::mat transpose_hinge_map = hinge_map.t();
	std::string hinge_type; //might be used later to display generic hinge types
	arma::vec position_vec_hinge; //3X1 position vector used for inertia computations
	arma::vec outboard_position_vec_hinge; //3X1 position vector used for transform from center of mass
	arma::vec position_vec_hinge_big; //6X1 position vector used for inertia computations
	arma::vec outboard_position_vec_hinge_big; //6X1 position vector used for transform from center of mass

	//arma::vec rotation_vec hinge; //should maybe be used the general formulation of the program	
	arma::vec state = {0,0}; //this is the initial conditions of the body, should use its generalized velocity/coordinates of which there might be multiple
	arma::mat inertial_matrix;
	double m;

	Body(double mass) : m(mass) { }

	virtual void compute_inertia_matrix() {};

	void set_postion_vec_hinge(arma::vec input_vec) {
		position_vec_hinge = input_vec;
		position_vec_hinge_big = join_vert(arma::zeros(3,1),input_vec);
		//std::cout << position_vec_hinge_big << std::endl;
	}

	void set_outboard_position_vec_hinge(arma:: vec input_vec) {
		outboard_position_vec_hinge = input_vec;
		outboard_position_vec_hinge_big = join_vert(arma::zeros(3,1),input_vec);
		//std::cout << outboard_position_vec_hinge_big << std::endl;
	}

	void set_hinge_map(arma::mat hinge_map_input) {
		hinge_map = hinge_map_input;
		transpose_hinge_map = hinge_map.t();
		state = arma::zeros(2*hinge_map_input.n_cols);
	}

	void set_hinge_state(arma::vec hinge_state_input) {
		if (2*hinge_map.n_rows == hinge_state_input.n_rows)
		{
			state = hinge_state_input;
			//std::cout << state;
		}
	}

};
class Rectangle: public Body {

public:
	double l;
	double b;
	double h;
	arma::vec transform_vector = {0,0,0,l,0,0}; //this is your transform vector - descrives the transfer the next hinge, the curre
	


	Rectangle(double length, double breadth, double height, double mass)
        : Body(mass), l(length), b(breadth), h(height) { }


	void compute_inertia_matrix()  override 	{
		//l,b, and h refer to the dimensions of length, breadth, and height
		//inertial matrix
		arma::mat inertial_matrix_rectangle = {{m*(1.0/12.0)*(std::pow(b,2)+std::pow(l,2)),0,0},
												 {0,m*(1.0/12.0)*(std::pow(b,2)+std::pow(h,2)),0},
												 {0,0,m*(1.0/12.0)*(std::pow(h,2)+std::pow(l,2))}};
		
		//parallel axis theorem
		arma::mat position_vec_tilde = tilde(-position_vec_hinge);
		arma::mat I_corner = inertial_matrix_rectangle+m*(position_vec_tilde.t()*position_vec_tilde);
		arma::mat I = arma::eye(3,3);
		arma::mat position_vec_tilde_non_neg = tilde(position_vec_hinge);

		//join the matrices and construct the inertia matrix
		arma::mat upper = join_horiz(I_corner,m*position_vec_tilde_non_neg);
		arma::mat lower = join_horiz(m*position_vec_tilde,m*I);

		inertial_matrix = join_vert(upper,lower);
	}

};

class Sphere: public Body {
public:
	double r;

};


//we need to use some pointser for this thing
class SystemOfBodies {

public:
	int n {0}; //numbers of bodies in system
	double t0 {0.0}; //time at beginning of simulation
	double t {2.0};	//time at end of simulation
	double dt {0.1}; //time step
	double system_gravity {9.81}; //gravity applied to system
	int system_total_dof = 0; //adds all the systems dofs - used for various computations, thus computated at body creation
	std::vector<int> system_dofs_distribution; //stores total dofs of each body in each index. Useful for various computations, when considering multi dof single bodies
	std::vector<double> system_equations; //for inverse dynamics
	std::vector<std::unique_ptr<Body>> bodies; //storage of each body with pointer
	std::vector<double> system_state; //stores entire initial state of system, first all positions, then all velocities
	arma::mat system_hinge; //not sure what this is

	
	//modified parsed result to fit arbitrary hinge_map
	ParsedData parseResults(const std::vector<double>& results, int system_total_dof)
	{
	    
	    const size_t row_size = 1 + 3*system_total_dof; 
	    const size_t steps = results.size() / row_size;

	    ParsedData data;
	    data.times.resize(steps);
	    data.accel.assign(system_total_dof, std::vector<double>(steps));
	    data.vel.assign(system_total_dof, std::vector<double>(steps));
	    data.pos.assign(system_total_dof, std::vector<double>(steps));

	    for (size_t s = 0; s < steps; s++)
	    {
	        size_t offset = s * row_size;
	        data.times[s] = results[offset];

	        // accelerations
	        for (size_t i = 0; i < system_total_dof; i++) {
	            data.accel[i][s] = results[offset + 1 + i];
	            data.vel[i][s] = results[offset + 1 + system_total_dof + i];
	            data.pos[i][s] = results[offset + 1 + 2*system_total_dof + i];
	        }
	    }

	    return data;
	}
	
	void update_system_state() {
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
	
	

	//this function, recives the generalized positions at each time step, converts the dofs of each body and uses it to compute the input vector needed for the spatial operator.
	arma::mat find_spatial_operator_input_vector(std::vector<arma::vec>& DoFs) {
		arma::mat return_vector = arma::zeros(n,6);

		//uses the modifyed theta
		for (int i = 0; i<n; ++i) {
			//std::cout << bodies[i]->position_vec_hinge_big << std::endl;
			//std::cout << bodies[i]->outboard_position_vec_hinge_big << std::endl;
			return_vector.row(i) = (bodies[i]->hinge_map.t()*DoFs[i] - (bodies[i]->position_vec_hinge_big-bodies[i]->outboard_position_vec_hinge_big)).t();
			//std::cout << return_vector.row(i) << std::endl;
		}
		return return_vector;
	}

	std::vector<arma::vec> to_arma_vec(std::vector<double> DoFs) {
		//std::cout << "yo";
		std::vector<arma::vec> return_vector;
		
		int start_idx = 0;
		//matbe shouldn't set it to false i dont know
		for (int i = 0; i<n; ++i) {
			
			return_vector.push_back(arma::vec(DoFs.data() + start_idx, system_dofs_distribution[i], false));			
			start_idx += system_dofs_distribution[i];
		}
		return return_vector;
	}

	std::vector<double> to_std_vec(std::vector<arma::vec>& DoFs) {
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
	arma::mat find_gravity_matrix_pseudo_acceleration() {
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
	
	void test_method() {
		//takes current state of the solver and creates the generalized coordinates
		//std::cout << "hello";
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
	
	void system_of_equations_forward_dynamics(const std::vector<double> &y, std::vector<double> &dydt, double t) {
		

		//takes current state of the solver and creates the generalized coordinates
	    std::vector<double> theta_standard_form(y.begin() , y.end()-y.size()/2);
	    std::vector<double> theta_dot_standard_form(y.begin()+y.size()/2, y.end());
	    std::vector<arma::vec> theta = to_arma_vec(theta_standard_form);
	    std::vector<arma::vec> theta_dot = to_arma_vec(theta_dot_standard_form);
	    for (int i = 0; i < theta.size(); ++i)
	    
	    {
	    	std::cout << "The postional state of theta " << i << " has the value of " << theta[i]<< std::endl;
	    }
	    

	    //hingemaps of the system
		//arma::mat hinge_map_1 = {0,0,1,0,0,0}; 
		//arma::mat hinge_map_1_transpose = hinge_map_1.t();

	    //matrix initialization this should be entirely unecessaty from a computational point of view and should be fixed to only stroe values which are needed on a later occasion

	    arma::mat P_plus = arma::zeros(6*(n+1), 6*(n+1));
	    arma::mat J_fractal_plus = arma::zeros(6, n+1);
	    arma::mat tau_bar = arma::zeros(6*n, 6*(n+1));
	    arma::mat P = arma::zeros(6*n, 6*(n+1));
	    arma::mat D;// = arma::zeros(n, n);
	    std::vector<arma::mat> G_fractal(n);// = arma::zeros(6, n+1);
	    arma::mat J_fractal = arma::zeros(6, n+1);
	    arma::mat eta;// = arma::zeros(1, n+1);
	    std::vector<arma::mat> frac_v(n);// = arma::zeros(1, n);
	    arma::mat accel = arma::zeros(6, n+1);
	    accel(4, n) = system_gravity;  // Equivalent to MATLAB's `accel(5,n+1) = -g;`
	    arma::mat accel_plus = arma::zeros(6, n+1);
	    arma::mat body_velocities = arma::zeros(6, n+1);
	    arma::mat inertia_matrix;
		
	    //we really need to do something about this
	    arma::mat transform_matrix = find_spatial_operator_input_vector(theta);
	    std::cout << transform_matrix << std::endl;
	    arma::mat spatial_operator_dt = find_spatial_operator(transform_matrix, n);
	    

		

		//now we start sweeping 3 times in total, first kinematics which has in part already been done through the spatial operator, but now velocities
		
		for (int k = n-1; k >= 0; --k) {
			body_velocities.col(k) = spatial_operator_dt(arma::span(6*(k+1),6*(k+2)-1),arma::span(6*(k+1),6*(k+2)-1)).t()*body_velocities.col(k+1)+bodies[k]->transpose_hinge_map*theta_dot[k];
		}

		//these are arrays storing the kth interval index, just so that you dont have to do the annoying indexing all the time
		std::array<int, 2> k_idx1;
		std::array<int, 2> k_idx2;

		for (int k = 0; k <= n-1; ++k) {

			k_idx1 = {6*(k),6*(k+1)-1};
			k_idx2 = {6*(k+1),6*(k+2)-1};

			bodies[k]->compute_inertia_matrix(); 

			arma::mat inertia_matrix = bodies[k]->inertial_matrix; 
			P(arma::span(k_idx1[0],k_idx1[1]),arma::span(k_idx1[0],k_idx1[1])) = spatial_operator_dt(arma::span(k_idx1[0],k_idx1[1]),arma::span(k_idx1[0],k_idx1[1]))*P_plus(arma::span(k_idx1[0],k_idx1[1]),arma::span(k_idx1[0],k_idx1[1]))*spatial_operator_dt(arma::span(k_idx1[0],k_idx1[1]),arma::span(k_idx1[0],k_idx1[1])).t()+inertia_matrix;
			D = bodies[k]->hinge_map*P(arma::span(k_idx1[0],k_idx1[1]),arma::span(k_idx1[0],k_idx1[1]))*bodies[k]->transpose_hinge_map;
			
			//std::cout <<"D = " << D << std::endl;
			//std::cout << "debugging "<< k << std::endl;
			//std::cout << P(arma::span(k_idx1[0], k_idx1[1]),arma::span(k_idx1[0], k_idx1[1]))*bodies[k]->transpose_hinge_map*arma::inv(arma::mat{D}) << std::endl;
			G_fractal[k] = P(arma::span(k_idx1[0], k_idx1[1]),arma::span(k_idx1[0], k_idx1[1]))*bodies[k]->transpose_hinge_map*arma::inv(arma::mat{D});
			//std::cout << "debugging here at 563" << std::endl;
			//std::cout << G_fractal << " G_fractal" << std::endl;
			//std::cout << bodies[k]->hinge_map << " hinge_map" << std::endl;

			tau_bar(arma::span(k_idx1[0],k_idx1[1]),arma::span(k_idx1[0],k_idx1[1])) = arma::eye(6,6)-G_fractal[k]*bodies[k]->hinge_map;
			
			P_plus(arma::span(k_idx2[0],k_idx2[1]),arma::span(k_idx2[0],k_idx2[1])) = tau_bar(arma::span(k_idx1[0],k_idx1[1]),arma::span(k_idx1[0],k_idx1[1]))*P(arma::span(k_idx1[0],k_idx1[1]),arma::span(k_idx1[0],k_idx1[1]));

			J_fractal.col(k) = spatial_operator_dt(arma::span(k_idx1[0],k_idx1[1]),arma::span(k_idx1[0],k_idx1[1]))*J_fractal_plus.col(k)+P(arma::span(k_idx1[0],k_idx1[1]),arma::span(k_idx1[0],k_idx1[1]))*coriolis_vector(bodies[k]->transpose_hinge_map,body_velocities.col(k),theta_dot[k])+gyroscopic_force_z(inertia_matrix,body_velocities.col(k));
				
			eta = -bodies[k]->hinge_map*J_fractal.col(k);

			frac_v[k] = arma::solve(D,eta);

			J_fractal_plus.col(k+1) = J_fractal.col(k)+G_fractal[k]*eta;
			
			
		}
		
		std::vector<arma::vec> theta_ddot(n);
		//std::cout << "adhnfgsiodjfo" << std::endl;
		for (int k = n-1; k >= 0; --k) {
			k_idx1 = {6*(k),6*(k+1)-1};
			k_idx2 = {6*(k+1),6*(k+2)-1};
			accel_plus.col(k) = spatial_operator_dt(arma::span(k_idx2[0],k_idx2[1]),arma::span(k_idx2[0],k_idx2[1])).t()*accel.col(k+1);
			//std::cout << accel_plus.col(k)<< std::endl;
			theta_ddot[k] = frac_v[k] - G_fractal[k].t()*accel_plus.col(k);
			//std::cout << "hellfaoisjdhfpoasid" << std::endl;
			
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

	}
	
	void solve_inverse_dynamics() {

	}
	
	void solve_forward_dynamics() {

		
	    // Single vector to store data
	    std::vector<double> results;

	    // Observer lambda - OBS does't work for the current state model
		auto obs = [&](const std::vector<double>& y, double t) {
		    // Create a temporary vector to hold the derivative.
		    std::vector<double> dydt(y.size(), 0.0);
		    
		    // Compute the derivative at the current state and time.
		    system_of_equations_forward_dynamics(y, dydt, t);
		    
		    
		    results.push_back(t);
		    
		    
		    for (size_t i = n; i < 2*system_total_dof; i++) {
		         results.push_back(dydt[i]);
		         results.push_back(y[i]);
		    }
		    
		    for (size_t i = 0; i < system_total_dof; i++) {
		         results.push_back(y[i]);
		    }
		};

		//starting time
	    
	    runge_kutta_cash_karp54<std::vector<double>> stepper;
	    integrate_const(
		    stepper,
		    [this](const std::vector<double>& y, std::vector<double>& dydt, double t) {
		        system_of_equations_forward_dynamics(y, dydt, t);
		    },
		    system_state, t0, t, dt, obs
		);


	    std::cout << "done" << std::endl;
	    
	    //std::cout << results.size() << std::endl;

	    //ParsedData formatted_results = parseResults(results,system_total_dof);

	    //compute_body_forward_dynamics(formatted_results);

	}
	
	void solve_hybrid_dynamics() {

	}
	//method that adds body to system
	void create_body(std::unique_ptr<Body> any_body) {
    //std::cout << "Adding body to system..." << std::endl;

    // Move the unique_ptr into the container and get a pointer to the inserted object
    auto it = bodies.insert(bodies.begin(), std::move(any_body)); // Move ownership
    Body* inserted_body = it->get(); // Retrieve the raw pointer

    // Now use inserted_body to access methods and properties
    ++n;
    //std::cout << "Updated n: " << n << std::endl;

    system_dofs_distribution.insert(system_dofs_distribution.begin(),inserted_body->hinge_map.n_rows);
    //std::cout << "Updated system_dofs_distribution size: " << system_dofs_distribution[n-1] << std::endl;

    system_total_dof += inserted_body->hinge_map.n_rows;
    //std::cout << system_total_dof << std::endl;
    // Example: Call methods
    //inserted_body->compute_inertia_matrix(); // Call method
    //std::cout << "Inertia matrix:\n" << inserted_body->inertial_matrix << std::endl;

    update_system_state();
    //std::cout << bodies[0]->hinge_map << std::endl;
	}

};



int main()
{
	auto rec_1 = std::make_unique<Rectangle>(2.0,8,0.2,0.1);
	rec_1->set_postion_vec_hinge({0, -rec_1->l/2.0, 0});
	rec_1->set_outboard_position_vec_hinge({0, rec_1->l/2.0, 0});
	rec_1->set_hinge_state({pi/2,0});
	rec_1->compute_inertia_matrix();

	auto rec_2 = std::make_unique<Rectangle>(2.0,8,0.2,0.1);
	arma::mat xx = {{0,1,0,0,0,0},{0,0,1,0,0,0}};
	arma::vec x = {pi/4,pi/3,0,0};
	rec_2->set_hinge_map(xx);
	rec_2->set_hinge_state(x);
	rec_2->set_postion_vec_hinge({0, -rec_1->l/2.0, 0});
	rec_2->set_outboard_position_vec_hinge({0, rec_1->l/2.0, 0});
	rec_2->compute_inertia_matrix();
	//std::cout <<rec_1->inertial_matrix << std::endl;

	SystemOfBodies system_1;
	system_1.create_body(move(rec_1));
	system_1.create_body(move(rec_2));
	

	system_1.solve_forward_dynamics();
	//std::cout << system_1.system_total_dof << std::endl;
	//std::cout << system_1.system_state[3] << std::endl;

	//std::cout << system_1.n <<std::endl << system_1.system_total_dof << std::endl << system_1.system_dofs_distribution << std::endl



	


	return 0;
}