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

//describes the inertia of a body, here it is a rectangles about a point z
arma::mat inertia_matrix_rect_z(double mass, double l, double b, double h, arma::vec position_vec) {
	//l,b, and h refer to the dimensions of length, breadth, and height
	//inertial matrix
	arma::mat inertial_matrix_rectangle = {{mass*(1.0/12.0)*(std::pow(b,2)+std::pow(l,2)),0,0},
											 {0,mass*(1.0/12.0)*(std::pow(b,2)+std::pow(h,2)),0},
											 {0,0,mass*(1.0/12.0)*(std::pow(h,2)+std::pow(l,2))}
};
	
	//parallel axis theorem
	arma::mat position_vec_tilde = tilde(-position_vec);
	arma::mat I_corner = inertial_matrix_rectangle+mass*(position_vec_tilde.t()*position_vec_tilde);
	arma::mat I = arma::eye(3,3);
	arma::mat position_vec_tilde_non_neg = tilde(position_vec);

	//join the matrices and construct the inertia matrix
	arma::mat upper = join_horiz(I_corner,mass*position_vec_tilde_non_neg);
	arma::mat lower = join_horiz(mass*position_vec_tilde,mass*I);

	arma::mat inertial_matrix = join_vert(upper,lower);
	//std::cout << inertial_matrix << std::endl;
	return inertial_matrix;
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
arma::mat coriolis_vector(arma::mat hinge_map,arma::mat velocity_vector,double generalized_velocities) {
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


arma::mat hinge_map_operator(arma::mat hinge_map_vector) {
	return 0;
}

arma::vec gravity_as_force(arma::mat spatial_operator, int k,int n) {
	
	double g = 9.81;
	arma::vec g_vec_inertial = {0,0,0,0,-g,0};

	for (int i = n-1; i >= 0; --i)
	{
		g_vec_inertial = spatial_operator(arma::span(6*(i+1),6*(i+2)-1),arma::span(6*(i+1),6*(i+2)-1)).t()*g_vec_inertial;
		//std::cout << g_vec_inertial;
	}
	//std::cout <<  std::endl;
	return g_vec_inertial;
}

// should use a specific combination of rigid body transforms
arma::mat find_inertial_frame_body_k(arma::mat global_rbt,int k) {


	return 0;
}


// System generalized accelerations
std::vector<double> accel(double t)
{
    std::vector<double> a(3);
    a[0] = -std::pow(pi,3)*t;
    a[1] = 2*std::pow(pi,3)*std::sin(2*pi*t + (pi/2.0));
    a[2] = -std::pow(pi,3)*std::sin(2*pi*t + (pi/2.0));
    return a;
}

//should be remade eventually maybe - this is only for inverse kinematics
void system_of_odes_with_accel_input(const std::vector<double> &y, std::vector<double> &dydt, double t)
{
    std::vector<double> a = accel(t);
    size_t n = a.size();

    // dv/dt = a(t)
    for(size_t i = 0; i < n; ++i)
        dydt[i] = a[i];

    // dx/dt = v
    for(size_t i = 0; i < n; ++i)
        dydt[n + i] = y[i];
}

void system_of_equations_forward_dynamics(const std::vector<double> &y, std::vector<double> &dydt, double t)
{
	//necessary for testing
    double mass = 8, l=2,  b=0.1,  h=0.2;
	double g = 9.81;
	int n = y.size()/2;
	
	//example positional vector - should be implemented in a function and stored with the body object
	arma::vec position_p = {0,-l/2,0};

    std::vector<double> theta(y.begin() , y.end()-3);
    //std::cout << y[1];
    std::vector<double> theta_dot(y.begin() + n, y.end());

    //hingemaps of the system
	arma::mat hinge_map_1 = {0,0,1,0,0,0}; 
	arma::mat hinge_map_1_transpose = hinge_map_1.t();

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
    accel(4, n) = g;  // Equivalent to MATLAB's `accel(5,n+1) = -g;`
    arma::mat accel_plus = arma::zeros(6, n+1);
    arma::mat body_velocities = arma::zeros(6, n+1);
    arma::mat inertia_matrix;

    //we really need to do something about this
    arma::mat transform_matix = {{0.5*theta[0],0,0.5*theta[0],0,-l,0},{0,0,theta[1],0,-l,0},{0,0,theta[2],0,-l,0}};
    arma::mat spatial_operator_dt = find_spatial_operator(transform_matix, n);


	//should be implemented in general and ideally as acting on the force instead further testing is necessary for that
	arma::mat gravity_vector = {{0,0,0},{0,0,0},{0,0,0},{0,0,g*sin(theta[2])},{0,0,g*cos(theta[2])},{0,0,0}};

	//now we start sweeping 3 times in total, first kinematics which has in part already been done through the spatial operator, but now velocities
	
	for (int k = n-1; k >= 0; --k) {
		body_velocities.col(k) = spatial_operator_dt(arma::span(6*(k+1),6*(k+2)-1),arma::span(6*(k+1),6*(k+2)-1)).t()*body_velocities.col(k+1)+hinge_map_1_transpose*theta_dot[k];
	}

	//these are arrays storing the kth interval index, just so that you dont have to do the annoying indexing all the time
	std::array<int, 2> k_idx1;
	std::array<int, 2> k_idx2;

	for (int k = 0; k <= n-1; ++k) {

		k_idx1 = {6*(k),6*(k+1)-1};
		k_idx2 = {6*(k+1),6*(k+2)-1};

		inertia_matrix = inertia_matrix_rect_z(mass,l,b,h,position_p);
		P(arma::span(k_idx1[0],k_idx1[1]),arma::span(k_idx1[0],k_idx1[1])) = spatial_operator_dt(arma::span(k_idx1[0],k_idx1[1]),arma::span(k_idx1[0],k_idx1[1]))*P_plus(arma::span(k_idx1[0],k_idx1[1]),arma::span(k_idx1[0],k_idx1[1]))*spatial_operator_dt(arma::span(k_idx1[0],k_idx1[1]),arma::span(k_idx1[0],k_idx1[1])).t()+inertia_matrix;
		D(k,k) = arma::as_scalar(hinge_map_1*P(arma::span(k_idx1[0],k_idx1[1]),arma::span(k_idx1[0],k_idx1[1]))*hinge_map_1_transpose);
		G_fractal.col(k) = P(arma::span(k_idx1[0], k_idx1[1]),arma::span(k_idx1[0], k_idx1[1]))*hinge_map_1_transpose*arma::inv(arma::mat{D(k,k)});
		tau_bar(arma::span(k_idx1[0],k_idx1[1]),arma::span(k_idx1[0],k_idx1[1])) = arma::eye(6,6)-G_fractal.col(k)*hinge_map_1;
		P_plus(arma::span(k_idx2[0],k_idx2[1]),arma::span(k_idx2[0],k_idx2[1])) = tau_bar(arma::span(k_idx1[0],k_idx1[1]),arma::span(k_idx1[0],k_idx1[1]))*P(arma::span(k_idx1[0],k_idx1[1]),arma::span(k_idx1[0],k_idx1[1]));
		J_fractal.col(k) = spatial_operator_dt(arma::span(k_idx1[0],k_idx1[1]),arma::span(k_idx1[0],k_idx1[1]))*J_fractal_plus.col(k)+P(arma::span(k_idx1[0],k_idx1[1]),arma::span(k_idx1[0],k_idx1[1]))*coriolis_vector(hinge_map_1_transpose,body_velocities.col(k),theta_dot[k])+gyroscopic_force_z(inertia_matrix,body_velocities.col(k));
		eta.col(k) = -hinge_map_1*J_fractal.col(k);

		frac_v(k) = arma::as_scalar(arma::solve(D(arma::span(k, k), arma::span(k, k)),eta.col(k)));
		J_fractal_plus.col(k+1) = J_fractal.col(k)+G_fractal.col(k)*eta.col(k);
		
	}
	
	std::vector<double> theta_ddot = {0,0,0};

	for (int k = n-1; k >= 0; --k) {
		k_idx1 = {6*(k),6*(k+1)-1};
		k_idx2 = {6*(k+1),6*(k+2)-1};
		accel_plus.col(k) = spatial_operator_dt(arma::span(k_idx2[0],k_idx2[1]),arma::span(k_idx2[0],k_idx2[1])).t()*accel.col(k+1);
		theta_ddot[k] = arma::as_scalar(frac_v(k) - G_fractal.col(k).t()*accel_plus.col(k));
		accel.col(k) = accel_plus.col(k)+hinge_map_1_transpose*theta_ddot[k]+coriolis_vector(hinge_map_1_transpose,body_velocities.col(k),theta_dot[k]);
	}

	    // dv/dt = a(t)
    for(size_t i = 0; i < n; ++i)
        dydt[i] = theta_dot[i];

    // dx/dt = v
    for(size_t i = 0; i < n; ++i)
        dydt[n + i] = theta_ddot[i];


}

struct ParsedData {
    std::vector<double> times;
    std::vector<std::vector<double>> accel;
    std::vector<std::vector<double>> vel;
    std::vector<std::vector<double>> pos;
};


std::vector<double> generalized_coordinates(size_t n,double t,double dt) {

    std::vector<double> state = {0,0,0,0.99,-pi/13,1.1/4}; // [v1,v2,v3, x1,x2,x3], all zero init

    double t0 = 0.0;
    // Single vector to store (time, accelerations, velocities, positions)
    std::vector<double> results;

    // Observer lambda
    auto obs = [&](const std::vector<double> &y, double t) {
        // y[0..2] = velocities, y[3..5] = positions
        // Compute current accelerations:
        std::vector<double> a = accel(t);

        // Push time:
        results.push_back(t);

        // Push accelerations:
        for (size_t i = 0; i < n; i++)
            results.push_back(a[i]);

        // Push velocities:
        for (size_t i = 0; i < n; i++)
            results.push_back(y[i]);

        // Push positions:
        for (size_t i = 0; i < n; i++)
            results.push_back(y[n + i]);
    };

    runge_kutta_cash_karp54<std::vector<double>> stepper;
    integrate_const(stepper, system_of_odes_with_accel_input, state, t0, t, dt, obs);
    return results;
}


//same story
ParsedData parseResults(const std::vector<double>& results)
{
    // Here, we assume n=3. If you'd like to infer n, you'd need additional logic.
    const size_t n = 3;
    const size_t row_size = 1 + 3*n; // 1 time + 3 for accel + 3 for vel + 3 for pos = 10 total
    const size_t steps = results.size() / row_size;

    ParsedData data;
    data.times.resize(steps);
    data.accel.assign(n, std::vector<double>(steps));
    data.vel.assign(n, std::vector<double>(steps));
    data.pos.assign(n, std::vector<double>(steps));

    for (size_t s = 0; s < steps; s++)
    {
        size_t offset = s * row_size;
        data.times[s] = results[offset];

        // accelerations
        for (size_t i = 0; i < n; i++)
            data.accel[i][s] = results[offset + 1 + i];

        // velocities
        for (size_t i = 0; i < n; i++)
            data.vel[i][s] = results[offset + 1 + n + i];

        // positions
        for (size_t i = 0; i < n; i++)
            data.pos[i][s] = results[offset + 1 + 2*n + i];
    }

    return data;
}

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

void compute_body_forward_dynamics(ParsedData parsed_thetas,int n, int t, double dt)
{
	double mass = 8, l=2,  b=0.1,  h=0.2;
	float g = 9.81f;
	//hingemaps of the system
	arma::mat hinge_map_1 = {0,0,1,0,0,0}; 
	arma::mat hinge_map_1_transpose = hinge_map_1.t();

	//example positional vector - should be implemented in a function and stored with the body object
	arma::vec position_p = {0,-l/2,0};
	
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
    accel(4, n) = g;  // Equivalent to MATLAB's `accel(5,n+1) = -g;`
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

		inertia_matrix = inertia_matrix_rect_z(mass,l,b,h,position_p);
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

	plot_velocities(store_velocities, parsed_thetas);
	plot_accelerations(store_accelerations,parsed_thetas);
	plot_forces(store_forces,parsed_thetas);
	//plot_generalized_forces(store_generalized_forces,parsed_thetas);
	plot_thetas(parsed_thetas);


}

void solve_dynamics(int n, double t, double dt) {

	double mass = 8, l=2,  b=0.1,  h=0.2;
	float g = 9.81f;

	//example positional vector - should be implemented in a function and stored with the body object
	arma::vec position_p = {0,-l/2,0};

	//initial state of system
	std::vector<double> state = {pi/4,-pi/4,pi/4,0,0,0};

    double t0 = 0.0;
    // Single vector to store (time, accelerations, velocities, positions)

    std::vector<double> results;
    //std::vector<std::vector<double>> results;

    // Observer lambda
 auto obs = [&](const std::vector<double>& y, double t) {
    // Create a temporary vector to hold the derivative.
    std::vector<double> dydt(y.size(), 0.0);
    
    // Compute the derivative at the current state and time.
    system_of_equations_forward_dynamics(y, dydt, t);
    
    // Now, store the data in the order expected by ParsedData:
    // 1. Time (1 element)
    results.push_back(t);
    
    // 2. Accelerations (3 elements) — from the second half of dydt.
    for (size_t i = 3; i < 6; i++) {
         results.push_back(dydt[i]);
    }
    
    // 3. Velocities (3 elements) — from the second half of the state.
    for (size_t i = 3; i < 6; i++) {
         results.push_back(y[i]);
    }
    
    // 4. Positions (3 elements) — from the first half of the state.
    for (size_t i = 0; i < 3; i++) {
         results.push_back(y[i]);
    }
};


    runge_kutta_cash_karp54<std::vector<double>> stepper;
    integrate_const(stepper, system_of_equations_forward_dynamics, state, t0, t, dt, obs);
    
    std::cout << results.size() << std::endl;

    ParsedData results2 = parseResults(results);

    compute_body_forward_dynamics(results2,n,t,dt);

}

void solve_inverse_dynamics(int n, int t, double dt) {

	double mass = 8, l=2,  b=0.1,  h=0.2;
	float g = 9.81f;
	//hingemaps of the system
	arma::mat hinge_map_1 = {0,0,1,0,0,0}; 
	arma::mat hinge_map_1_transpose = hinge_map_1.t();
	
	std::vector<double> thetas = generalized_coordinates(n,t,dt);

	//not very happy with this implementation but will stay for now
	ParsedData parsed_thetas = parseResults(thetas);

	//example positional vector - should be implemented in a function and stored with the body object
	arma::vec position_p = {0,-l/2,0};
	
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

	plot_velocities(store_velocities, parsed_thetas);
	plot_accelerations(store_accelerations,parsed_thetas);
	plot_forces(store_forces,parsed_thetas);
	plot_generalized_forces(store_generalized_forces,parsed_thetas);
	plot_thetas(parsed_thetas);
}

int main()
{
	
	//number of system bodies
	int n = 3;

	//time to simulate until
	double t = 4;

	//time step
	double dt = 0.01;
	
	solve_dynamics(n,t,dt);
	
	return 0;
}