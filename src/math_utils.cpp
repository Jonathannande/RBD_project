//
// Created by Jonathan on 28/09/2025.
//
#include "math_utils.h"
#include <armadillo>

arma::mat tilde(arma::mat vector) {
	arma::mat tilde_matrix = {
		{0,-vector(2),vector(1)},
		{vector(2),0,-vector(0)},
		{-vector(1),vector(0),0}
	};

	return tilde_matrix;
}


//rotation matrix using euler angles
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


// Alternative: Direct conversion without intermediate quaternion storage
arma::mat rotation_matrix_qua(const arma::mat& rotation_vec) {
    double theta = rotation_vec(0);
    double phi = rotation_vec(1);
    double gamma = rotation_vec(2);

    //half'' angles
    double cx = cos(theta * 0.5), sx = sin(theta * 0.5);
    double cy = cos(phi * 0.5),   sy = sin(phi * 0.5);
    double cz = cos(gamma * 0.5), sz = sin(gamma * 0.5);

    // Quaternion components
    double w = cx * cy * cz + sx * sy * sz;
    double x = sx * cy * cz - cx * sy * sz;
    double y = cx * sy * cz + sx * cy * sz;
    double z = cx * cy * sz - sx * sy * cz;

    // Normalize
    double norm = sqrt(w*w + x*x + y*y + z*z);
    w /= norm; x /= norm; y /= norm; z /= norm;

    // Direct matrix computation
    arma::mat R(3, 3);
    R(0,0) = 1 - 2*(y*y + z*z);  R(0,1) = 2*(x*y - w*z);      R(0,2) = 2*(x*z + w*y);
    R(1,0) = 2*(x*y + w*z);      R(1,1) = 1 - 2*(x*x + z*z);  R(1,2) = 2*(y*z - w*x);
    R(2,0) = 2*(x*z - w*y);      R(2,1) = 2*(y*z + w*x);      R(2,2) = 1 - 2*(x*x + y*y);

    return R;
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

	// assemble the upper and lower half
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

	// assemble the upper and lower half
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


	return spatial_operator;
}