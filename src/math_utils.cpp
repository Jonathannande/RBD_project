//
// Created by Jonathan on 28/09/2025.
//
#include "math_utils.h"
#include <armadillo>

arma::mat::fixed<3, 3> tilde(const arma::mat &vector) {

  arma::mat::fixed<3, 3> tilde_matrix = {{0, -vector(2), vector(1)},
                                         {vector(2), 0, -vector(0)},
                                         {-vector(1), vector(0), 0}};

  return tilde_matrix;
}

arma::mat::fixed<3, 3> tilde_fast(const arma::vec::fixed<3> &v) {
  arma::mat::fixed<3, 3> r(arma::fill::zeros);
  r(0, 1) = -v(2);
  r(0, 2) = v(1);
  r(1, 0) = v(2);
  r(1, 2) = -v(0);
  r(2, 0) = -v(1);
  r(2, 1) = v(0);
  return r;
}

// rotation matrix using euler angles
arma::mat rotation_matrix(const arma::mat &rotation_vec) {
  // basic euler

  double theta = rotation_vec(0);
  double phi = rotation_vec(1);
  double gamma = rotation_vec(2);

  arma::mat R_x = {
      {1, 0, 0},
      {0, cos(theta), -sin(theta)},
      {0, sin(theta), cos(theta)},

  };
  arma::mat R_y = {
      {cos(phi), 0, sin(phi)}, {0, 1, 0}, {-sin(phi), 0, cos(phi)}};
  arma::mat R_z = {
      {cos(gamma), -sin(gamma), 0}, {sin(gamma), cos(gamma), 0}, {0, 0, 1}};
  arma::mat R_xyz = R_x * R_y * R_z;
  return R_xyz;
}

// Alternative: Direct conversion without intermediate quaternion storage
arma::mat::fixed<3, 3> rotation_matrix_qua(const arma::mat &rotation_vec) {
  double theta = rotation_vec(0);
  double phi = rotation_vec(1);
  double gamma = rotation_vec(2);

  // half'' angles
  double cx = cos(theta * 0.5), sx = sin(theta * 0.5);
  double cy = cos(phi * 0.5), sy = sin(phi * 0.5);
  double cz = cos(gamma * 0.5), sz = sin(gamma * 0.5);

  // Quaternion components
  double w = cx * cy * cz + sx * sy * sz;
  double x = sx * cy * cz - cx * sy * sz;
  double y = cx * sy * cz + sx * cy * sz;
  double z = cx * cy * sz - sx * sy * cz;

  // Normalize
  double norm = sqrt(w * w + x * x + y * y + z * z);
  w /= norm;
  x /= norm;
  y /= norm;
  z /= norm;

  // Direct matrix computation
  arma::mat::fixed<3, 3> R;
  R(0, 0) = 1 - 2 * (y * y + z * z);
  R(0, 1) = 2 * (x * y - w * z);
  R(0, 2) = 2 * (x * z + w * y);
  R(1, 0) = 2 * (x * y + w * z);
  R(1, 1) = 1 - 2 * (x * x + z * z);
  R(1, 2) = 2 * (y * z - w * x);
  R(2, 0) = 2 * (x * z - w * y);
  R(2, 1) = 2 * (y * z + w * x);
  R(2, 2) = 1 - 2 * (x * x + y * y);

  return R;
}

arma::mat::fixed<6, 6> rb_transform(const arma::mat &rotation_vec,
                                    const arma::mat &position_vec) {
  // define rotation matrix and tilde position matrix
  const arma::mat rotations = rotation_matrix_qua(rotation_vec);
  const arma::mat position_vec_tilde = tilde(position_vec.t());

  // rotate positions to the new frame
  const arma::mat rotated_positions = position_vec_tilde * rotations;

  // join the matrices and construct the rigid body transform
  const arma::mat upper = join_horiz(rotations, rotated_positions);
  const arma::mat lower = join_horiz(arma::zeros(3, 3), rotations);

  return join_vert(upper, lower);
}

arma::mat::fixed<6, 6> rb_transform_transpose(const arma::mat &rbt) {
  return rbt.t();
}

// 6x6 matrix used in various computations
arma::mat::fixed<6, 6> tilde_velocity(const arma::mat &velocity_vector) {

  // do the tildes
  const arma::mat angular_tilde = tilde(velocity_vector.rows(0, 2));
  const arma::mat translational_tilde = tilde(velocity_vector.rows(3, 5));

  // assemble the upper and lower half
  const arma::mat upper =
      join_horiz(angular_tilde, arma::zeros<arma::mat>(3, 3));
  const arma::mat lower = join_horiz(translational_tilde, angular_tilde);

  // assemble the matrix
  return join_vert(upper, lower);
}

arma::mat::fixed<6, 6> tilde_velocity_fast(const arma::vec::fixed<6> &v) {
  arma::mat::fixed<6, 6> r(arma::fill::zeros);
  // top-left: tilde(angular)
  r(0, 1) = -v(2);
  r(0, 2) = v(1);
  r(1, 0) = v(2);
  r(1, 2) = -v(0);
  r(2, 0) = -v(1);
  r(2, 1) = v(0);
  // bottom-left: tilde(translational)
  r(3, 1) = -v(5);
  r(3, 2) = v(4);
  r(4, 0) = v(5);
  r(4, 2) = -v(3);
  r(5, 0) = -v(4);
  r(5, 1) = v(3);
  // bottom-right: tilde(angular)
  r(3, 4) = -v(2);
  r(3, 5) = v(1);
  r(4, 3) = v(2);
  r(4, 5) = -v(0);
  r(5, 3) = -v(1);
  r(5, 4) = v(0);
  return r;
}

// transpose of tilde_velocity
arma::mat::fixed<6, 6> bar_velocity(const arma::mat &velocity_vector) {
  // do the tildes
  const arma::mat angular_tilde = tilde(velocity_vector.rows(0, 2));
  const arma::mat translational_tilde = tilde(velocity_vector.rows(3, 5));

  // assemble the upper and lower half
  const arma::mat upper = join_horiz(angular_tilde, translational_tilde);
  const arma::mat lower =
      join_horiz(arma::zeros<arma::mat>(3, 3), angular_tilde);

  // assemble the matrix
  return join_vert(upper, lower);
}
// Add to your code temporarily
arma::mat::fixed<6, 6> bar_velocity_fast(const arma::vec::fixed<6> &v) {
  arma::mat::fixed<6, 6> r(arma::fill::zeros);
  r(0, 1) = -v(2);
  r(0, 2) = v(1);
  r(1, 0) = v(2);
  r(1, 2) = -v(0);
  r(2, 0) = -v(1);
  r(2, 1) = v(0);
  r(3, 4) = -v(2);
  r(3, 5) = v(1);
  r(4, 3) = v(2);
  r(4, 5) = -v(0);
  r(5, 3) = -v(1);
  r(5, 4) = v(0);
  r(0, 4) = -v(5);
  r(0, 5) = v(4);
  r(1, 3) = v(5);
  r(1, 5) = -v(3);
  r(2, 3) = -v(4);
  r(2, 4) = v(3);
  return r;
}
// gyroscopic force for body frame, eq. 2.28
arma::vec::fixed<6> gyroscopic_force_z(const arma::mat &inertial_matrix,
                                       const arma::mat &velocity_vector) {
  const arma::mat velocity_bar = bar_velocity_fast(velocity_vector);
  return velocity_bar * inertial_matrix * velocity_vector;
}
