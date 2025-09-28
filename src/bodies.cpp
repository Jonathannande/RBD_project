//
// Created by Jonathan on 28/09/2025.
//

#include "bodies.h"
#include "math_utils.h"
#include <cmath>

// Body class implementations
Body::Body(double mass) : m(mass) {
    hinge_map = {0,0,1,0,0,0};
    transpose_hinge_map = hinge_map.t();
    state = {0,0};
}

void Body::set_postion_vec_hinge(arma::vec input_vec) {
    position_vec_hinge = input_vec;
    position_vec_hinge_big = join_vert(arma::zeros(3,1), input_vec);
}

void Body::set_outboard_position_vec_hinge(arma::vec input_vec) {
    outboard_position_vec_hinge = input_vec;
    outboard_position_vec_hinge_big = join_vert(arma::zeros(3,1), input_vec);
}

void Body::set_hinge_map(arma::mat hinge_map_input) {
    hinge_map = hinge_map_input;
    transpose_hinge_map = hinge_map.t();
    state = arma::zeros(2*hinge_map_input.n_cols);
}

void Body::set_hinge_state(arma::vec hinge_state_input) {
    if (2*hinge_map.n_rows == hinge_state_input.n_rows) {
        state = hinge_state_input;
    }
}

// Rectangle class implementations
Rectangle::Rectangle(double length, double breadth, double height, double mass)
    : Body(mass), l(length), b(breadth), h(height) {
    transform_vector = {0,0,0,l,0,0};
}

void Rectangle::compute_inertia_matrix() {
    // l,b, and h refer to the dimensions of length, breadth, and height
    // inertial matrix
    arma::mat inertial_matrix_rectangle = {
        {m*(1.0/12.0)*(std::pow(b,2)+std::pow(l,2)), 0, 0},
        {0, m*(1.0/12.0)*(std::pow(b,2)+std::pow(h,2)), 0},
        {0, 0, m*(1.0/12.0)*(std::pow(h,2)+std::pow(l,2))}
    };

    // parallel axis theorem
    arma::mat position_vec_tilde = tilde(-position_vec_hinge);
    arma::mat I_corner = inertial_matrix_rectangle + m*(position_vec_tilde.t()*position_vec_tilde);
    arma::mat I = arma::eye(3,3);
    arma::mat position_vec_tilde_non_neg = tilde(position_vec_hinge);

    // join the matrices and construct the inertia matrix
    arma::mat upper = join_horiz(I_corner, m*position_vec_tilde_non_neg);
    arma::mat lower = join_horiz(m*position_vec_tilde, m*I);

    inertial_matrix = join_vert(upper, lower);
}