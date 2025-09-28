//
// Created by Jonathan on 28/09/2025.
//

#pragma once
#include <armadillo>
#include <string>
#include <memory>

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
    arma::vec state = {0,0}; //this is the initial conditions of the body, should use its generalized velocity/position of which there might be multiple
    arma::mat inertial_matrix;
    double m; //mass

    Body(double mass);
    virtual void compute_inertia_matrix() = 0;  // Pure virtual
    void set_postion_vec_hinge(arma::vec input_vec);
    void set_outboard_position_vec_hinge(arma::vec input_vec);
    void set_hinge_map(arma::mat hinge_map_input);
    void set_hinge_state(arma::vec hinge_state_input);
};

class Rectangle : public Body {
public:
    double l, b, h;
    arma::vec transform_vector;

    Rectangle(double length, double breadth, double height, double mass);
    void compute_inertia_matrix() override;
};

class Sphere : public Body {
public:
    double r;
    // Add constructor and compute_inertia_matrix when you implement it
};