//
// Created by Jonathan on 28/09/2025.
//

#pragma once
#include <armadillo>
#include <string>

class Body {
public:


    // hinge types
    arma::mat hinge_map = {0,0,1,0,0,0}; // Defualt hingemap
    arma::mat transpose_hinge_map = hinge_map.t();
    std::string hinge_type;
    bool is_dependent_hinge_map; //is used for 2 DOF hinge

    // own hinge position, vector position(s) of children
    arma::vec hinge_pos; //6X1 position vector used for inertia computations - not used in BWA
    arma::vec out_hinge_pos; //6X1 position vector used for transform from center of mass

    // Graph attributes
    int body_ID;
    std::vector<int> children_ID;
    int parent_ID;

    //arma::vec rotation_vec hinge; //should maybe be used the general formulation of the program
    arma::vec state = {0,0}; // Initial conditions of body,uses generalized position/velocity. structure - state = {p_1,p_2,...,p_ndof,v_1,v_2,...,v_ndof)
    arma::mat inertial_matrix; // Inertial matrix
    std::vector<std::string> inverse_dynamics_funcs; //saved in pairs of 3, for each dof of the body from left to right.
    double m; //mass

    Body(double mass);
    virtual void compute_inertia_matrix() = 0;  // Pure virtual

    // Setting methods
    void set_position_vec_hinge(arma::vec &input_vec);
    void set_outboard_position_vec_hinge(arma::vec input_vec);
    void set_hinge_map(arma::mat hinge_map_input);
    void set_hinge_state(arma::vec hinge_state_input);
    void set_functions_for_inverse_dyn(const std::vector<std::string> &funcs);
    void set_position_for_inverse_dyn(std::string func);
    void set_velocity_for_inverse_dyn(std::string func);
    void set_acceleration_for_inverse_dyn(std::string func);
};

class Rectangle_computed : public Body {
public:
    double l, b, h;
    arma::vec transform_vector;

    Rectangle_computed(double length, double breadth, double height, double mass);
    void compute_inertia_matrix() override;
};

class Sphere : public Body {
public:
    double r;
    // Add constructor and compute_inertia_matrix when you implement it
};