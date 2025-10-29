//
// Created by Jonathan on 10/1/2025.
//

#ifndef MYPROJECT_SYSTEM_OF_BODIES_H
#define MYPROJECT_SYSTEM_OF_BODIES_H

#include <armadillo>
#include <vector>
#include <memory>
#include "bodies.h"
#include "data_types.h"

class SystemOfBodies {

    // Simulation attributes
    const double t0 {0.0};
    const double t {4.0};
    const double dt {0.01};
    const double system_gravity {9.81};
    int n {0};
    int system_total_dof = {0};
    bool has_dynamic_time_step{false};

    // System attributes
    std::vector<int> system_dofs_distribution;
    std::vector<double> system_equations;
public:
    std::vector<std::unique_ptr<Body>> bodies;
private:
    std::vector<double> system_state;
    arma::mat system_hinge;
    arma::mat graph_matrix;
    bool is_cannonical{false};

    // Span attributes (might get changed)
    std::vector<arma::span> span_k1_;
    std::vector<arma::span> span_k2_;
    bool spans_initialized{false};

    // Sorward dynamic specific attributes
    struct forward_parameters{
        arma::mat P_plus;
        arma::mat J_fractal_plus;
        arma::mat tau_bar;
        arma::mat P;
        arma::mat J_fractal;
        arma::mat accel;
        arma::mat accel_plus;
        arma::mat body_velocities;
        arma::mat body_forces;
        std::vector<arma::mat> G_fractal;
        std::vector<arma::mat> frac_v;
        arma::mat D;
        arma::mat eta;
        std::vector<double> dydt_out;
        int hidden_index = 0;

        forward_parameters(int n_,int system_total_dof_):
            P_plus(arma::zeros(6*(n_+1), 6*(n_+1))),
            J_fractal_plus(arma::zeros(6, n_+1)),
            tau_bar(arma::zeros(6*n_, 6*(n_+1))),
            P(arma::zeros(6*n_, 6*(n_+1))),
            J_fractal(arma::zeros(6, n_+1)),
            accel(arma::zeros(6, n_+1)),
            accel_plus(arma::zeros(6, n_+1)),
            body_velocities(arma::zeros(6, n_+1)),
            body_forces(arma::zeros(6, n_+1)),
            G_fractal(n_),
            frac_v(n_),
            dydt_out(system_total_dof_*2, 0.0)

        {}
    };


    // Sorward dynamic specific attributes
    struct forward_parameters_2{
        std::vector<arma::mat::fixed<6,6>> P_plus;
        std::vector<arma::vec::fixed<6>> J_fractal_plus;
        std::vector<arma::vec::fixed<6>> J_fractal;
        std::vector<arma::mat::fixed<6,6>> tau_bar;
        std::vector<arma::mat::fixed<6,6>> P;
        arma::mat::fixed<6,6> accel;
        arma::mat::fixed<6,6> accel_plus;
        std::vector<arma::vec::fixed<6>> body_velocities;
        std::vector<arma::vec::fixed<6>> body_forces;
        std::vector<arma::vec::fixed<6>> G_fractal;
        std::vector<arma::mat::fixed<6,6>> frac_v;
        arma::mat D;
        arma::vec::fixed<6> eta;
        std::vector<double> dydt_out;
        int hidden_index = 0;

        forward_parameters_2(const int n_,const int system_total_dof_):
            P_plus(n_+1),
            J_fractal_plus(n_+1),
            tau_bar(n_),
            P(n_),
            J_fractal(n_),
            body_velocities(n_+1),
            body_forces(n_+1),
            G_fractal(n_+1),
            frac_v(n_),
            dydt_out(system_total_dof_*2, 0.0)

        {}
    };




public:
    // Methods


    // Formatting methods
    ParsedData parseResults(const std::vector<double>& results) const;

    std::vector<arma::vec> to_arma_vec(const std::vector<double>& DoFs) const; // used for conversion of state types

    std::vector<double> to_std_vec(const std::vector<arma::vec>& DoFs); // used for conversion of state types


    // System structure
    void update_system_state();

    void create_body(std::unique_ptr<Body> any_body);

    void set_BWA();

    void set_system_structure();


    // System/body transform methods
    arma::mat find_spatial_operator_input_vector(const std::vector<arma::vec>& DoFs) const;

    arma::mat find_gravity_matrix_pseudo_acceleration();

    arma::mat find_spatial_operator_input_vector(std::vector<double> po);

    arma::mat find_spatial_operator(const arma::mat& rigid_body_transform_vector); //just a placeholder for inverse dynamics

    std::vector<arma::mat::fixed<6,6>> find_spatial_operator_2(const std::vector<arma::vec>& state) const;

    // run simulation methods
    void solve_inverse_dynamics();

    void solve_forward_dynamics();


    // Forward dynamics specific
    void system_of_equations_forward_dynamics_2(const std::vector<double> &y, std::vector<double> &dydt, forward_parameters_2 &p);

    void system_of_equations_forward_dynamics(const std::vector<double> &y, std::vector<double> &dydt, forward_parameters& p);

    void set_stepper_type(const bool& is_dynamic);


    // Inverse dynamics specific
    ParsedData inverse_run_funcs();

    std::vector<double> inverse_run_funcs_2();



};

#endif //MYPROJECT_SYSTEM_OF_BODIES_H