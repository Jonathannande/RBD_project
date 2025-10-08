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
public:  // ADD THIS
    int n {0};
    double t0 {0.0};
    double t {4.0};
    double dt {0.01};
    double system_gravity {9.81};
    int system_total_dof = 0;
    std::vector<int> system_dofs_distribution;
    std::vector<double> system_equations;
    std::vector<std::unique_ptr<Body>> bodies;
    std::vector<double> system_state;
    arma::mat system_hinge;
    arma::mat graph_matrix;

    // Methods
    ParsedData parseResults(const std::vector<double>& results);

    void update_system_state();

    arma::mat find_spatial_operator_input_vector(std::vector<arma::vec>& DoFs);

    std::vector<arma::vec> to_arma_vec(std::vector<double> DoFs);

    std::vector<double> to_std_vec(std::vector<arma::vec>& DoFs);

    arma::mat find_gravity_matrix_pseudo_acceleration();

    void test_method();

    void system_of_equations_forward_dynamics(const std::vector<double> &y, std::vector<double> &dydt, double t,
        arma::mat P_plus, arma::mat J_fractal_plus, arma::mat tau_bar, arma::mat P, arma::mat J_fractal,
        arma::mat accel, arma::mat accel_plus, arma::mat body_velocities, std::vector<arma::mat> G_fractal,
        std::vector<arma::mat> frac_v,arma::mat eta,arma::mat D);

    void solve_forward_dynamics();

    void solve_hybrid_dynamics();

    void get_states_forward_dynamics();

    void create_body(std::unique_ptr<Body> any_body);

    ParsedData inverse_run_funcs();

    void solve_inverse_dynamics();

};

#endif //MYPROJECT_SYSTEM_OF_BODIES_H