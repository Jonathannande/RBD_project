//
// Created by Jonathan on 10/7/2025.
//
#include <boost/math/constants/constants.hpp>
#include "tests_utils.h"
#include "system_of_bodies.h"
#include <array>

using namespace boost::math::float_constants;

void test_single_body() {

    auto rec_1 = std::make_unique<Rectangle>(2.0, 0.1, 0.2, 8.0);
    rec_1->set_postion_vec_hinge({0, -rec_1->l/2.0, 0});
    rec_1->set_outboard_position_vec_hinge({0, rec_1->l/2.0, 0});
    rec_1->set_hinge_state({pi/2,0});
    rec_1->compute_inertia_matrix();

    SystemOfBodies system;

    system.create_body(std::move(rec_1));
    system.solve_forward_dynamics();
}

void test_n_body_system(int n) {
    SystemOfBodies system;
    for (int i = 0; i < n; ++i) {
        auto rec = std::make_unique<Rectangle>(2.0, 0.1, 0.2, 8.0);
        rec->set_postion_vec_hinge({0, -rec->l/2.0, 0});
        rec->set_outboard_position_vec_hinge({0, rec->l/2.0, 0});
        rec->set_hinge_state({pi/2, 0});
        rec->compute_inertia_matrix();

        system.create_body(std::move(rec));
    }
    system.solve_forward_dynamics();

}

void test_three_body_from_course() {
    std::array<double, 3> array = {pi/4, -pi/4, pi/4};
    SystemOfBodies system;
    for (int i = 0; i < 3; ++i) {
        auto rec = std::make_unique<Rectangle>(2.0, 0.1, 0.2, 8.0);
        rec->set_postion_vec_hinge({0, -rec->l/2.0, 0});
        rec->set_outboard_position_vec_hinge({0, rec->l/2.0, 0});
        rec->set_hinge_state({array[i], 0});
        rec->compute_inertia_matrix();

        system.create_body(std::move(rec));
    }
    system.solve_forward_dynamics();
}