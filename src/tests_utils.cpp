//
// Created by Jonathan on 10/7/2025.
//

#include <boost/math/constants/constants.hpp>
#include "tests_utils.h"
#include "system_of_bodies.h"
#include <raylib.h>

using namespace boost::math::float_constants;

void test_single_body() {

    auto rec_1 = std::make_unique<Rectangle_computed>(2.0, 0.1, 0.2, 8.0);
    arma::vec vec=  {0, -rec_1->l/2.0, 0};
    rec_1->set_position_vec_hinge(vec);
    rec_1->set_outboard_position_vec_hinge({0, rec_1->l/2.0, 0});
    rec_1->set_hinge_state({pi/2,0});
    rec_1->compute_inertia_matrix();

    SystemOfBodies system;

    system.create_body(std::move(rec_1));
    system.solve_forward_dynamics();
}
/*
void test_single_body_multi_dof() {

    auto rec_1 = std::make_unique<Rectangle_computed>(2.0, 0.1, 0.2, 8.0);
    arma::vec vec=  {0, -rec_1->l/2.0, 0};
    rec_1->set_position_vec_hinge(vec);
    rec_1->set_outboard_position_vec_hinge({0, rec_1->l/2.0, 0});
    arma::mat xx = {{1,0,0,0,0,0},{0,0,1,0,0,0},{0,1,0,0,0,0}};
    rec_1->set_hinge_map(xx);
    rec_1->set_hinge_state({pi/2,0,pi/2,0,0,0});
    rec_1->compute_inertia_matrix();

    SystemOfBodies system;

    system.create_body(std::move(rec_1));
    system.set_stepper_type(true);
    system.solve_forward_dynamics();
}
*/
void test_n_body_system(const int n) {
    SystemOfBodies system;
    std::array<double,2> array = {pi/4, -pi/4};
    for (int i = 0; i < n; ++i) {
        auto rec = std::make_unique<Rectangle_computed>(2.0, 0.1, 0.2, 8.0);
        arma::vec vec=  {0, -rec->l/2.0, 0};
        rec->set_position_vec_hinge(vec);
        rec->set_outboard_position_vec_hinge({0, rec->l/2.0, 0});
        rec->set_hinge_state({array[i%2], 0});
        rec->compute_inertia_matrix();

        system.create_body(std::move(rec));
    }
    system.set_stepper_type(true);
    system.solve_forward_dynamics();

}

void test_three_body_from_course() {
    std::array<double, 3> array = {pi/4, -pi/4, pi/4};
    SystemOfBodies system;
    for (int i = 0; i < 3; ++i) {
        auto rec = std::make_unique<Rectangle_computed>(2.0, 0.1, 0.2, 8.0);
        arma::vec vec=  {0, -rec->l/2.0, 0};
        rec->set_position_vec_hinge(vec);
        rec->set_outboard_position_vec_hinge({0, rec->l/2.0, 0});
        rec->set_hinge_state({array[i], 0});
        rec->compute_inertia_matrix();


        system.create_body(std::move(rec));
    }
    system.set_stepper_type(true);
    system.solve_forward_dynamics();
}

void test_inverse_dynamics_three_body_from_course() {

    //std::array<double, 3> array = {pi/4, -pi/4, pi/4};
    SystemOfBodies system;
    for (int i = 0; i < 3; ++i) {
        auto rec = std::make_unique<Rectangle_computed>(2.0, 0.1, 0.2, 8.0);
        arma::vec vec=  {0, -rec->l/2.0, 0};
        rec->set_position_vec_hinge(vec);
        rec->set_outboard_position_vec_hinge(-vec);
        rec->compute_inertia_matrix();
        rec->set_functions_for_inverse_dyn({"(pi/4)*sin(2*pi*t+(pi/2))","((pi^2)/2)*cos(2*pi*t+(pi/2))","-(pi^3)*sin(2*pi*t+(pi/2))"});

        system.create_body(std::move(rec));
    }
    //system.solve_inverse_dynamics();
}

void test_dense() {
    std::array<double, 3> array = {pi/4, -pi/4, pi/4};
    SystemOfBodies system;
    for (int i = 0; i < 3; ++i) {
        auto rec = std::make_unique<Rectangle_computed>(2.0, 0.1, 0.2, 8.0);
        arma::vec vec=  {0, -rec->l/2.0, 0};
        rec->set_position_vec_hinge(vec);
        rec->set_outboard_position_vec_hinge({0, rec->l/2.0, 0});
        rec->set_hinge_state({array[i], 0});
        rec->compute_inertia_matrix();

        system.create_body(std::move(rec));
    }
    system.set_stepper_type(true);
    system.solve_forward_dynamics();
}


void test_three_body_from_course_with_viz() {
    std::array<double, 3> array = {pi/4, -pi/4, pi/4};
    SystemOfBodies system;
    for (int i = 0; i < 3; ++i) {
        auto rec = std::make_unique<Rectangle_computed>(2.0, 0.1, 0.2, 8.0);
        arma::vec vec = {0, -rec->l/2.0, 0};
        rec->set_position_vec_hinge(vec);
        rec->set_outboard_position_vec_hinge({0, rec->l/2.0, 0});
        rec->set_hinge_state({array[i], 0});
        rec->compute_inertia_matrix();

        system.create_body(std::move(rec));
    }
    system.set_stepper_type(true);

}

void test_raylib() {


    std::array<double, 3> array = {pi/4, -pi/4, pi/4};
    SystemOfBodies system;
    for (int i = 0; i < 3; ++i) {
        auto rec = std::make_unique<Rectangle_computed>(2.0, 0.1, 0.2, 8.0);
        arma::vec vec = {0, -rec->l/2.0, 0};
        rec->set_position_vec_hinge(vec);
        rec->set_outboard_position_vec_hinge({0, rec->l/2.0, 0});
        rec->set_hinge_state({array[i], 0});
        rec->compute_inertia_matrix();

        system.create_body(std::move(rec));
    }
    system.set_stepper_type(true);
    // Initialization
    //--------------------------------------------------------------------------------------
    const int screenWidth = 800;
    const int screenHeight = 450;

    InitWindow(screenWidth, screenHeight, "raylib [models] example - geometric shapes");

    // Define the camera to look into our 3d world
    Camera camera = { 0 };
    camera.position = (Vector3){ 0.0f, 10.0f, 10.0f };
    camera.target = (Vector3){ 0.0f, 0.0f, 0.0f };
    camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    SetTargetFPS(60);               // Set our game to run at 60 frames-per-second
    //--------------------------------------------------------------------------------------
    while (!WindowShouldClose())    // Detect window close button or ESC key
    {
        // Update
        //----------------------------------------------------------------------------------
        // TODO: Update your variables here
        //----------------------------------------------------------------------------------

        // Draw
        //----------------------------------------------------------------------------------
        BeginDrawing();

            ClearBackground(RAYWHITE);

            BeginMode3D(camera);

                DrawCube((Vector3){0.0f, 0.0f, 0.0f}, 0.1f, 0.2f, 2.0f, RED);

                DrawGrid(10, 1.0f);        // Draw a grid

            EndMode3D();

            DrawFPS(10, 10);

        EndDrawing();
        //----------------------------------------------------------------------------------
    }

    // De-Initialization
    //--------------------------------------------------------------------------------------
    CloseWindow();        // Close window and OpenGL context
    //--------------------------------------------------------------------------------------

}