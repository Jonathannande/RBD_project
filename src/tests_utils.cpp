//
// Created by Jonathan on 10/7/2025.
//

#include "tests_utils.h"
#include "system_of_bodies.h"
#include <boost/math/constants/constants.hpp>
#include <raylib.h>
#include <raymath.h>
#include <rlgl.h>

using namespace boost::math::float_constants;

void test_single_body() {

  auto rec_1 = std::make_unique<Rectangle_computed>(2.0, 0.1, 0.2, 8.0);
  arma::vec vec = {0, -rec_1->l / 2.0, 0};
  rec_1->set_position_vec_hinge(vec);
  rec_1->set_outboard_position_vec_hinge({0, rec_1->l / 2.0, 0});
  rec_1->set_hinge_state({pi / 2, 0});
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
  std::array<double, 2> array = {pi / 4, -pi / 4};
  std::array<double, 2> array_2 = {1.0, 2.0};
  for (int i = 0; i < n; ++i) {
    auto rec =
        std::make_unique<Rectangle_computed>(array[i % 2], 0.1, 0.2, 8.0);
    arma::vec vec = {0, -rec->l / 2.0, 0};
    rec->set_position_vec_hinge(vec);
    rec->set_outboard_position_vec_hinge({0, rec->l / 2.0, 0});
    rec->set_hinge_state({array[i % 2], 0});
    rec->compute_inertia_matrix();

    system.create_body(std::move(rec));
  }
  system.set_stepper_type(true);
  system.solve_forward_dynamics();
}

void test_three_body_from_course() {
  std::array<double, 3> array = {pi / 4, -pi / 4, pi / 4};
  SystemOfBodies system;
  for (int i = 0; i < 3; ++i) {
    auto rec = std::make_unique<Rectangle_computed>(2.0, 0.1, 0.2, 8.0);
    arma::vec vec = {0, -rec->l / 2.0, 0};
    rec->set_position_vec_hinge(vec);
    rec->set_outboard_position_vec_hinge({0, rec->l / 2.0, 0});
    rec->set_hinge_state({array[i], 0});
    rec->compute_inertia_matrix();

    system.create_body(std::move(rec));
  }
  system.prep_system();
  system.set_stepper_type(true);
  system.solve_forward_dynamics();
}

void test_inverse_dynamics_three_body_from_course() {

  // std::array<double, 3> array = {pi/4, -pi/4, pi/4};
  SystemOfBodies system;
  for (int i = 0; i < 3; ++i) {
    auto rec = std::make_unique<Rectangle_computed>(2.0, 0.1, 0.2, 8.0);
    arma::vec vec = {0, -rec->l / 2.0, 0};
    rec->set_position_vec_hinge(vec);
    rec->set_outboard_position_vec_hinge(-vec);
    rec->compute_inertia_matrix();
    rec->set_functions_for_inverse_dyn({"(pi/4)*sin(2*pi*t+(pi/2))",
                                        "((pi^2)/2)*cos(2*pi*t+(pi/2))",
                                        "-(pi^3)*sin(2*pi*t+(pi/2))"});

    system.create_body(std::move(rec));
  }
  // system.solve_inverse_dynamics();
}

void test_dense() {
  std::array<double, 3> array = {pi / 4, -pi / 4, pi / 4};
  SystemOfBodies system;
  for (int i = 0; i < 3; ++i) {
    auto rec = std::make_unique<Rectangle_computed>(2.0, 0.1, 0.2, 8.0);
    arma::vec vec = {0, -rec->l / 2.0, 0};
    rec->set_position_vec_hinge(vec);
    rec->set_outboard_position_vec_hinge({0, rec->l / 2.0, 0});
    rec->set_hinge_state({array[i], 0});
    rec->compute_inertia_matrix();

    system.create_body(std::move(rec));
  }
  system.set_stepper_type(true);
  system.solve_forward_dynamics();
}

void test_three_body_from_course_with_viz() {
  std::array<double, 3> array = {pi / 4, -pi / 4, pi / 4};
  SystemOfBodies system;
  for (int i = 0; i < 3; ++i) {
    auto rec = std::make_unique<Rectangle_computed>(2.0, 0.1, 0.2, 8.0);
    arma::vec vec = {0, -rec->l / 2.0, 0};
    rec->set_position_vec_hinge(vec);
    rec->set_outboard_position_vec_hinge({0, rec->l / 2.0, 0});
    rec->set_hinge_state({array[i], 0});
    rec->compute_inertia_matrix();

    system.create_body(std::move(rec));
  }
  system.set_stepper_type(false);
}
// Rotate a point around a pivot
Vector2 RotatePointAroundPivot(Vector2 point, Vector2 pivot, float angle) {
  float s = sinf(angle);
  float c = cosf(angle);

  Vector2 translated = {point.x - pivot.x, point.y - pivot.y};
  Vector2 rotated = {translated.x * c - translated.y * s,
                     translated.x * s + translated.y * c};

  return (Vector2){rotated.x + pivot.x, rotated.y + pivot.y};
}

void test_tree_dynamics() {

  std::array<double, 5> array_hinge_state = {0, -pi / 4, pi / 4, pi / 2, 0};
  auto rec = Rectangle_computed(2.0, 0.1, 0.2, 8.0);

  std::array<arma::vec, 5> array_out_vec = {{{0, rec.l / 2, 0},
                                             {0.05, rec.l / 2, 0},
                                             {-0.05, rec.l / 2, 0},
                                             {0, rec.l / 2, 0},
                                             {0, rec.l / 2, 0}}};
  //    std::cout<<array_out_vec<<std::endl;
  SystemOfBodies system;
  for (int i = 0; i < 5; ++i) {
    auto rec = std::make_unique<Rectangle_computed>(2.0, 0.1, 0.2, 8.0);
    arma::vec vec = {0, -rec->l / 2.0, 0};
    rec->set_position_vec_hinge(vec);
    rec->set_hinge_state({array_hinge_state[i], 0});
    rec->compute_inertia_matrix();
    system.create_body(std::move(rec));
  }

  system.prep_system();
  system.set_parent(4, 5, true);
  system.set_parent(3, 5, false);
  system.set_parent(2, 4, true);
  system.set_parent(1, 3, true);

  system.bodies[4]->set_outboard_position_vec_hinge_push(array_out_vec[1]);
  system.bodies[4]->set_outboard_position_vec_hinge_push(array_out_vec[2]);
  system.bodies[3]->set_outboard_position_vec_hinge_push(array_out_vec[0]);
  system.bodies[2]->set_outboard_position_vec_hinge_push(array_out_vec[0]);
  system.bodies[1]->set_outboard_position_vec_hinge_push(array_out_vec[0]);
  system.bodies[0]->set_outboard_position_vec_hinge_push(array_out_vec[0]);

  system.set_stepper_type(false);
  system.solve_forward_dynamics_tree();
}

void test_raylib() {

  std::array<double, 3> array = {pi / 4, -pi / 4, pi / 4};
  SystemOfBodies system;
  for (int i = 0; i < 3; ++i) {
    auto rec = std::make_unique<Rectangle_computed>(2, 0.1, 0.2, 8.0);
    arma::vec vec = {0, rec->l / 2.0, 0};
    rec->set_position_vec_hinge(vec);
    rec->set_outboard_position_vec_hinge(-vec);
    rec->set_hinge_state({array[i], 0});
    rec->compute_inertia_matrix();

    system.create_body(std::move(rec));
  }
  system.set_stepper_type(true);
  system.solve_forward_dynamics();

  InitWindow(1200, 800, "Pendulum Chain");

  Camera3D camera = {0};
  camera.position = (Vector3){10.0f, 5.0f, 15.0f};
  camera.target = (Vector3){0.0f, 0.0f, 0.0f};
  camera.up = (Vector3){0.0f, 1.0f, 0.0f};
  camera.fovy = 45.0f;
  camera.projection = CAMERA_PERSPECTIVE;

  SetTargetFPS(60);

  int n = system.n;    // number of pendulums
  float length = 2.0f; // rectangle length

  // ParsedData data = system.parseResults(results);
  int currentStep = 0;
  bool playing = false;

  while (!WindowShouldClose()) {
    UpdateCamera(&camera, CAMERA_FREE);

    if (IsKeyPressed(KEY_SPACE))
      playing = !playing;
    if (playing)
      currentStep++;
    if (IsKeyPressed(KEY_RIGHT))
      currentStep++;
    if (IsKeyPressed(KEY_LEFT) && currentStep > 0)
      currentStep--;

    BeginDrawing();
    ClearBackground(RAYWHITE);
    BeginMode3D(camera);

    DrawGrid(20, 1.0f);

    Vector3 currentPivot = {0, 0, 0}; // First pivot at origin

    for (int i = 0; i < n; i++) {
      std::cout << "n: " << n
                << " pos.size(): " << system.parsed_data.pos.size()
                << " steps: " << system.parsed_data.pos[0].size()
                << " currentStep: " << currentStep << std::endl;
      // double angle = data.pos[i][currentStep];
      double angle = system.parsed_data.pos[i][currentStep];

      // Draw rectangle - rotate about the pivot, not center
      rlPushMatrix();
      rlTranslatef(currentPivot.x, currentPivot.y,
                   currentPivot.z);        // Move to pivot
      rlRotatef(angle * RAD2DEG, 1, 0, 0); // Rotate about pivot
      DrawCube((Vector3){0, -length / 2, 0}, 0.2f, length, 0.2f,
               BLUE); // Draw offset from pivot
      DrawCubeWires((Vector3){0, -length / 2, 0}, 0.2f, length, 0.2f, BLACK);
      rlPopMatrix();

      DrawSphere(currentPivot, 0.05f, RED);

      // Next pivot is at the end of this rectangle
      currentPivot =
          (Vector3){currentPivot.x, currentPivot.y - length * -cosf(angle),
                    currentPivot.z + length * -sinf(angle)};
    }

    EndMode3D();

    DrawText(TextFormat("Step: %d | SPACE: play/pause", currentStep), 10, 10,
             20, BLACK);

    EndDrawing();
  }

  CloseWindow();
}
