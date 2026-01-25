//
// Created by Jonathan on 10/7/2025.
//

#include "tests_utils.h"
#include "system_of_bodies.h"
#include <boost/math/constants/constants.hpp>
#include <chrono>
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
  system.prep_system();
  system.solve_forward_dynamics();
}

void test_spherical_hinge() {

  auto rec_1 = std::make_unique<Rectangle_computed>(2.0, 0.1, 0.2, 8.0);
  arma::vec vec = {0, -rec_1->l / 2.0, 0};
  rec_1->set_position_vec_hinge(vec);
  rec_1->set_hinge_map("spherical");
  rec_1->set_hinge_state({pi / 2, pi / 2, pi / 2, 0, 0, 0});
  rec_1->compute_inertia_matrix();

  SystemOfBodies system;

  system.create_body(std::move(rec_1));

  system.bodies[0]->set_outboard_position_vec_hinge_push(vec);
  system.prep_system();
  system.set_stepper_type(true);
  system.solve_forward_dynamics_tree();
}

void test_universal_hinge() {

  auto rec_1 = std::make_unique<Rectangle_computed>(2.0, 0.1, 0.2, 8.0);
  arma::vec vec = {0, -rec_1->l / 2.0, 0};
  rec_1->set_position_vec_hinge(vec);
  rec_1->set_hinge_map("universal");
  rec_1->set_hinge_state({pi / 6, -pi / 6, 0, 0});
  rec_1->compute_inertia_matrix();

  SystemOfBodies system;

  system.create_body(std::move(rec_1));

  system.bodies[0]->set_outboard_position_vec_hinge_push(vec);
  system.prep_system();
  system.set_stepper_type(false);
  system.solve_forward_dynamics_tree();
}

void test_n_body_system(const int n) {
  SystemOfBodies system;
  std::array<double, 2> array = {pi / 2, -pi / 2};
  std::array<double, 2> array_2 = {1.0, 2.0};
  auto rec = std::make_unique<Rectangle_computed>(2, 0.1, 0.2, 8.0);
  arma::vec vec = {0, -rec->l / 2.0, 0};
  rec->set_position_vec_hinge(vec);
  rec->set_outboard_position_vec_hinge({0, rec->l / 2.0, 0});
  rec->set_hinge_state({pi / 4, 0});
  rec->compute_inertia_matrix();

  system.create_body(std::move(rec));

  for (int i = 1; i < n; ++i) {
    auto rec = std::make_unique<Rectangle_computed>(2, 0.1, 0.2, 8.0);
    arma::vec vec = {0, -rec->l / 2.0, 0};
    rec->set_position_vec_hinge(vec);
    rec->set_outboard_position_vec_hinge({0, rec->l / 2.0, 0});
    rec->set_hinge_state({array[i % 2], 0});
    rec->compute_inertia_matrix();

    system.create_body(std::move(rec));
  }

  system.prep_system();
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

    system.prep_system();
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

  system.prep_system();
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

  system.prep_system();
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

  auto start = std::chrono::high_resolution_clock::now();
  std::array<double, 5> array_hinge_state = {pi / 6, -pi / 5, pi / 7, pi / 2,
                                             pi / 3};
  array_hinge_state = {pi / 4, pi / 4, 0, pi / 4, pi / 2};
  auto rec = Rectangle_computed(2.0, 0.1, 0.2, 8.0);

  std::array<arma::vec, 5> array_out_vec = {{{0, rec.l / 2, 0},
                                             {0, rec.l / 2, 0},
                                             {0, rec.l / 2, 0},
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
  system.bodies[0]->children_ID[0] = 0;
  system.bodies[1]->children_ID[0] = 0;

  system.set_stepper_type(true);
  system.solve_forward_dynamics_tree();

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  std::cout << "Full program run time: " << duration.count()
            << " microseconds\n";
}

// GRAPHICS NOT WORKING

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

void test_rayy() {
  InitWindow(800, 600, "Draggable Pendulum");
  SetTargetFPS(60);

  Camera3D camera = {{0, 0, 12}, {0, 0, 0}, {0, 1, 0}, 45, CAMERA_PERSPECTIVE};

  float length = 3.0f;
  Vector3 pivot = {0, 2, 0};

  // Pendulum state
  float theta = 0.5f;
  float omega = 0.0f; // angular velocity

  // Physics
  float gravity = 9.8f;
  float damping = 0.5f;

  // Dragging
  bool dragging = false;

  while (!WindowShouldClose()) {
    float dt = GetFrameTime();

    // Compute pendulum end position
    Vector3 end = {pivot.x + length * sinf(theta),
                   pivot.y - length * cosf(theta), 0};

    // Mouse ray
    Ray ray = GetScreenToWorldRay(GetMousePosition(), camera);

    // Project mouse onto z=0 plane
    float t = -ray.position.z / ray.direction.z;
    Vector3 mouseWorld = {ray.position.x + ray.direction.x * t,
                          ray.position.y + ray.direction.y * t, 0};

    // Check if clicking near the end
    if (IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) {
      float dist = Vector3Distance(mouseWorld, end);
      if (dist < 0.5f) {
        dragging = true;
      }
    }
    if (IsMouseButtonReleased(MOUSE_LEFT_BUTTON)) {
      dragging = false;
    }

    // Physics update
    if (dragging) {
      // Compute angle to mouse position
      float dx = mouseWorld.x - pivot.x;
      float dy = mouseWorld.y - pivot.y;
      float targetTheta = atan2f(dx, -dy);

      // Spring toward mouse
      float springK = 50.0f;
      float torque = springK * (targetTheta - theta);
      omega += torque * dt;
      omega *= 0.9f; // extra damping while dragging
    } else {
      // Gravity torque: -g/L * sin(theta)
      float alpha = -(gravity / length) * sinf(theta);
      omega += alpha * dt;
      omega -= damping * omega * dt;
    }

    theta += omega * dt;

    BeginDrawing();
    ClearBackground(RAYWHITE);
    BeginMode3D(camera);

    DrawSphere(pivot, 0.05f, DARKGRAY);
    DrawCylinderEx(pivot, end, 0.08f, 0.08f, 8, dragging ? BLUE : RED);
    DrawSphere(end, dragging ? 0.25f : 0.15f, dragging ? BLUE : MAROON);
    DrawGrid(10, 1.0f);

    // Draw mouse target while dragging
    if (dragging) {
      DrawSphere(mouseWorld, 0.1f, GREEN);
    }

    EndMode3D();

    DrawText("Click and drag the pendulum end", 10, 10, 20, GRAY);

    EndDrawing();
  }

  CloseWindow();
}

void test_chain_dynamics() {
  auto start = std::chrono::high_resolution_clock::now();

  const int num_bodies = 10;
  auto rec = Rectangle_computed(2.0, 0.1, 0.2, 8.0);

  SystemOfBodies system;

  // Create bodies with varying initial angles
  for (int i = 0; i < num_bodies; ++i) {
    auto body = std::make_unique<Rectangle_computed>(2.0, 0.1, 0.2, 8.0);
    if (i > 5) {

      auto body = std::make_unique<Rectangle_computed>(2.0, 0.1, 0.2, 80.0);
    }
    arma::vec vec = {0, -body->l / 2.0, 0};
    body->set_position_vec_hinge(vec);

    // Varying initial angles
    double init_angle = pi / 6.0 + (i * pi / 20.0);
    body->set_hinge_state({init_angle, 0});
    body->compute_inertia_matrix();
    system.create_body(std::move(body));
  }

  system.prep_system();

  // Chain structure: each body is parent of the next
  // Body 10 (index 9) is root
  // Body 9 is child of 10
  // Body 8 is child of 9
  // etc.
  for (int i = num_bodies - 2; i >= 0; i--) {
    system.set_parent(i + 1, i + 2, true);
  }

  // Set outboard vectors
  arma::vec out_vec = {0, rec.l / 2, 0};
  for (int i = 0; i < num_bodies; i++) {
    system.bodies[i]->set_outboard_position_vec_hinge_push(out_vec);
  }

  system.bodies[0]->children_ID[0] = 0;

  system.set_stepper_type(true);
  system.solve_forward_dynamics_tree();

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "Full program run time: " << duration.count()
            << " microseconds\n";
}
