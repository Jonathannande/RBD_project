
#include "system_of_bodies.h"
#include <armadillo>
#include <raylib.h>

void SystemOfBodies::animate_tree(const ParsedData &data) {
  InitWindow(1600, 900, "Tree Animation");
  SetTargetFPS(60);

  Camera3D camera = {{0, 0, 45}, {0, -5, 0}, {0, 1, 0}, 45, CAMERA_PERSPECTIVE};

  const double l = 2.0;
  size_t step = 0;
  size_t total_steps = data.times.size();
  float time_accum = 0.0f;

  std::vector<Vector3> start_pos(n);
  std::vector<Vector3> end_pos(n);
  std::vector<double> world_angle(n);

  Color colors[] = {RED, BLUE, GREEN, ORANGE, PURPLE, YELLOW, PINK, SKYBLUE};
  bool paused = true;

  while (!WindowShouldClose()) {
    if (IsKeyPressed(KEY_SPACE))
      paused = !paused;
    if (!paused) {
      float dt = GetFrameTime();
      time_accum += dt;

      while (step < total_steps - 1 &&
             time_accum > data.times[step + 1] - data.times[0]) {
        step++;
      }

      // Forward kinematics from root to leaves
      // Process in reverse order (root has highest index)
      for (int k = n - 1; k >= 0; k--) {
        double theta = data.pos[k][step];

        if (bodies[k]->parent_ID == -1 || k == n - 1) {
          // Root body - starts at origin
          start_pos[k] = {0, 0, 0};
          world_angle[k] = theta;
        } else {
          // Child body - starts at parent's end
          int parent = bodies[k]->parent_ID;
          start_pos[k] = end_pos[parent - 1];
          world_angle[k] = world_angle[parent - 1] + theta;
        }

        double len = dynamic_cast<Rectangle_computed *>(bodies[k].get())->l;
        end_pos[k] = {start_pos[k].x + (float)(len * sin(world_angle[k])),
                      start_pos[k].y - (float)(len * cos(world_angle[k])), 0};
      }
    }
    BeginDrawing();
    ClearBackground(RAYWHITE);
    BeginMode3D(camera);

    // Draw all bodies
    for (size_t k = 0; k < n; k++) {
      DrawCylinderEx(start_pos[k], end_pos[k], 0.08f, 0.08f, 8, colors[k % 8]);
      DrawSphere(start_pos[k], 0.08f, DARKGRAY);
      DrawSphere(end_pos[k], 0.08f, DARKGRAY);
    }

    EndMode3D();

    DrawText(TextFormat("Time: %.2f s", data.times[step]), 10, 10, 20, GRAY);
    DrawText(TextFormat("Step: %d / %d", (int)step, (int)total_steps), 10, 35,
             20, GRAY);

    EndDrawing();
  }

  CloseWindow();
}

void SystemOfBodies::animate_tree_universal(const ParsedData &data) {
  InitWindow(800, 600, "Tree Animation - Universal Joint");
  SetTargetFPS(60);

  Camera3D camera = {
      {10, 5, 10}, {0, -3, 0}, {0, 1, 0}, 45, CAMERA_PERSPECTIVE};

  const double l = 2.0;
  size_t step = 0;
  size_t total_steps = data.times.size();
  float time_accum = 0.0f;

  std::vector<Vector3> start_pos(n);
  std::vector<Vector3> end_pos(n);
  std::vector<arma::mat33> world_rotation(n);

  Color colors[] = {RED, BLUE, GREEN, ORANGE, PURPLE, YELLOW, PINK, SKYBLUE};

  while (!WindowShouldClose()) {
    float dt = GetFrameTime();
    time_accum += dt;

    while (step < total_steps - 1 &&
           time_accum > data.times[step + 1] - data.times[0]) {
      step++;
    }

    // Track which DOF index we're at
    int dof_index = 0;

    for (int k = n - 1; k >= 0; k--) {
      int body_dofs = bodies[k]->hinge_map.n_rows;

      // Build local rotation matrix from generalized coordinates
      arma::mat33 local_rot;

      if (body_dofs == 1) {
        // Simple revolute joint about Z (or whatever your default is)
        double theta = data.pos[dof_index][step];
        double c = cos(theta), s = sin(theta);
        // Rotation about Z
        local_rot = {{c, -s, 0}, {s, c, 0}, {0, 0, 1}};
      } else if (body_dofs == 2 && bodies[k]->is_dependent_hinge_map) {
        // Universal joint: R_x(theta1) * R_z(theta2)
        double theta1 = data.pos[dof_index][step];
        double theta2 = data.pos[dof_index + 1][step];
        double c1 = cos(theta1), s1 = sin(theta1);
        double c2 = cos(theta2), s2 = sin(theta2);

        // R_x(theta1) * R_z(theta2)
        local_rot = {
            {c2, -s2, 0}, {c1 * s2, c1 * c2, -s1}, {s1 * s2, s1 * c2, c1}};
      }

      if (bodies[k]->parent_ID == 0 || k == n - 1) {
        // Root body
        start_pos[k] = {0, 0, 0};
        world_rotation[k] = local_rot;
      } else {
        // Child body
        int parent = bodies[k]->parent_ID - 1;
        start_pos[k] = end_pos[parent];
        world_rotation[k] = world_rotation[parent] * local_rot;
      }

      // Body points in -Y direction in local frame
      arma::vec3 local_vec = {0, -l, 0};
      arma::vec3 world_vec = world_rotation[k] * local_vec;

      end_pos[k] = {start_pos[k].x + (float)world_vec(0),
                    start_pos[k].y + (float)world_vec(1),
                    start_pos[k].z + (float)world_vec(2)};

      dof_index += body_dofs;
    }

    BeginDrawing();
    ClearBackground(RAYWHITE);
    BeginMode3D(camera);

    for (size_t k = 0; k < n; k++) {
      DrawCylinderEx(start_pos[k], end_pos[k], 0.08f, 0.08f, 8, colors[k % 8]);
      DrawSphere(start_pos[k], 0.1f, DARKGRAY);
      DrawSphere(end_pos[k], 0.1f, DARKGRAY);
    }

    DrawGrid(10, 1.0f);
    EndMode3D();

    DrawText(TextFormat("Time: %.2f s", data.times[step]), 10, 10, 20, GRAY);
    DrawText(TextFormat("Step: %d / %d", (int)step, (int)total_steps), 10, 35,
             20, GRAY);

    EndDrawing();
  }

  CloseWindow();
}
