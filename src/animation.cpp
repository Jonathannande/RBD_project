
#include "system_of_bodies.h"
#include <armadillo>
#include <raylib.h>

void SystemOfBodies::animate_tree(const ParsedData &data) {
  InitWindow(800, 600, "Tree Animation");
  SetTargetFPS(60);

  Camera3D camera = {{0, 0, 15}, {0, -5, 0}, {0, 1, 0}, 45, CAMERA_PERSPECTIVE};

  const double l = 2.0;
  size_t step = 0;
  size_t total_steps = data.times.size();
  float time_accum = 0.0f;

  std::vector<Vector3> start_pos(n);
  std::vector<Vector3> end_pos(n);
  std::vector<double> world_angle(n);

  Color colors[] = {RED, BLUE, GREEN, ORANGE, PURPLE, YELLOW, PINK, SKYBLUE};

  while (!WindowShouldClose()) {
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

      end_pos[k] = {start_pos[k].x + (float)(l * sin(world_angle[k])),
                    start_pos[k].y - (float)(l * cos(world_angle[k])), 0};
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
