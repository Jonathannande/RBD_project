#include "tests_utils.h"

int main() {
  constexpr int number_of_bodies{240};
  constexpr int seed{256};

  // test_single_body_multi_dof();
  // test_n_body_system(n);
  // test_chain_dynamics();
  // test_spherical_hinge();
  // test_universal_hinge();
  // test_inverse_dynamics_three_body_from_course();
  // test_dense();
  // test_three_body_from_course_with_viz();
  // test_raylib();
  // test_rayy();

  // test_three_body_from_course();
  test_random_tree(number_of_bodies, seed);
  // test_tree_dynamics();
  return 0;
}
