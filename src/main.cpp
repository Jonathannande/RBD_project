#include "tests_utils.h"

int main() {
  constexpr int n{80};

  // test_single_body_multi_dof();
  // test_three_body_from_course();
  // test_n_body_system(n);
  test_tree_dynamics();
  // test_chain_dynamics();
  //   test_spherical_hinge();
  //   test_universal_hinge();
  //        test_inverse_dynamics_three_body_from_course();
  //        test_dense();
  //        test_three_body_from_course_with_viz();
  //        test_raylib();

  // test_rayy();
  return 0;
}
