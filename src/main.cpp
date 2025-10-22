#include "tests_utils.h"
#include "muParser.h"



int main()
{
	constexpr int n{40};

	//test_single_body_multi_dof();
	//test_three_body_from_course();
	test_n_body_system(n);
	//test_inverse_dynamics_three_body_from_course();

	return 0;
}