#ifndef TEST_PROBLEM_HXX
#define TEST_PROBLEM_HXX
#include "cqp.hxx"
#include <iostream>
#include <iomanip>

/* primal problem (P)
 * 	min_x x^T*0.5*Q*x + x^T*c
 * 	s.t   A*x == b
 * 	      x >= 0
 */
class test_problem {
	bool tests_print_messages = true; //Tests should print data about input, output, and known error?
	bool valid; //Valid problem specification?
	bool have_optimal_x; //To compare computed solution against.
	bool have_optimal_y;
	
	cqp::index m, n;
	cqp::vector b, c;
	cqp::matrix A, Q;
	cqp::vector x, y; //(x,y) value returned by last call to solve or homotopy_algorithm.
	//x initialized to 0 if no initial_x specified by user, y initialized to 0;
	cqp::vector z; //== Q*x - A.transpose()*y + c
	cqp::vector w; //== A*x-b
	cqp::vector optimal_x, optimal_y; //true optimal solution if known.
	cqp::vector optimal_z; //for error computation have if have_optimal_x && have_optimal_y
	cqp::vector optimal_w;  //for error computation. have if have_optimal_x
	
	cqp::vector initial_x; //default value for homotopy_algorithm;
	
	cqp::scalar ineq_gap; // == x.dot(z)
	cqp::scalar eq_gap; // == w.dot(y)
	cqp::scalar gap; //== ineq_gap + eq_gap
	cqp::scalar optimal_ineq_gap; //should be 0 with exact arithmetic
	cqp::scalar optimal_eq_gap; //should be 0 with exact arithmetic
	cqp::scalar optimal_gap; //should be 0 with exact arithmetic
	
	cqp::scalar ineq_error; //L-infinity/max norm of [max(x,0) ; max(z,0)] 
	cqp::scalar eq_error; //max norm of w
	cqp::scalar error; //feasibility error == max(ineq_error, eq_error)
	cqp::scalar optimal_ineq_error;
	cqp::scalar optimal_eq_error; 
	cqp::scalar optimal_error; 

	cqp::scalar gamma, beta, eta; //default values for homotopy_algorithm.

	public:
	test_problem(int max_numerator = 6553, int max_denominator = 6553, bool definite = false, bool tests_print_messages = true); 
	test_problem(const cqp::matrix & Q, const cqp::vector & c, const cqp::matrix & A, const cqp::vector & b, const cqp::vector & initial_x, bool tests_print_messages = true);
	test_problem(const cqp::matrix & Q, const cqp::vector & c, const cqp::matrix & A, const cqp::vector & b, bool tests_print_messages = true);

	void set_x(cqp::vector input);
	void set_y(cqp::vector input);
	void set_optimal_x(cqp::vector input);
	void set_optimal_y(cqp::vector input);
	void set_initial_x(cqp::vector input);

	int set_gamma(cqp::scalar input);
	int set_beta(cqp::scalar input); 
	int set_eta(cqp::scalar input);

	cqp::vector get_x(cqp::vector input) { return x; }
	cqp::vector get_y(cqp::vector input) {  return y; }
	cqp::vector get_optimal_x(cqp::vector input) { return optimal_x; }
	cqp::vector get_optimal_y(cqp::vector input) { return optimal_y; }
	cqp::vector get_initial_x(cqp::vector input) { return initial_x; }

	cqp::scalar get_beta(cqp::scalar input) { return beta; }
	cqp::scalar get_eta(cqp::scalar input) { return eta; }
	cqp::scalar get_gamma(cqp::scalar input) { return gamma; }

	cqp::scalar objective_function(cqp::vector input) {
		return input.dot(0.5*Q*input + c);
	}

	int set_problem_box(int max_numerator = 6553, int max_denominator = 6553, bool definite = false); 
	//Finding point in a rectangle closest to a given point. Randomly generated on each call.

	int set_problem(const cqp::matrix & Q, const cqp::vector & c, const cqp::matrix & A, const cqp::vector & b, const cqp::vector & initial_x); //if initial_x for homotopy algorithm known.
	//Set a custom problem. Destroys previous values for all previous member variables.

	int set_problem(const cqp::matrix & Q, const cqp::vector & c, const cqp::matrix & A, const cqp::vector & b) { set_problem(Q, c, A, b, cqp::vector::Zero(A.cols())); return 0; }

	void test_homotopy_algorithm(cqp::scalar beta, cqp::scalar eta, cqp::scalar gamma, cqp::vector initial_x);

	void test_homotopy_algorithm() { test_homotopy_algorithm(beta, eta, gamma, initial_x); }

	void test_solve();
};
#endif
