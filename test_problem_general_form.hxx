#ifndef TEST_PROBLEM_GENERAL_FORM_HXX
#define TEST_PROBLEM_GENERAL_FORM_HXX
#include "cqp.hxx"
#include <iostream>
#include <iomanip>

/* primal problem (P)
 * 	min_x x^T*0.5*Q*x + x^T*c
 * 	s.t   A*x == a
 * 	      B*x >= b
 *
 * dual problem (D)
 * 	max_{x,y,z} x^T*0.5*Q*x + x^T*c - y^T*(A*x-a) - z^T*(B*x-b)
 * 	s.t	    Q*x - A^T*y - B^T*z == -c
 * 				      z >= 0
 */
struct test_problem_general_form {
	int print_level; //0 no messages, 1 print test return values, gap, error & warnings, 2 print all from 1 and input matrices and vectors and output vectors, 3 prints all from 2 + start/stop messages 
	bool problem_set;
	cqp::matrix Q, A, B;
	cqp::vector c, a, b;
	cqp::vector x, y, z;

	cqp::scalar gap; //== y^T*(A*x-a)+z^T*(B*x-b)
	/* for a vector v let max(0,v) denote the component-wise maximum of v and the zero vector, i.e. ith component is max(0,v_i) where v_i ith entry of v
	 * for m-vector and n-vector v let [u; v] denote the (m+n)-vector whose first m components are those of u and remaining n components those of v
	 * cqp::vector ineq_error_vec; //==max(0, -[z ; Bx-b])
	 * cqp::vector eq_error_vec; //==[A*x-a ; Q*x-A^T*y-B^T*z+c]
	 */
	cqp::vector error_vector; //==[ineq_error_vec ; error_vec]
	
	cqp::scalar objective_function(cqp::vector input) { 
		return input.dot(0.5*Q*input + c);
	}
	cqp::scalar dual_objective_function(cqp::vector input1, cqp::vector input2, cqp::vector input3) { 
		return ((input1.transpose()*0.5*Q + c.transpose() - input2.transpose()*A-input3.transpose()*B)*input1).value()+input2.dot(a)+input3.dot(b);
	}

	test_problem_general_form(int print_level = 0);

	test_problem_general_form(const Eigen::Ref<const cqp::matrix> & Q, const Eigen::Ref<const cqp::vector> & c, 
			const Eigen::Ref<const cqp::matrix> & A, const Eigen::Ref<const cqp::vector> & a, 
			const Eigen::Ref<const cqp::matrix> & B, const Eigen::Ref<const cqp::vector> & b, bool print_level = 0);

	int set_problem_box(int max_numerator = 65536, int max_denominator = 65536); //Finding point in a rectangle closest to a given point.

	int set_problem(const Eigen::Ref<const cqp::matrix> & Q, const Eigen::Ref<const cqp::vector> & c, 
			const Eigen::Ref<const cqp::matrix> & A, const Eigen::Ref<const cqp::vector> & a, 
			const Eigen::Ref<const cqp::matrix> & B, const Eigen::Ref<const cqp::vector> & b);

	void test_solve();
};
#endif
