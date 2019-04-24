#include "test_problem_general_form.hxx"

test_problem_general_form::test_problem_general_form(int print_level) : print_level(print_level), problem_set(false) {}

test_problem_general_form::test_problem_general_form(const Eigen::Ref<const cqp::matrix> & Q, const Eigen::Ref<const cqp::vector> & c, 
			const Eigen::Ref<const cqp::matrix> & A, const Eigen::Ref<const cqp::vector> & a, 
			const Eigen::Ref<const cqp::matrix> & B, const Eigen::Ref<const cqp::vector> & b, bool print_level) : print_level(print_level) {
	set_problem(Q, c, A, a, B, b);
}

/* set the problem to
 * 	min_x x^T*x - 2*target^T*x
 * 	 s.t. -x >= -upper_bound
 * 	       x >= 0
 */
int test_problem_general_form::set_problem_box(int max_numerator, int max_denominator) {
	if(max_numerator < 0 || max_denominator < 1) {
		if(print_level) {
			std::cout << "\nWarning in test_problem::set_problem_box(int, int): require max_numerator >= 0 and max_denominator>=1." << std::endl;
		}
		problem_set = false;
		return 1;
	} 
	
	problem_set = true;
	cqp::index m = 5; //number primal_variables == half the number of inequality constraints

	cqp::vector num(m), den(m);
	num = (max_numerator*(Eigen::Matrix<double, Eigen::Dynamic, 1>::Random(m)+Eigen::Matrix<double, Eigen::Dynamic, 1>::Constant(m,1))/2).cast<int>().cast<mpq_class>();
	den = (Eigen::Matrix<double, Eigen::Dynamic, 1>::Constant(m,1)+(max_denominator-1)*(Eigen::Matrix<double, Eigen::Dynamic, 1>::Random(m)+Eigen::Matrix<double, Eigen::Dynamic, 1>::Constant(m,1))/2).cast<int>().cast<mpq_class>();

	cqp::vector upper_bound = num.cwiseQuotient(den); 

	num = (max_numerator*(Eigen::Matrix<double, Eigen::Dynamic, 1>::Random(m)+Eigen::Matrix<double, Eigen::Dynamic, 1>::Constant(m,1))/2).cast<int>().cast<mpq_class>();
	den = (Eigen::Matrix<double, Eigen::Dynamic, 1>::Constant(m,1)+(max_denominator-1)*(Eigen::Matrix<double, Eigen::Dynamic, 1>::Random(m)+Eigen::Matrix<double, Eigen::Dynamic, 1>::Constant(m,1))/2).cast<int>().cast<mpq_class>();
	cqp::vector target = num.cwiseQuotient(den);
	
	c = -2*target;
	Q = 2*cqp::matrix::Identity(m,m);
	A = cqp::matrix::Zero(0,m); 
	a = cqp::vector::Zero(0);
	B = cqp::matrix::Zero(2*m,m);
	B << -cqp::matrix::Identity(m,m), cqp::matrix::Identity(m,m);
	b = cqp::vector::Zero(2*m);
	b.head(m) = -upper_bound;

	if(print_level>2) {
		std::cout << "problem set successfully.\n";
	}

	return 0;
}

//checks for consistent dimensions. returns nonzero if and does not set if inconsistent dimensions.
int test_problem_general_form::set_problem(const Eigen::Ref<const cqp::matrix> & Q, const Eigen::Ref<const cqp::vector> & c, 
			const Eigen::Ref<const cqp::matrix> & A, const Eigen::Ref<const cqp::vector> & a, 
			const Eigen::Ref<const cqp::matrix> & B, const Eigen::Ref<const cqp::vector> & b) { 
	cqp::index num_var, num_eq, num_ineq;
	num_var = Q.rows(); num_eq = A.rows(); num_ineq = B.rows();
	if((Q.cols() != num_var) || (c.size() != num_var) || (A.cols() != num_var) || (B.cols() != num_var) || (a.size() != num_eq) || (b.size() != num_ineq)) {
		if(print_level) {
			std::cout << "Warning in \"test_problem_general_form::set_problem\". Incompatible matrix dimensions. Problem not set.\n";
		}
		return 1;
	}

	this->Q = Q;	this->A = A;	this->B = B;
	this->c = c;	this->a = a;	this->b = b;

	if(print_level>2) {
		std::cout << "Problem set successfully.\n";
	}

	return 0;
} 

void test_problem_general_form::test_solve() { 
	if(print_level > 2) {
		std::cout << "\n****************************test_problem_general_form::test_solve****************************\n" << std::endl;
	}

	std::cout.precision(17);
	int format_width = 16;
	if(print_level >= 2) {
		std::cout << "\nQ == \n" << Q << "\n\nc == \n" << c << "\n\nA == \n" << A << "\n\na == \n" << a << "\n\nB == \n" << B << "\n\nb == \n" << b << "\n\n";

		if(print_level > 2) {
				std::cout << "Calling \"cqp::solve(Q, c, A, a, B, b, x, y, z)\"." << std::endl;
		}
	}

	cqp::vector2 solve_return_value = cqp::solve(Q,c,A,a,B,b,x,y,z);

	error_vector = cqp::vector::Zero(z.rows()+B.rows()+A.rows()+Q.rows());
	error_vector.head(z.rows()+B.rows()) = error_vector.head(z.rows()+B.rows()).cwiseMax(-(cqp::vector(z.rows()+B.rows()) << z, B*x-b).finished()); //[max(0,-z) ; max(0,-(B*x-b))]
	error_vector.tail(A.rows()+Q.rows()) << A*x-a, Q*x-A.transpose()*y-B.transpose()*z+c;// = (cqp::vector(A.rows()+Q.rows()) << A*x-a, Q*x-A.transpose()*y-B.transpose()*z+c).finished()

	gap = y.dot(A*x-a)+z.dot(B*x-b);

	if(print_level >= 1) {
		if(print_level >= 2) {
			if(print_level > 2) {
				std::cout << "Call complete." << '\n' << std::endl;
			}
			std::cout << "x == \n" << x << "\n\n";
			std::cout << "y == \n" << y << "\n\n";
			std::cout << "z == \n" << z << "\n\n";
		//	std::cout << "error_vector == \n" << error_vector.cast<cqp::float_scalar>() << "\n\n";
		}
		
		format_width = 9;
		cqp::scalar error_max_norm = 0;
		if(error_vector.size()) { //if nonempty
			error_max_norm = error_vector.cwiseAbs().maxCoeff();
		}

		format_width = 22;
		std::cout << std::setw(format_width) << std::left << "solve_return_value == " << solve_return_value.cast<cqp::float_scalar>().transpose() << "\n";
		std::cout << std::setw(format_width) << std::left << "error_max_norm == " << cqp::float_scalar(error_max_norm) << "\n";
		std::cout <<  std::setw(format_width) << std::left << "gap == " << cqp::float_scalar(gap) << "\n\n";
	}
}
