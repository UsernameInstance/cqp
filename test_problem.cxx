#include "test_problem.hxx"
test_problem::test_problem(int max_numerator, int max_denominator, bool definite, bool tests_print_messages) : have_optimal_x(false), have_optimal_y(false) {
	if(!set_problem_box(max_numerator, max_denominator, definite)) {
		this->tests_print_messages = tests_print_messages;
	}
}

test_problem::test_problem(const cqp::matrix & Q, const cqp::vector & c, const cqp::matrix & A, const cqp::vector & b, const cqp::vector & initial_x, bool tests_print_messages) : have_optimal_x(false), have_optimal_y(false) {
	if(!set_problem(Q, c, A, b, initial_x)) {
		this->tests_print_messages = tests_print_messages;
	}
}

test_problem::test_problem(const cqp::matrix & Q, const cqp::vector & c, const cqp::matrix & A, const cqp::vector & b, bool tests_print_messages) : have_optimal_x(false), have_optimal_y(false) {
	if(!set_problem(Q, c, A, b)) {
		this->tests_print_messages = tests_print_messages;
	} 
}

void test_problem::set_x(cqp::vector input) { 
	if(input.size() == n) {
		x = input;
		w = A*x-b;
		eq_gap = w.dot(y);
		gap = ineq_gap+eq_gap;

		eq_error = 0;
		if(m) {
			eq_error = w.cwiseAbs().maxCoeff();
		}
		ineq_error = 0;
		if(n) {
			ineq_error = cqp::matrix::Zero(n,1).cwiseMax(-x).maxCoeff();
			ineq_error = cqp::max(ineq_error, cqp::matrix::Zero(n,1).cwiseMax(-z).maxCoeff());
		}
		error = cqp::max(eq_error, ineq_error);
	} else {
		std::cout << "\nError in \"void test_problem::set_x(cqp::vector)\": x must be an " << n << "-vector." << std::endl;
	}
}

void test_problem::set_y(cqp::vector input) { 
	if(input.size() == m) {
		y = input;
		z = Q*x-A*y+c;
		ineq_gap = x.dot(z);
		eq_gap = w.dot(y);
		gap = ineq_gap+eq_gap;

		ineq_error = 0;
		if(n) {
			ineq_error = cqp::matrix::Zero(n,1).cwiseMax(-x).maxCoeff();
			ineq_error = cqp::max(ineq_error, cqp::matrix::Zero(n,1).cwiseMax(-z).maxCoeff());
		}
		error = cqp::max(eq_error, ineq_error);

	}
	else {
		std::cout << "\nError in \"void test_problem::set_y(cqp::vector)\": y must be an " << m << "-vector." << std::endl;
	}
}

void test_problem::set_optimal_x(cqp::vector input) { 
	if(input.size() == n) {
		optimal_x = input;
		have_optimal_x = true;
		optimal_w = A*optimal_x-b;
		if(have_optimal_y) {
			optimal_eq_gap = optimal_w.dot(optimal_y);
		}
		optimal_gap = optimal_ineq_gap+optimal_eq_gap;

		optimal_eq_error = 0;
		if(m) {
			optimal_eq_error = optimal_w.cwiseAbs().maxCoeff();
		}
		optimal_ineq_error = 0;
		if(n) {
			optimal_ineq_error = cqp::matrix::Zero(n,1).cwiseMax(-optimal_x).maxCoeff();
		}
		if(have_optimal_y) {
			optimal_z = Q*optimal_x - A.transpose()*optimal_y+c;
			if(n) {
				optimal_ineq_error = cqp::max(optimal_ineq_error, cqp::matrix::Zero(n,1).cwiseMax(-optimal_z).maxCoeff());
			}
		}
		optimal_error = cqp::max(optimal_eq_error, optimal_ineq_error);
	} else {
		std::cout << "\nError in \"void test_problem::set_optimal_x(cqp::vector)\": optimal_x must be an " << n << "-vector." << std::endl;
	}
}

void test_problem::set_optimal_y(cqp::vector input) { 
	if(input.size() == m) {
		optimal_y = input;
		have_optimal_y = true;
		
		if(have_optimal_x) {
			optimal_z = Q*optimal_x-A*optimal_y+c;
			optimal_ineq_gap = optimal_x.dot(optimal_z);
			optimal_eq_gap = optimal_w.dot(optimal_y);
			optimal_gap = optimal_ineq_gap+optimal_eq_gap;
			if(n) {
				optimal_ineq_error = cqp::max(optimal_ineq_error, cqp::matrix::Zero(n,1).cwiseMax(-optimal_z).maxCoeff());
			}
			optimal_error = cqp::max(optimal_eq_error, optimal_ineq_error);
		}
	} else {
		std::cout << "\nError in \"void test_problem::set_optimal_y(cqp::vector)\": optimal_y must be an " << m << "-vector." << std::endl;
	}
}

void test_problem::set_initial_x(cqp::vector input) { 
	if(input.size() == n) {
		initial_x = input;
	} else {
		std::cout << "\nError in \"void test_problem::set_initial_x(cqp::vector)\": initial_x must be an " << n << "-vector." << std::endl;
	}
}

int test_problem::set_beta(cqp::scalar input) {
	if(input < 0 || input > 1) {
		std::cout << "\nError in test_problem::set_beta(cqp::scalar): require beta > 0 and beta < 1." << std::endl;
		return 1;
	} else {
		beta = input;
		return 0;
	}
}


int test_problem::set_eta(cqp::scalar input) {
	if(input < 0) {
		std::cout << "\nError in test_problem::set_eta(cqp::scalar): require eta > 0." << std::endl;
		return 1;
	} else {
		eta = input;
		return 0;
	}
}

int test_problem::set_gamma(cqp::scalar input) {
	if(input < 0) {
		std::cout << "\nError in test_problem::set_gamma(cqp::scalar): require gamma > 0." << std::endl;
		return 1;
	} else {
		gamma = input;
		return 0;
	}
}

int test_problem::set_problem_box(int max_numerator, int max_denominator, bool definite) { 	
	if(max_numerator < 0 || max_denominator < 1) {
		std::cout << "\nError in test_problem::set_problem_box(int, int, bool): require max_numerator >= 0 and max_denominator>=1." << std::endl;
		valid = false;
		return 1;
	} 
	valid = true;

	m = 5; n = 2*m; //x_{m+1}, ..., x_{2*m} are slack variables.

	cqp::vector num(m), den(m);
	num = (max_numerator*(Eigen::Matrix<double, Eigen::Dynamic, 1>::Random(m)+Eigen::Matrix<double, Eigen::Dynamic, 1>::Constant(m,1))/2).cast<int>().cast<mpq_class>();
	den = (Eigen::Matrix<double, Eigen::Dynamic, 1>::Constant(m,1)+(max_denominator-1)*(Eigen::Matrix<double, Eigen::Dynamic, 1>::Random(m)+Eigen::Matrix<double, Eigen::Dynamic, 1>::Constant(m,1))/2).cast<int>().cast<mpq_class>();

	b = num.cwiseQuotient(den);

	num = (max_numerator*(Eigen::Matrix<double, Eigen::Dynamic, 1>::Random(m)+Eigen::Matrix<double, Eigen::Dynamic, 1>::Constant(m,1))/2).cast<int>().cast<mpq_class>();
	den = (Eigen::Matrix<double, Eigen::Dynamic, 1>::Constant(m,1)+(max_denominator-1)*(Eigen::Matrix<double, Eigen::Dynamic, 1>::Random(m)+Eigen::Matrix<double, Eigen::Dynamic, 1>::Constant(m,1))/2).cast<int>().cast<mpq_class>();
	cqp::vector target = num.cwiseQuotient(den);
	
	c.resize(n); c << (-2*target), cqp::vector::Zero(m);

	Q = cqp::matrix::Zero(n,n); Q.block(0,0,m,m) = 2*cqp::matrix::Identity(m,m);
	A.resize(m, n); A << cqp::matrix::Identity(m,m), cqp::matrix::Identity(m,m);

	/*if original problem is in form
	 * 	min x^T 0.5Q x + c^T x
	 * 	    A*x <= b
	 * 	    x >= 0
	 * with Q positive definite then introducing slack variables s_x without altering the objective function results in problem with new matrix that is positive semidefinite and not positive definite.
	 * To fix this after introducing slack variables s_x add 0.5*u*(A*x+y-b)^T * (A*x+y-b) to the objective function with u>0. This gives new problem
	 *      min                [ Q+u*A^T*A  u*A^T ][ x ]			     [ x ]
	 *	    [x^T s_x^T] 0.5[    u*A      u*I  ][s_x]  + [c^T-u*b^T*A  -u*b^T][s_x]
	 *	    A*x+y = b
	 *	    x   >= 0
	 *	    s_x >= 0
	 * which has positive definite matrix whenever Q positive definite.
	 */
	if(definite) {
		Q += A.transpose()*A; 
		c -= A.transpose()*b;
	}

	x = cqp::vector::Zero(n);
	y = cqp::vector::Zero(m);
	w = -b;
	z = c;

	ineq_gap = 0;
	eq_gap = 0;
	gap = ineq_gap + eq_gap;

	eq_error = 0;
	if(m) {
		eq_error = w.cwiseAbs().maxCoeff();
	}
	ineq_error = 0;
	if(n) {
		ineq_error = cqp::matrix::Zero(n,1).cwiseMax(-x).maxCoeff();
		ineq_error = cqp::max(ineq_error, cqp::matrix::Zero(n,1).cwiseMax(-z).maxCoeff());
	}
	error = cqp::max(eq_error, ineq_error);

        initial_x.resize(n); initial_x << b/2.0, b/2.0; //initial point center of rectangle.
	beta = 0.5; gamma = 0x1p-31; eta = cqp::sqrt(((initial_x.asDiagonal())*(Q*x+c)).squaredNorm())/beta;
	eta = 1+cqp::scalar(eta.get_num()/eta.get_den());

	optimal_x = cqp::vector::Zero(n); 
	for(int i=0; i<m; ++i) { 
		optimal_x(i) = ((target(i) < 0)?0:((target(i) <= b(i))?target(i):b(i))); 
		optimal_x(m+i) = b(i)-optimal_x(i);
	}
	have_optimal_x = true;

	optimal_w = A*optimal_x-b;

	optimal_y = (optimal_x.asDiagonal()*A.transpose()).fullPivLu().solve(optimal_x.asDiagonal()*(Q*optimal_x+c));
	have_optimal_y = true;

	optimal_z = Q*optimal_x - A.transpose()*optimal_y + c;

	optimal_ineq_gap = optimal_x.dot(optimal_z);
	optimal_eq_gap = optimal_w.dot(optimal_y);
	optimal_gap = optimal_ineq_gap + optimal_eq_gap;

	optimal_eq_error = 0;
	if(m) {
		optimal_eq_error = optimal_w.cwiseAbs().maxCoeff();
	}
	optimal_ineq_error = 0;
	if(n) {
		optimal_ineq_error = cqp::matrix::Zero(n, 1).cwiseMax(-optimal_x).maxCoeff();
		optimal_ineq_error = cqp::max(optimal_ineq_error, cqp::matrix::Zero(n, 1).cwiseMax(-optimal_z).maxCoeff());
	}
	optimal_error = cqp::max(optimal_eq_error, optimal_ineq_error);

	return 0;
}



int test_problem::set_problem(const cqp::matrix & Q, const cqp::vector & c, const cqp::matrix & A, const cqp::vector & b, const cqp::vector & initial_x) {
	cqp::index m, n;
	m = A.rows(); n = A.cols();

	if(Q.cols() != n || c.size() !=n || initial_x.size() !=n || b.rows() != m) {
		std::cout << "\nError in test_problem::set_problem: input matrices dimension mismatch." << std::endl;
		valid = false;
		return 1;
	}

	valid = true;
	this->m = m; this->n = n;
	this->Q = Q; this->c = c; this->A = A; this->b = b; this->initial_x = initial_x;

	x = cqp::vector::Zero(n);
	y = cqp::vector::Zero(m);
	w = -b;
	z = c;

	ineq_gap = 0;
	eq_gap = 0;
	gap = ineq_gap + eq_gap;

	eq_error = 0;
	if(m) {
		eq_error = w.cwiseAbs().maxCoeff();
	}
	ineq_error = 0;
	if(n) {
		ineq_error = cqp::matrix::Zero(n, 1).cwiseMax(-x).maxCoeff();
		ineq_error = cqp::max(ineq_error, cqp::matrix::Zero(n, 1).cwiseMax(-z).maxCoeff());
	}
	error = cqp::max(eq_error, ineq_error);

	optimal_x = cqp::vector::Zero(n); //need values
	optimal_y = cqp::vector::Zero(m);
	optimal_w = -b;
	optimal_z = c;
	
	have_optimal_x = false;
	have_optimal_y = false;

	return 0;
}

void test_problem::test_homotopy_algorithm(cqp::scalar beta, cqp::scalar eta, cqp::scalar gamma, cqp::vector initial_x) {
	std::cout << "\n****************************test_homotopy_algorithm****************************\n" << std::endl;

	std::cout.precision(17);
	int format_width = 9;
	
	if(tests_print_messages) {
		std::cout << "\nQ == \n" << Q << "\n\nc == \n" << c << "\n\nA == \n" << A << "\n\nb == \n" << b << "\n\ninitial_x == \n" << initial_x << "\n\n";
		std::cout << std::setw(format_width) << std::left << "beta == " << cqp::float_scalar(beta)  << "\n" 
			  << std::setw(format_width) << std::left << "gamma == " << cqp::float_scalar(gamma)  << "\n"
			  << std::setw(format_width) << std::left << "eta == " << cqp::float_scalar(eta) << "\n" << std::endl;
		std::cout << "Calling \"cqp::homotopy_algorithm(Q, c, A, b, initial_x, beta, gamma, eta, x, y)\"." << std::endl;
	}
	
	cqp::homotopy_algorithm(Q, c, A, b, initial_x, beta, gamma, eta, x, y);

	z = (Q*x+c-A.transpose()*y);
	w = A*x - b;
	ineq_gap = x.transpose()*z;
	eq_gap = w.transpose()*y;
	gap = ineq_gap+eq_gap;

	eq_error = 0;
	if(m) {
		eq_error = w.cwiseAbs().maxCoeff();
	}
	ineq_error = 0;
	if(n) {
		ineq_error = cqp::matrix::Zero(n, 1).cwiseMax(-x).maxCoeff();
		ineq_error = cqp::max(ineq_error, cqp::matrix::Zero(n, 1).cwiseMax(-z).maxCoeff());
	}
	error = cqp::max(eq_error, ineq_error);

	if(tests_print_messages) {
		std::cout << "Call complete." << '\n' << std::endl;

		std::cout << "x == \n" << x << "\n\n";
		std::cout << "y == \n" << y << "\n\n";

		if(have_optimal_x) {
			std::cout << "optimal_x == \n" << optimal_x << "\n\n";
		}
		if(have_optimal_y) {
			std::cout << "optimal_y == \n" << optimal_y << "\n\n";
		}

		std::cout << "z == Q*x+c-A.transpose()*y == \n" << z << "\n\n";
		std::cout << "w == A*x-b == \n" << w << "\n\n";

		if(have_optimal_x) {
			if(have_optimal_y) {
				std::cout << "optimal_z == Q*optimal_x+c-A.transpose()*optimal_y == \n" << optimal_z << "\n\n";
			}
			std::cout << "optimal_w == A*optimal_x-b == \n" << optimal_w << "\n\n";
		}
		
		format_width = 14;
		std::cout << std::setw(format_width) << std::left << "ineq_error == " << cqp::float_scalar(ineq_error) << "\n";
		std::cout << std::setw(format_width) << std::left << "eq_error == " << cqp::float_scalar(eq_error) << "\n";
		std::cout <<  std::setw(format_width) << std::left << "error == " << cqp::float_scalar(error) << "\n\n";

		if(have_optimal_x && have_optimal_y) {
			format_width = 22;

			std::cout << std::setw(format_width) << std::left << "optimal_ineq_error == " << cqp::float_scalar(optimal_ineq_error) << "\n";
			std::cout << std::setw(format_width) << std::left << "optimal_eq_error == " << cqp::float_scalar(optimal_eq_error) << "\n";
			std::cout << std::setw(format_width) << std::left << "optimal_error == " << cqp::float_scalar(optimal_error) << "\n\n";
		}

		format_width = 28;
		std::cout << std::setw(format_width) << std::left << "predicted_gap == n*gamma == " << cqp::float_scalar(n*gamma) << "\n\n";

		format_width = 26;
		std::cout << std::setw(format_width) << std::left << "ineq_gap == x^T*z == " << cqp::float_scalar(ineq_gap) << "\n";
		std::cout << std::setw(format_width) << std::left << "eq_gap == w^T*y == " << cqp::float_scalar(eq_gap) << "\n";
		std::cout <<  std::setw(format_width) << std::left << "gap == ineq_gap+eq_gap == " << cqp::float_scalar(gap) << "\n\n";

		if(have_optimal_x && have_optimal_y) {
			format_width = 50;

			std::cout << std::setw(format_width) << std::left << "optimal_ineq_gap == optimal_x^T*optimal_z == " << cqp::float_scalar(optimal_ineq_gap) << "\n";
			std::cout << std::setw(format_width) << std::left << "optimal_eq_gap == optimal_w^T*optimal_y == " << cqp::float_scalar(optimal_eq_gap) << "\n";
			std::cout << std::setw(format_width) << std::left << "optimal_gap == optimal_ineq_gap+optimal_eq_gap == " << cqp::float_scalar(optimal_gap) << "\n\n";
		}

		format_width = 33;
		std::cout << std::setw(format_width) << std::left << "objective_function(x) == " << cqp::float_scalar(objective_function(x)) << "\n";
		if(have_optimal_x) {
			std::cout << std::setw(format_width) << std::left << "objective_function(optimal_x) == " << cqp::float_scalar(objective_function(optimal_x)) << "\n\n";
			format_width = 55;
			std::cout << std::setw(format_width) << std::left << "objective_function(x) - objective_function(optimal_x) == " << cqp::float_scalar(objective_function(x) - objective_function(optimal_x)) << std::endl;
		} else {
			std::cout << std::endl;
		}
	}
}


void test_problem::test_solve() {
	std::cout << "\n****************************test_solve****************************\n" << std::endl;

	std::cout.precision(17);
	int format_width = 9;

	if(tests_print_messages) {
		std::cout << "\nQ == \n" << Q << "\n\nc == \n" << c << "\n\nA == \n" << A << "\n\nb == \n" << b << "\n\n";
		std::cout << "Calling \"cqp::solve(Q, c, A, b, x, y)\"." << std::endl;
	}

	cqp::solve(Q, c, A, b, x, y);

	z = (Q*x+c-A.transpose()*y);
	w = A*x - b;
	ineq_gap = x.transpose()*z;
	eq_gap = w.transpose()*y;
	gap = ineq_gap+eq_gap;

	eq_error = 0;
	if(m) {
		eq_error = w.cwiseAbs().maxCoeff();
	}
	ineq_error = 0;
	if(n) {
		ineq_error = cqp::matrix::Zero(n, 1).cwiseMax(-x).maxCoeff();
		ineq_error = cqp::max(ineq_error, cqp::matrix::Zero(n, 1).cwiseMax(-z).maxCoeff());
	}
	error = cqp::max(eq_error, ineq_error);

	if(tests_print_messages) {
		std::cout << "Call complete." << '\n' << std::endl;

		std::cout << "x == \n" << x << "\n\n";
		std::cout << "y == \n" << y << "\n\n";

		if(have_optimal_x) {
			std::cout << "optimal_x == \n" << optimal_x << "\n\n";
		}
		if(have_optimal_y) {
			std::cout << "optimal_y == \n" << optimal_y << "\n\n";
		}

		std::cout << "z == Q*x+c-A.transpose()*y == \n" << z << "\n\n";
		std::cout << "w == A*x-b == \n" << w << "\n\n";

		if(have_optimal_x) {
			if(have_optimal_y) {
				std::cout << "optimal_z == Q*optimal_x+c-A.transpose()*optimal_y == \n" << optimal_z << "\n\n";
			}
			std::cout << "optimal_w == A*optimal_x-b == \n" << optimal_w << "\n\n";
		}

		format_width = 14;
		std::cout << std::setw(format_width) << std::left << "ineq_error == " << cqp::float_scalar(ineq_error) << "\n";
		std::cout << std::setw(format_width) << std::left << "eq_error == " << cqp::float_scalar(eq_error) << "\n";
		std::cout <<  std::setw(format_width) << std::left << "error == " << cqp::float_scalar(error) << "\n\n";

		if(have_optimal_x && have_optimal_y) {
			format_width = 22;

			std::cout << std::setw(format_width) << std::left << "optimal_ineq_error == " << cqp::float_scalar(optimal_ineq_error) << "\n";
			std::cout << std::setw(format_width) << std::left << "optimal_eq_error == " << cqp::float_scalar(optimal_eq_error) << "\n";
			std::cout << std::setw(format_width) << std::left << "optimal_error == " << cqp::float_scalar(optimal_error) << "\n\n";
		}

		format_width = 28;
		std::cout << std::setw(format_width) << std::left << "predicted_gap == n*gamma == " << cqp::float_scalar(n*gamma) << "\n\n";

		format_width = 26;
		std::cout << std::setw(format_width) << std::left << "ineq_gap == x^T*z == " << cqp::float_scalar(ineq_gap) << "\n";
		std::cout << std::setw(format_width) << std::left << "eq_gap == w^T*y == " << cqp::float_scalar(eq_gap) << "\n";
		std::cout <<  std::setw(format_width) << std::left << "gap == ineq_gap+eq_gap == " << cqp::float_scalar(gap) << "\n\n";

		if(have_optimal_x && have_optimal_y) {
			format_width = 50;

			std::cout << std::setw(format_width) << std::left << "optimal_ineq_gap == optimal_x^T*optimal_z == " << cqp::float_scalar(optimal_ineq_gap) << "\n";
			std::cout << std::setw(format_width) << std::left << "optimal_eq_gap == optimal_w^T*optimal_y == " << cqp::float_scalar(optimal_eq_gap) << "\n";
			std::cout << std::setw(format_width) << std::left << "optimal_gap == optimal_ineq_gap+optimal_eq_gap == " << cqp::float_scalar(optimal_gap) << "\n\n";
		}

		format_width = 33;
		std::cout << std::setw(format_width) << std::left << "objective_function(x) == " << cqp::float_scalar(objective_function(x)) << "\n";
		if(have_optimal_x) {
			std::cout << std::setw(format_width) << std::left << "objective_function(optimal_x) == " << cqp::float_scalar(objective_function(optimal_x)) << "\n\n";
			format_width = 55;
			std::cout << std::setw(format_width) << std::left << "objective_function(x) - objective_function(optimal_x) == " << cqp::float_scalar(objective_function(x) - objective_function(optimal_x)) << std::endl;
		} else {
			std::cout << std::endl;
		}
	}
}
