#include "misc_tests.hxx"

void test_unconstrained() {
	std::cout << "\n****************************test_unconstrained****************************\n" << std::endl;
	cqp::matrix Q;
	cqp::vector c,x;
	
	std::cout << "no variables\n";
	Q = cqp::matrix::Zero(0,0); //no variables
	std::cout << "Q == \n" << Q << '\n';
	c = cqp::vector::Zero(0);
	std::cout << "c == \n" << c << '\n';
	x.resize(0);
	std::cout << "cqp::solve(Q,c,x) == " << cqp::solve(Q,c,x).transpose() << "\n";
	std::cout << "objective_function(x) == " << x.transpose()*(0.5*Q*x+c) << "\n";
	std::cout << "duality gap  == " << x.transpose()*(Q*x+c) << "\n";
	std::cout << "x == \n" << x << "\n\n";

	std::cout << "zero Q, zero c\n";
	Q = cqp::matrix::Zero(2,2); //zero Q, zero c
	std::cout << "Q == \n" << Q << '\n';
	c.resize(2); c << 0,0;
	std::cout << "c == \n" << c << '\n';
	x.resize(2);
	std::cout << "cqp::solve(Q,c,x) == " << cqp::solve(Q,c,x).transpose() << "\n";
	std::cout << "objective_function(x) == " << x.transpose()*(0.5*Q*x+c) << "\n";
	std::cout << "duality gap  == " << x.transpose()*(Q*x+c) << "\n";
	std::cout << "x == \n" << x << "\n\n";

	std::cout << "zero Q, nonzero c\n";
	Q = cqp::matrix::Zero(2,2); //zero Q, nonzero c
	std::cout << "Q == \n" << Q << '\n';
	c.resize(2); c << -1, 1;
	std::cout << "c == \n" << c << '\n';
	x.resize(2);
	std::cout << "cqp::solve(Q,c,x) == " << cqp::solve(Q,c,x).transpose() << "\n";
	std::cout << "objective_function(x) == " << x.transpose()*(0.5*Q*x+c) << "\n";
	std::cout << "duality gap  == " << x.transpose()*(Q*x+c) << "\n";
	std::cout << "x == \n" << x << "\n\n";

	std::cout << "positive definite Q\n";
	Q = 2*cqp::matrix::Identity(2,2); //positive definite Q 
	std::cout << "Q == \n" << Q << '\n';
	c.resize(2); c << -1, 1;
	std::cout << "c == \n" << c << '\n';
	x.resize(2);
	std::cout << "cqp::solve(Q,c,x) == " << cqp::solve(Q,c,x).transpose() << "\n";
	std::cout << "objective_function(x) == " << x.transpose()*(0.5*Q*x+c) << "\n";
	std::cout << "duality gap  == " << x.transpose()*(Q*x+c) << "\n";
	std::cout << "x == \n" << x << "\n\n";

	std::cout << "nonzero positive semidefinite Q, Q*x+c==0 has a solution\n";
	Q.resize(2,2); Q << 2, 0, 0, 0; //positive semidefinite Q, Q*x + c == 0 has solution
	std::cout << "Q == \n" << Q << '\n';
	c.resize(2); c << 2, 0;
	std::cout << "c == \n" << c << '\n';
	x.resize(2);
	std::cout << "cqp::solve(Q,c,x) == " << cqp::solve(Q,c,x).transpose() << "\n";
	std::cout << "objective_function(x) == " << x.transpose()*(0.5*Q*x+c) << "\n";
	std::cout << "duality gap  == " << x.transpose()*(Q*x+c) << "\n";
	std::cout << "x == \n" << x << "\n\n";

	std::cout << "nonzero positive semidefinite Q, Q*x+c==0 does not have a solution\n";
	Q.resize(2,2); Q << 2, 0, 0, 0; //positive semidefinite Q, Q*x + c == 0 has solution
	std::cout << "Q == \n" << Q << '\n';
	c.resize(2); c << 2, 2;
	std::cout << "c == \n" << c << '\n';
	x.resize(2);
	std::cout << "cqp::solve(Q,c,x) == " << cqp::solve(Q,c,x).transpose() << "\n";
	std::cout << "objective_function(x) == " << x.transpose()*(0.5*Q*x+c) << "\n";
	std::cout << "duality gap  == " << x.transpose()*(Q*x+c) << "\n";
	std::cout << "x == \n" << x << "\n\n";

	std::cout << "random\n";
	Q = cqp::matrix::Random(3,3)*111; Q = Q.transpose()*Q; //random
	std::cout << "Q == \n" << Q << '\n';
	c = cqp::matrix::Random(3,1)*111;
	std::cout << "c == \n" << c << '\n';
	x.resize(3);
	std::cout << "cqp::solve(Q,c,x) == " << cqp::solve(Q,c,x).transpose() << "\n";
	std::cout << "objective_function(x) == " << x.transpose()*(0.5*Q*x+c) << "\n";
	std::cout << "duality gap  == " << x.transpose()*(Q*x+c) << "\n";
	std::cout << "x == \n" << x << "\n\n";
}

void test_general_form() {
	test_problem_general_form the_problem(3);

	cqp::matrix Q, A, B;
	cqp::vector c, a, b;
	
	std::cout << "no variables, no constraints\n";
	Q = cqp::matrix::Zero(0,0); 
	A = cqp::matrix::Zero(0,0); 
	B = cqp::matrix::Zero(0,0); 
	c = cqp::vector::Zero(0);
	a = cqp::vector::Zero(0);
	b = cqp::vector::Zero(0);
	the_problem.set_problem(Q,c,A,a,B,b);
	the_problem.test_solve();

	std::cout << "no variables, infeasible eq constraints\n";
	Q = cqp::matrix::Zero(0,0); 
	A = cqp::matrix::Zero(1,0); 
	B = cqp::matrix::Zero(0,0); 
	c = cqp::vector::Zero(0);
	a = cqp::vector::Constant(1,1);
	b = cqp::vector::Zero(0);
	the_problem.set_problem(Q,c,A,a,B,b);
	the_problem.test_solve();

	std::cout << "no variables, feasible eq constraints\n";
	Q = cqp::matrix::Zero(0,0); 
	A = cqp::matrix::Zero(1,0); 
	B = cqp::matrix::Zero(0,0); 
	c = cqp::vector::Zero(0);
	a = cqp::vector::Zero(1);
	b = cqp::vector::Zero(0);
	the_problem.set_problem(Q,c,A,a,B,b);
	the_problem.test_solve();

	std::cout << "no variables, infeasible ineq constraints\n";
	Q = cqp::matrix::Zero(0,0); 
	A = cqp::matrix::Zero(0,0); 
	B = cqp::matrix::Zero(1,0); 
	c = cqp::vector::Zero(0);
	a = cqp::vector::Zero(0);
	b = cqp::vector::Constant(1,1);
	the_problem.set_problem(Q,c,A,a,B,b);
	the_problem.test_solve();

	std::cout << "no variables, feasible ineq constraints\n";
	Q = cqp::matrix::Zero(0,0); 
	A = cqp::matrix::Zero(0,0); 
	B = cqp::matrix::Zero(1,0); 
	c = cqp::vector::Zero(0);
	a = cqp::vector::Zero(0);
	b = cqp::vector::Zero(1);
	the_problem.set_problem(Q,c,A,a,B,b);
	the_problem.test_solve();

	std::cout << "unbounded \n";
	Q.resize(2,2); Q << 2, 0, 0, 0; 
	c.resize(2); c << 2, -2;
	A.resize(1,2); A << 2, 0; 
	a.resize(1); a << 4;
	B.resize(1,2); B << 0, 1; 
	b.resize(1); b << 0;
	the_problem.set_problem(Q,c,A,a,B,b);
	the_problem.test_solve();

	std::cout << "infeasible \n";
	Q.resize(2,2); Q << 2, 1, 1, 1; 
	c.resize(2); c << -1,1 ;
	A.resize(1,2); A << 1, 1; 
	a.resize(1); a << 1;
	B.resize(2,2); B << 0, 1, 1, 0; 
	b.resize(2); b << 0, 2;
	the_problem.set_problem(Q,c,A,a,B,b);
	the_problem.test_solve();

	std::cout << "box\n";
	the_problem.set_problem_box(16,16);
	the_problem.test_solve();

	cqp::index num_var, num_eq, num_ineq;
	double scale;
	Eigen::MatrixXd dmat;
	Eigen::VectorXd dvec;
	cqp::index num_row, num_col;
	Eigen::MatrixXi imat;
	int imatmax = 1;

	std::cout << "random\n";
	num_var = 4, num_eq = 1, num_ineq = 3;
	scale = 32;

	num_row = num_var;
	num_col = num_var;
	dmat = Eigen::MatrixXd::Random(num_row,num_col)*scale;
	imat = dmat.cast<int>();
	imat = imat.transpose()*imat;
	imatmax = std::max(1, imat.cwiseAbs().maxCoeff());
	imat *= static_cast<int>(scale);
	imat /= imatmax;
	Q = imat.cast<cqp::scalar>();

	dmat = (Eigen::MatrixXd::Random(num_row,num_col)+Eigen::MatrixXd::Constant(num_row,num_col,1))*scale/2;
	dmat = dmat.cwiseMax(Eigen::MatrixXd::Constant(num_row,num_col,1));
	imat = dmat.cast<int>();
	imat = imat.transpose()*imat;
	imatmax = std::max(1, imat.cwiseAbs().maxCoeff());
	imat *= static_cast<int>(scale);
	imat /= imatmax;
	imat = imat.cwiseMax(Eigen::MatrixXi::Constant(num_row,num_col,1)); //denominator >=1
	Q = Q.cwiseQuotient(imat.cast<cqp::scalar>());

	dvec = Eigen::VectorXd::Random(num_row)*scale;
	c = dvec.cast<int>().cast<cqp::scalar>();
	dvec = (Eigen::VectorXd::Random(num_row)+Eigen::VectorXd::Constant(num_row,1))*scale/2;
	dvec = dvec.cwiseMax(Eigen::VectorXd::Constant(num_row,1)); //denominator >=1
	c = c.cwiseQuotient(dvec.cast<int>().cast<cqp::scalar>());

	num_row = num_eq;
	num_col = num_var;
	dmat = Eigen::MatrixXd::Random(num_row,num_col)*scale;
	A = dmat.cast<int>().cast<cqp::scalar>();
	dmat = (Eigen::MatrixXd::Random(num_row,num_col)+Eigen::MatrixXd::Constant(num_row,num_col,1))*scale/2;
	dmat = dmat.cwiseMax(Eigen::MatrixXd::Constant(num_row,num_col,1)); //denominator >=1
	A = A.cwiseQuotient(dmat.cast<int>().cast<cqp::scalar>());

	dvec = Eigen::VectorXd::Random(num_row)*scale;
	a = dvec.cast<int>().cast<cqp::scalar>();
	dvec = (Eigen::VectorXd::Random(num_row)+Eigen::VectorXd::Constant(num_row,1))*scale/2;
	dvec = dvec.cwiseMax(Eigen::VectorXd::Constant(num_row,1)); //denominator >=1
	a = a.cwiseQuotient(dvec.cast<int>().cast<cqp::scalar>());

	num_row = num_ineq;
	num_col = num_var;
	dmat = Eigen::MatrixXd::Random(num_row,num_col)*scale;
	B = dmat.cast<int>().cast<cqp::scalar>();
	dmat = (Eigen::MatrixXd::Random(num_row,num_col)+Eigen::MatrixXd::Constant(num_row,num_col,1))*scale/2;
	dmat = dmat.cwiseMax(Eigen::MatrixXd::Constant(num_row,num_col,1)); //denominator >=1
	B = B.cwiseQuotient(dmat.cast<int>().cast<cqp::scalar>());

	dvec = Eigen::VectorXd::Random(num_row)*scale;
	b = dvec.cast<int>().cast<cqp::scalar>();
	dvec = (Eigen::VectorXd::Random(num_row)+Eigen::VectorXd::Constant(num_row,1))*scale/2;
	dvec = dvec.cwiseMax(Eigen::VectorXd::Constant(num_row,1)); //denominator >=1
	b = b.cwiseQuotient(dvec.cast<int>().cast<cqp::scalar>());

	the_problem.set_problem(Q,c,A,a,B,b);
	the_problem.test_solve();

	std::cout << "random2\n";
	num_var = 4, num_eq = 2, num_ineq = 2;
	scale = 37;

	num_row = num_var;
	num_col = num_var;
	dmat = Eigen::MatrixXd::Random(num_row,num_col)*scale;
	imat = dmat.cast<int>();
	imat = imat.transpose()*imat;
	imatmax = std::max(1, imat.cwiseAbs().maxCoeff());
	imat *= static_cast<int>(scale);
	imat /= imatmax;
	Q = imat.cast<cqp::scalar>();

	dmat = (Eigen::MatrixXd::Random(num_row,num_col)+Eigen::MatrixXd::Constant(num_row,num_col,1))*scale/2;
	dmat = dmat.cwiseMax(Eigen::MatrixXd::Constant(num_row,num_col,1));
	imat = dmat.cast<int>();
	imat = imat.transpose()*imat;
	imatmax = std::max(1, imat.cwiseAbs().maxCoeff());
	imat *= static_cast<int>(scale);
	imat /= imatmax;
	imat = imat.cwiseMax(Eigen::MatrixXi::Constant(num_row,num_col,1)); //denominator >=1
	Q = Q.cwiseQuotient(imat.cast<cqp::scalar>());

	dvec = Eigen::VectorXd::Random(num_row)*scale;
	c = dvec.cast<int>().cast<cqp::scalar>();
	dvec = (Eigen::VectorXd::Random(num_row)+Eigen::VectorXd::Constant(num_row,1))*scale/2;
	dvec = dvec.cwiseMax(Eigen::VectorXd::Constant(num_row,1)); //denominator >=1
	c = c.cwiseQuotient(dvec.cast<int>().cast<cqp::scalar>());

	num_row = num_eq;
	num_col = num_var;
	dmat = Eigen::MatrixXd::Random(num_row,num_col)*scale;
	A = dmat.cast<int>().cast<cqp::scalar>();
	dmat = (Eigen::MatrixXd::Random(num_row,num_col)+Eigen::MatrixXd::Constant(num_row,num_col,1))*scale/2;
	dmat = dmat.cwiseMax(Eigen::MatrixXd::Constant(num_row,num_col,1)); //denominator >=1
	A = A.cwiseQuotient(dmat.cast<int>().cast<cqp::scalar>());

	dvec = Eigen::VectorXd::Random(num_row)*scale;
	a = dvec.cast<int>().cast<cqp::scalar>();
	dvec = (Eigen::VectorXd::Random(num_row)+Eigen::VectorXd::Constant(num_row,1))*scale/2;
	dvec = dvec.cwiseMax(Eigen::VectorXd::Constant(num_row,1)); //denominator >=1
	a = a.cwiseQuotient(dvec.cast<int>().cast<cqp::scalar>());

	num_row = num_ineq;
	num_col = num_var;
	dmat = Eigen::MatrixXd::Random(num_row,num_col)*scale;
	B = dmat.cast<int>().cast<cqp::scalar>();
	dmat = (Eigen::MatrixXd::Random(num_row,num_col)+Eigen::MatrixXd::Constant(num_row,num_col,1))*scale/2;
	dmat = dmat.cwiseMax(Eigen::MatrixXd::Constant(num_row,num_col,1)); //denominator >=1
	B = B.cwiseQuotient(dmat.cast<int>().cast<cqp::scalar>());

	dvec = Eigen::VectorXd::Random(num_row)*scale;
	b = dvec.cast<int>().cast<cqp::scalar>();
	dvec = (Eigen::VectorXd::Random(num_row)+Eigen::VectorXd::Constant(num_row,1))*scale/2;
	dvec = dvec.cwiseMax(Eigen::VectorXd::Constant(num_row,1)); //denominator >=1
	b = b.cwiseQuotient(dvec.cast<int>().cast<cqp::scalar>());

	the_problem.set_problem(Q,c,A,a,B,b);
	the_problem.test_solve();

	std::cout << "random3\n";
	num_var = 4, num_eq = 3, num_ineq = 1;
	scale = 26;

	num_row = num_var;
	num_col = num_var;
	dmat = Eigen::MatrixXd::Random(num_row,num_col)*scale;
	imat = dmat.cast<int>();
	imat = imat.transpose()*imat;
	imatmax = std::max(1, imat.cwiseAbs().maxCoeff());
	imat *= static_cast<int>(scale);
	imat /= imatmax;
	Q = imat.cast<cqp::scalar>();

	dmat = (Eigen::MatrixXd::Random(num_row,num_col)+Eigen::MatrixXd::Constant(num_row,num_col,1))*scale/2;
	dmat = dmat.cwiseMax(Eigen::MatrixXd::Constant(num_row,num_col,1));
	imat = dmat.cast<int>();
	imat = imat.transpose()*imat;
	imatmax = std::max(1, imat.cwiseAbs().maxCoeff());
	imat *= static_cast<int>(scale);
	imat /= imatmax;
	imat = imat.cwiseMax(Eigen::MatrixXi::Constant(num_row,num_col,1)); //denominator >=1
	Q = Q.cwiseQuotient(imat.cast<cqp::scalar>());

	dvec = Eigen::VectorXd::Random(num_row)*scale;
	c = dvec.cast<int>().cast<cqp::scalar>();
	dvec = (Eigen::VectorXd::Random(num_row)+Eigen::VectorXd::Constant(num_row,1))*scale/2;
	dvec = dvec.cwiseMax(Eigen::VectorXd::Constant(num_row,1)); //denominator >=1
	c = c.cwiseQuotient(dvec.cast<int>().cast<cqp::scalar>());

	num_row = num_eq;
	num_col = num_var;
	dmat = Eigen::MatrixXd::Random(num_row,num_col)*scale;
	A = dmat.cast<int>().cast<cqp::scalar>();
	dmat = (Eigen::MatrixXd::Random(num_row,num_col)+Eigen::MatrixXd::Constant(num_row,num_col,1))*scale/2;
	dmat = dmat.cwiseMax(Eigen::MatrixXd::Constant(num_row,num_col,1)); //denominator >=1
	A = A.cwiseQuotient(dmat.cast<int>().cast<cqp::scalar>());

	dvec = Eigen::VectorXd::Random(num_row)*scale;
	a = dvec.cast<int>().cast<cqp::scalar>();
	dvec = (Eigen::VectorXd::Random(num_row)+Eigen::VectorXd::Constant(num_row,1))*scale/2;
	dvec = dvec.cwiseMax(Eigen::VectorXd::Constant(num_row,1)); //denominator >=1
	a = a.cwiseQuotient(dvec.cast<int>().cast<cqp::scalar>());

	num_row = num_ineq;
	num_col = num_var;
	dmat = Eigen::MatrixXd::Random(num_row,num_col)*scale;
	B = dmat.cast<int>().cast<cqp::scalar>();
	dmat = (Eigen::MatrixXd::Random(num_row,num_col)+Eigen::MatrixXd::Constant(num_row,num_col,1))*scale/2;
	dmat = dmat.cwiseMax(Eigen::MatrixXd::Constant(num_row,num_col,1)); //denominator >=1
	B = B.cwiseQuotient(dmat.cast<int>().cast<cqp::scalar>());

	dvec = Eigen::VectorXd::Random(num_row)*scale;
	b = dvec.cast<int>().cast<cqp::scalar>();
	dvec = (Eigen::VectorXd::Random(num_row)+Eigen::VectorXd::Constant(num_row,1))*scale/2;
	dvec = dvec.cwiseMax(Eigen::VectorXd::Constant(num_row,1)); //denominator >=1
	b = b.cwiseQuotient(dvec.cast<int>().cast<cqp::scalar>());

	the_problem.set_problem(Q,c,A,a,B,b);
	the_problem.test_solve();
}

void test_empty_standard_form() {
	test_problem test;
	cqp::matrix Q, A;
	cqp::vector c, b;

	Q = cqp::matrix::Identity(5,5);
	A = cqp::matrix::Zero(0,5);
	b = cqp::vector::Zero(0);
	c = cqp::matrix::Constant(5,1,1);
	test.set_problem(Q, c, A, b);
	test.test_solve();
}

void test_general_form(int, char**) {
}

void test_standard_form(int argc, char** argv) {
	enum type_problem {
		TYPE_PROBLEM_BOX,
		TYPE_PROBLEM_HS76,
		TYPE_PROBLEM_HS35,
		TYPE_PROBLEM_UNKNOWN 
	};

	enum type_test {
		TYPE_TEST_SOLVE,
		TYPE_TEST_HOMOTOPY,
		TYPE_TEST_UNKNOWN
	};

	bool print_info = false;

	bool have_read_kind_problem = false;
	bool have_read_kind_test = false;

	int argc_problem = 0, argc_test = 0;
	int index_problem = 0, index_test = argc;
	
	type_problem kind_problem = TYPE_PROBLEM_UNKNOWN;
	type_test kind_test = TYPE_TEST_SOLVE;
	for(int i = 1; i < argc; ++i) {
		std::string current_arg(argv[i]);
		if(is_double(current_arg)) {
			continue;
		} else if(current_arg == "box") { 
			if(have_read_kind_problem) {
				std::cout << "Error: duplicate problem type specification\n\n";
				print_info = true;
				break;
			}
			kind_problem = TYPE_PROBLEM_BOX;
			index_problem = i;
			have_read_kind_problem = true;
		} else if(current_arg == "hs76" || current_arg == "HS76") { 
			if(have_read_kind_problem) {
				std::cout << "Error: duplicate problem type specification\n\n";
				print_info = true;
				break;
			}
			kind_problem = TYPE_PROBLEM_HS76;
			index_problem = i;
			have_read_kind_problem = true;
		} else if(current_arg == "hs35" || current_arg == "HS35") { 
			if(have_read_kind_problem) {
				std::cout << "Error: duplicate problem type specification\n\n";
				print_info = true;
				break;
			}
			kind_problem = TYPE_PROBLEM_HS35;
			index_problem = i;
			have_read_kind_problem = true;
		} else if(current_arg == "homotopy" || current_arg == "homotopy_algorithm" || current_arg == "test_homotopy" || current_arg == "test_homotopy_algorithm" || 
				current_arg == "solve" || current_arg == "test_solve") {
			if(have_read_kind_test) {
				std::cout << "Error: duplicate test type specification\n\n";
				print_info = true;
				break;
			}

			if(current_arg == "solve" || current_arg == "test_solve") {
				kind_test = TYPE_TEST_SOLVE;
			} else { 
				kind_test = TYPE_TEST_HOMOTOPY;
			}
			index_test = i;
			have_read_kind_test = true;
		} else {
			std::cout << "Error: invalid argument \"" << current_arg << "\"\n\n";
			print_info = true;
			break;
		}
	}

	test_problem test;
	cqp::matrix Q, A;
	cqp::vector c, b, x;

	argc_test = argc-1-index_test;
	argc_problem = index_test-1-index_problem;
	switch(kind_problem) {
		case TYPE_PROBLEM_BOX:
			for(int i=index_problem+1; i<=index_problem+argc_problem; ++i) {
				if(!is_long(argv[i])) {
					std::cout << "Error: invalid argument \"" << argv[i] << "\" to \"test_problem::set_problem_box\"." 
						<< std::endl;
					print_info = true;
					break;
				}
			}

			switch(argc_problem) {
				case 3:
					test.set_problem_box(std::stoi(argv[index_problem+1]), std::stoi(argv[index_problem+2]), std::stoi(argv[index_problem+3]));
					break;
				case 2:
					test.set_problem_box(std::stoi(argv[index_problem+1]), std::stoi(argv[index_problem+2])); 
					break;
				case 1:
					test.set_problem_box(std::stoi(argv[index_problem+1]));
					break;
				default:
					test.set_problem_box();
			}
			break;
		case TYPE_PROBLEM_HS76:
			Q = cqp::matrix::Zero(7,7);
			Q.topLeftCorner(4,4) <<  2,  0, -1,  0,
						 0,  1,  0,  0,
						-1,  0,  2,  1,
						 0,  0,  1,  1;

			c = cqp::vector::Zero(7);
			c.head(4) << -1, -3, 1, -1;

			A = cqp::matrix::Zero(3,7);
			A.topLeftCorner(3,4) <<  1,  2,  1,  1,
						 3,  1,  2, -1,
						 0, -1, -4,  0;
			A.topRightCorner(3,3) = cqp::matrix::Identity(3,3);

			b.resize(3);
			b << 5, 4, -1.5;

			x.resize(7); //interior point known but not corresponding eta s.t. criteria of closeness satisfied
			x << 0.55, 0.55, 0.55, 0.55, 5-2.75, 4-2.75, -1.5+2.75;

			test.set_problem(Q, c, A, b, x);
			test.set_gamma(1e-8);
			test.set_beta(0.5);
			test.set_eta(cqp::sqrt(2*(x.asDiagonal()*(Q*x+c)).squaredNorm()));
			break;
		case TYPE_PROBLEM_HS35:
			Q.resize(4,4);
			Q <<  4,  2,  2,  0,
			      2,  4,  0,  0,
			      2,  0,  2,  0,
			      0,  0,  0,  0;

			c.resize(4);
			c << -8, -6, -4, 0;

			A.resize(1,4);
			A << -1, -1, -3, -1; 

			b.resize(1);
			b << -3;

			x.resize(4); 
			x << .75, .75, .375, .75;
			//eta >= ||X*(Q*x+c)||/beta

			test.set_problem(Q, c, A, b, x);
			test.set_gamma(1e-8);
			test.set_beta(0.5);
			test.set_eta(cqp::sqrt(2*(x.asDiagonal()*(Q*x+c)).squaredNorm()));
			break;
		default:
			print_info = true;
	}

	if((kind_test == TYPE_TEST_UNKNOWN) || print_info) {
		int w1 = 15, w2 = 25;
		std::cout << "usage:\n"
				<< "(1)\t" << "cqp_tests problem_name problem_args test_name test_args\n"
				<< "(2)\t" << "cqp_tests problem_args test_name test_args\n"
				<< "(3)\t" << "cqp_tests problem_name problem_args\n"
				<< "(3)\t" << "cqp_tests problem_args\n\n"

			<< std::setw(w1) << std::left << "problem_name" << std::setw(w2) << std::left << "problem_args" << "notes\n"
		      << std::setw(w1) << std::left << "box" << std::setw(w2) << std::left << "uint, uint, bool" << "bool value must be numeric (i.e. 0 or 1).\n" 
		      << std::setw(w1) << std::left << "hs76" << std::setw(w2) << std::left << " " << "\n" 
		      << std::setw(w1) << std::left << "hs35" << std::setw(w2) << std::left << " " << "\n\n" 

			<< std::setw(w1) << std::left << "test_name" << std::setw(w2) << std::left << "test_args" << "notes \n"
		      << std::setw(w1) << std::left << "solve" << std::setw(w2) << std::left << " " << "This is default test.\n"
		      << std::setw(w1) << std::left << "homotopy" << std::setw(w2) << std::left << "double, double, double" << "all args >= 0, 0 < second arg < 1\n"
		      << std::endl;

		return;
	}

	switch(kind_test) {
		case TYPE_TEST_HOMOTOPY:
			switch(argc_test) { //fall through intentional
				case 3:
					test.set_eta(std::stod(argv[index_test+3]));
				case 2:
					test.set_beta(std::stod(argv[index_test+2]));
				case 1:
					test.set_gamma(std::stod(argv[index_test+1]));
			}
			test.test_homotopy_algorithm();
			break;
		case TYPE_TEST_SOLVE: 
			test.test_solve();
			break;
		default:
			break;
	}
}

void test_empty_eigen() {
	cqp::matrix M1, M2, M3, M4, M5, M6;
	cqp::vector v1, v2, v3, v4;
	v1 = cqp::vector::Zero(5);
	v1 << 0, 1, 2, 3, 4;
	v2 = cqp::vector::Zero(0);

	M1 = cqp::matrix::Zero(0,5);
	M2 = cqp::matrix::Zero(5,0);
	M3 = cqp::matrix::Zero(0,0);
	M4 = cqp::matrix::Zero(0,0);
	M5 = cqp::matrix::Zero(0,0);
	M6 = cqp::matrix::Zero(0,0);

	M3 = M1+M1;
	M4 = 3.2*M1;
	M5 = M1*M2;
	M6 = M2*M1;

	v3 = M1*v1;
	v4 = M2*v2;

	std::cout << "(M1.rows(), M1.cols()) == " << '(' << M1.rows() << ", " << M1.cols() << ')' << '\n';
	std::cout << "(M2.rows(), M2.cols()) == " << '(' << M2.rows() << ", " << M2.cols() << ')' << '\n';
	std::cout << "((M1+M1).rows(), (M1+M1).cols()) == " << '(' << M3.rows() << ", " << M3.cols() << ')' << '\n';
	std::cout << "((3.2*M1).rows(), (3.2*M1).cols()) == " << '(' << M4.rows() << ", " << M4.cols() << ')' << '\n';
	std::cout << "((M1*M2).rows(), (M1*M2).cols()) == " << '(' << M5.rows() << ", " << M5.cols() << ')' << '\n';
	std::cout << "((M2*M1).rows(), (M2*M1).cols()) == " << '(' << M6.rows() << ", " << M6.cols() << ')' << '\n';
	std::cout << "(v1.rows(), v1.cols()) == " << '(' << v1.rows() << ", " << v1.cols() << ')' << '\n';
	std::cout << "(v2.rows(), v2.cols()) == " << '(' << v2.rows() << ", " << v2.cols() << ')' << '\n';
	std::cout << "(M1*v1.rows(), M1*v1.cols()) == " << '(' << v3.rows() << ", " << v3.cols() << ')' << '\n';
	std::cout << "(M2*v2.rows(), M2*v2.cols()) == " << '(' << v4.rows() << ", " << v4.cols() << ')' << '\n';

	std::cout << "M1 == \n" << M1 << '\n';
	std::cout << "M2 == \n" << M2 << '\n';
	std::cout << "M1+M1 == \n" << M3 << '\n';
	std::cout << "3.2*M1== \n" << M4 << '\n';
	std::cout << "M1*M2 == \n" << M5 << '\n';
	std::cout << "M2*M1 == \n" << M6 << '\n';
	std::cout << "M1*v1 == \n" << v3 << '\n';
	std::cout << "M2*v2 == \n" << v4 << '\n';
}

void test_FullPivLU(const Eigen::Matrix<mpq_class, Eigen::Dynamic, Eigen::Dynamic> &M) {
	std::cout << "\n****************************test_FullPivLU***************************\n" << std::endl;
	int m = M.rows(); int n = M.cols();

	Eigen::FullPivLU<Eigen::Matrix<mpq_class, Eigen::Dynamic, Eigen::Dynamic>> LU(M);
	std::cout << "M == \n" << M << '\n' << std::endl;
	std::cout << "LU == \n" << LU.matrixLU() << '\n' << std::endl;
	Eigen::Matrix<mpq_class, Eigen::Dynamic, Eigen::Dynamic> L(m,m); L = Eigen::Matrix<mpq_class, Eigen::Dynamic, Eigen::Dynamic>::Identity(m,m);
	L.block(0,0,m,std::min(m,n)).triangularView<Eigen::StrictlyLower>() = LU.matrixLU().block(0,0,m,std::min(m,n));
	std::cout << "L == \n" << L << '\n' << std::endl;
	Eigen::Matrix<mpq_class, Eigen::Dynamic, Eigen::Dynamic> U(m,n); U = Eigen::Matrix<mpq_class, Eigen::Dynamic, Eigen::Dynamic>::Constant(m,n,0); 
	U = LU.matrixLU().triangularView<Eigen::Upper>();
	std::cout << "U == \n" << U << '\n' << std::endl;
	Eigen::Matrix<mpq_class, Eigen::Dynamic, Eigen::Dynamic> P(LU.permutationP()*Eigen::Matrix<mpq_class, Eigen::Dynamic, Eigen::Dynamic>::Identity(m,m));
	std::cout << "P == \n" <<  P << '\n' << std::endl;
	std::cout << "LU.permutationP().indices().size() == \n" << LU.permutationP().indices().size() << '\n' << std::endl;
	std::cout << "LU.permutationP().indices() == \n" << LU.permutationP().indices() << '\n' << std::endl;
	std::cout << "P^-1 == \n" <<  P.transpose() << '\n' << std::endl;
//	Eigen::PermutationMatrix<-1, -1, int> Pinv(LU.permutationP().inverse());
	std::cout << "Eigen::PermutationMatrix<-1, -1, int>(LU.permutationP().inverse()).indices().size() == \n" << Eigen::PermutationMatrix<-1, -1, int>(LU.permutationP().inverse()).indices().size() << '\n' << std::endl;
	std::cout << "Eigen::PermutationMatrix<-1, -1, int>(LU.permutationP().inverse()).indices() == \n" << Eigen::PermutationMatrix<-1, -1, int>(LU.permutationP().inverse()).indices() << '\n' << std::endl;

	Eigen::Matrix<mpq_class, Eigen::Dynamic, Eigen::Dynamic> Q(LU.permutationQ()*Eigen::Matrix<mpq_class, Eigen::Dynamic, Eigen::Dynamic>::Identity(n,n));
	std::cout << "Q == \n" <<  Q << '\n' << std::endl;
	std::cout << "LU.permutationQ().indices().size() == \n" << LU.permutationQ().indices().size() << '\n' << std::endl;
	std::cout << "LU.permutationQ().indices() == \n" << LU.permutationQ().indices() << '\n' << std::endl;
	std::cout << "Q^-1 == \n" <<  Q.transpose() << '\n' << std::endl;
	std::cout << "Eigen::PermutationMatrix<-1, -1, int>(LU.permutationQ().inverse()).indices().size() == \n" << Eigen::PermutationMatrix<-1, -1, int>(LU.permutationQ().inverse()).indices().size() << '\n' << std::endl;
	std::cout << "Eigen::PermutationMatrix<-1, -1, int>(LU.permutationQ().inverse()).indices() == \n" << Eigen::PermutationMatrix<-1, -1, int>(LU.permutationQ().inverse()).indices() << '\n' << std::endl;

	std::cout << "L*U == \n" << L*U << '\n' << std::endl;	
	std::cout << "P^-1*L*U*Q^-1 == \n" << P.transpose()*L*U*Q.transpose() << '\n' << std::endl;

	std::cout << "LU.image(M) == \n" << LU.image(M) << '\n' << std::endl;
	std::cout << "LU.kernel() == \n" << LU.kernel() << '\n' << std::endl;
	std::cout << "LU.nonzeroPivots() == \n" << LU.nonzeroPivots() << '\n' << std::endl;
	std::cout << "LU.maxPivot() == \n" << LU.maxPivot() << '\n' << std::endl;
	std::cout << "LU.threshold() == \n" << LU.threshold() << '\n' << std::endl;
	std::cout << "LU.maxPivot()*LU.threshold() == \n" << LU.maxPivot()*LU.threshold() << '\n' << std::endl;
	
	Eigen::Matrix<mpq_class, Eigen::Dynamic, 1> x, y, num, den;
	num = (1000*Eigen::Matrix<double, Eigen::Dynamic, 1>::Random(m)).cast<int>().cast<mpq_class>();
	den = (Eigen::Matrix<double, Eigen::Dynamic, 1>::Constant(m,1)+999*(Eigen::Matrix<double, Eigen::Dynamic, 1>::Random(m)+Eigen::Matrix<double, Eigen::Dynamic, 1>::Constant(m,1))/2).cast<int>().cast<mpq_class>();
	y = num.cwiseQuotient(den);
	std::cout << "y == \n" << y << "\n";
	x = LU.solve(y);
	std::cout << "x == LU.solve(y) == \n" << x << "\n";
	std::cout << "M*x-y == \n" << M*x-y << "\n";
}

void test_FullPivLU(void) {
	Eigen::Matrix<mpq_class, Eigen::Dynamic, Eigen::Dynamic> M(5,3); M << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15;
	test_FullPivLU(M);

	M.resize(3,4); M << 1,1,1,1,
				 1,1,1,2,
				 1,1,1,3;
	test_FullPivLU(M);

			    M << 8,   4,    2,    1,
				 4,   2,    1,  1.5,
				 1, 0.5, 0.25, 1.25;
	test_FullPivLU(M);

	M.resize(3,6);      M <<  1,  2,  3,  4,  5,  6,
				  7,  8,  9, 10, 11, 12,
			         13, 14, 15, 16, 17, 18;	 
	test_FullPivLU(M);
}

void test_echelon(const cqp::matrix &M) {
	std::cout << "\n****************************test_echelon****************************\n" << std::endl;
	cqp::matrix red(M);
	cqp::scalar piv_max = 0; //max absolute value of a pivot.
	cqp::index ind_row = 0; cqp::index ind_col = 0;
	cqp::index rows = red.rows(); cqp::index cols = red.cols();
	while(ind_row < rows && ind_col < cols) {
		cqp::index ind_max = 0;
		cqp::scalar val_max = red.col(ind_col).tail(rows-ind_row).cwiseAbs().maxCoeff(&ind_max); //can replace cwiseAbs() with different function to determine "largest" pivot
		ind_max += ind_row; //since using tail() offsets index
		if(val_max == 0) {
			++ind_col;
			continue;
		}
		if(val_max > piv_max) {
			piv_max = val_max;
		}
		if(ind_max != ind_row) {
			red.row(ind_row).swap(red.row(ind_max));
		}
		cqp::index rrows = rows-1-ind_row;
		cqp::index rcols = cols-ind_col;
		if(0<rrows) {
			red.bottomRightCorner(rrows,rcols) -= red.col(ind_col).tail(rrows)*(red.row(ind_row).tail(rcols)/red(ind_row, ind_col));
		}
		++ind_row;
		++ind_col;
	}
	
	std::cout << "M == \n" << M << '\n' << std::endl;
	std::cout << "red == \n" << red << '\n' << std::endl;
	std::cout << "piv_max == \n" << piv_max << '\n' << std::endl;
	std::cout << "threshold == \n" << Eigen::NumTraits<cqp::scalar>::epsilon()*std::min(rows,cols) << '\n' << std::endl;
	std::cout << "piv_max*threshold == \n" << piv_max*Eigen::NumTraits<cqp::scalar>::epsilon()*std::min(rows,cols) << '\n' << std::endl;
}

void test_echelon(void) {
	cqp::matrix M(3,4); M << 1,1,1,1,
				 1,1,1,2,
				 1,1,1,3;
	test_echelon(M);

			    M << 8,   4,    2,    1,
				 4,   2,    1,  1.5,
				 1, 0.5, 0.25, 1.25;
	test_echelon(M);

	M.resize(3,6);      M <<  1,  2,  3,  4,  5,  6,
				  7,  8,  9, 10, 11, 12,
			         13, 14, 15, 16, 17, 18;	 
	test_echelon(M);
}

void test_purify(void) {
	std::cout << "\n****************************test_purify****************************\n" << std::endl;
	std::cout.precision(17);

	//test finding point in a rectangle closest to a given point.
	int m = 5; int n = 2*m; //x_{m+1}, ..., x_{2*m} are slack variables.

	cqp::scalar low_b, high_b; low_b = 1; high_b = 16384;
	cqp::vector b = (cqp::matrix::Constant(m,1,1)+cqp::matrix::Random(m,1))*0.5*(high_b-low_b)+cqp::matrix::Constant(m,1,low_b);  //b>0, b(i) gives upper bound on x(i).
	
	cqp::scalar low_target, high_target; low_target = 0; high_target = b.minCoeff(); //If target is in the rectangle then optimal x should have first m components equal to target.
	cqp::vector target = (cqp::matrix::Constant(m,1,1)+cqp::matrix::Random(m,1))*0.5*(high_target-low_target)+cqp::matrix::Constant(m,1,low_target);//target point 

	cqp::vector c(n); c << (-2*target), cqp::vector::Zero(m);

	cqp::matrix Q = cqp::matrix::Zero(n,n); Q.block(0,0,m,m) = 2*cqp::matrix::Identity(m,m);

	cqp::matrix A(m,n); A << cqp::matrix::Identity(m,m), cqp::matrix::Identity(m,m);

	cqp::vector x(n); x << b/2.0, b/2.0; //initial point center of rectangle.
	cqp::vector y(m); //for dual part of solution

	std::cout << "target == \n" << target << std::endl;
	std::cout << "\nQ == \n" << Q << "\n\nc == \n" << c << "\n\nA == \n" << A << "\n\nb == \n" << b << "\n\nx == \n" << x;

	cqp::scalar beta, gamma, eta;
	beta = 0.5; gamma = 1e-1; eta = cqp::sqrt(((x.asDiagonal())*(Q*x+c)).squaredNorm())/beta;
	cqp::scalar epsilon = 1e-1;

	int format_width = 11; 
	std::cout << "\n\n" << std::setw(format_width) << std::left << "beta == " << beta 
		  << "\n" << std::setw(format_width) << std::left << "gamma == " << gamma 
		  << "\n" << std::setw(format_width) << std::left << "eta == " << eta 
		  << "\n" << std::setw(format_width) << std::left << "epsilon == " << epsilon << std::endl;

	std::cout << "\nCalling \"cqp::homotopy_algorithm(Q, c, A, b, x, beta, gamma, eta, x, y)\"." << std::endl;
	cqp::homotopy_algorithm(Q, c, A, b, x, beta, gamma, eta, x, y);
	std::cout << "Call complete." << std::endl;

	std::cout << "\nx == \n" << x;
	std::cout << "\n\ny == \n" << y;
	cqp::vector z = (Q*x+c-A.transpose()*y);
	std::cout << "\n\nz == Q*x+c-A.transpose()*y == \n" << z;
	cqp::vector w = A*x - b;
	std::cout << "\n\nw == A*x-b == \n" << w;
	
	cqp::vector pure_x = x;
	cqp::vector pure_y = y;
	std::cout << "\n\nCalling \"cqp::purify(Q, c, A, b, x, y, epsilon, pure_x, pure_y)\"." << std::endl;
	cqp::purify(Q, c, A, b, x, y, epsilon, pure_x, pure_y);
	std::cout << "Call complete." << std::endl;

	std::cout << "\npure_x == \n" << pure_x;
	std::cout << "\n\npure_y == \n" << pure_y;
	cqp::vector pure_z = (Q*pure_x+c-A.transpose()*pure_y);
	std::cout << "\n\npure_z == Q*pure_x+c-A.transpose()*pure_y == \n" << pure_z;
	cqp::vector pure_w = A*pure_x - b;
	std::cout << "\n\npure_w == A*pure_x-b == \n" << pure_w;

	cqp::vector optimal_x(n); for(int i=0; i<m; ++i) { optimal_x(i) = ((target(i) < 0)?0:((target(i) <= b(i))?target(i):b(i))); optimal_x(m+i) = b(i)-optimal_x(i); }
	std::cout << "\n\noptimal_x == \n" << optimal_x;
	cqp::vector optimal_y = (optimal_x.asDiagonal()*A.transpose()).fullPivLu().solve(optimal_x.asDiagonal()*(Q*optimal_x+c));
	std::cout << "\n\noptimal_y == \n" << optimal_y;	
	cqp::vector optimal_z = (Q*optimal_x+c-A.transpose()*optimal_y);
	std::cout << "\n\noptimal_z == Q*optimal_x+c-A.transpose()*optimal_y == \n" << optimal_z;		
	cqp::vector optimal_w = A*optimal_x - b;
	std::cout << "\n\noptimal_w == A*optimal_x-b == \n" << optimal_w;

	format_width = 51;
	cqp::scalar gap = x.transpose()*z;
	cqp::scalar pure_gap = pure_x.transpose()*pure_z;
	cqp::scalar optimal_gap = optimal_x.transpose()*optimal_z;
	std::cout << "\n\n" << std::setw(format_width) << std::left << "predicted_gap == n*gamma == " << n*gamma;
	std::cout << "\n" << std::setw(format_width) << std::left << "gap == x^T*z == " << gap;
	std::cout << "\n" << std::setw(format_width) << std::left << "pure_gap == pure_x^T*pure_z == " << pure_gap;
	std::cout << "\n" << std::setw(format_width) << std::left << "optimal_gap == optimal_x^T*optimal_z == " << optimal_gap;

	cqp::scalar eq_gap = w.transpose()*y;
	cqp::scalar pure_eq_gap = pure_w.transpose()*pure_y;
	cqp::scalar optimal_eq_gap = optimal_w.transpose()*optimal_y;
	std::cout << "\n\n" << std::setw(format_width) << std::left << "eq_gap == w^T*y == " << eq_gap;
	std::cout << "\n" << std::setw(format_width) << std::left << "pure_eq_gap == pure_w^T*pure_y == " << pure_eq_gap;
	std::cout << "\n" << std::setw(format_width) << std::left << "optimal_eq_gap == optimal_w^T*optimal_y == " << optimal_eq_gap;

	std::cout << "\n\n" << std::setw(format_width) << std::left << "total_gap == gap+eq_gap == " << gap+eq_gap;
	std::cout << "\n" << std::setw(format_width) << std::left << "pure_total_gap == pure_gap+pure_eq_gap == " << pure_gap+pure_eq_gap;
	std::cout << "\n" << std::setw(format_width) << std::left << "optimal_total_gap == optimal_gap+optimal_eq_gap == " << optimal_gap+optimal_eq_gap;

	format_width = 22;
	cqp::scalar f_x = (x.transpose()*0.5*Q*x + c.transpose()*x).value();
	std::cout << "\n\n" << std::setw(format_width) << std::left << "f_x == " << f_x;
	cqp::scalar f_pure_x = (pure_x.transpose()*0.5*Q*pure_x + c.transpose()*pure_x).value();
	std::cout << "\n" << std::setw(format_width) << std::left << "f_pure_x == " << f_pure_x;
	cqp::scalar f_optimal_x = (optimal_x.transpose()*0.5*Q*optimal_x + c.transpose()*optimal_x).value();
	std::cout << "\n" << std::setw(format_width) << std::left << "f_optimal_x == " << f_optimal_x;

	format_width = 36;
	std::cout << "\n\n" << std::setw(format_width) << std::left << "f_x - f_pure_x == " << f_x - f_pure_x;
	std::cout << "\n" << std::setw(format_width) << std::left << "f_x - f_optimal_x == " << f_x - f_optimal_x;
	std::cout << "\n" << std::setw(format_width) << std::left << "f_pure_x - f_optimal_x == " << f_pure_x - f_optimal_x;

	std::cout << "\n\n" << std::setw(format_width) << std::left << "f_x + target^T * target == " << f_x + (target.transpose() * target).value();
	std::cout << "\n" << std::setw(format_width) << std::left << "f_pure_x + target^T * target == " << f_pure_x + (target.transpose() * target).value();
	std::cout << "\n" << std::setw(format_width) << std::left << "f_optimal_x + target^T * target == " << f_optimal_x + (target.transpose() * target).value() << std::endl;
}
