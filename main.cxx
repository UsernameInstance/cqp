#include "misc_tests.hxx"
void test_mpq_class();
void test_NumTraits_mpf_class();
void test_linear_algebra_mpf_class();
void test_simple_fraction();
void test_sqrt();

int main(int argc, char* argv[]) { 
	/* tests
	test_FullPivLU();
	test_echelon();
	test_purify();
	test_double_scale_factor();
	test_long_double_scale_factor();
	test_empty_eigen();
	test_empty_standard_form();
	test_standard_form(argc, argv);
	test_unconstrained();
	test_general_form();
	test_mpq_class();
	test_NumTraits_mpf_class();
	test_linear_algebra_mpf_class();
	mpf_set_default_prec(128);
	test_general_form();
	test_simple_fraction();
	test_sqrt();
	*/

	mpf_set_default_prec(128);
	if(argc > 1) {
		std::string first_arg(argv[1]);
		if(first_arg == "general") {
			test_general_form();
		} else {
			test_standard_form(argc, argv);
		}
	} else {
		test_standard_form(argc, argv);
	}

	return 0;
}

void test_sqrt() {
	mpq_class input;

	for(int num = 1; num < 4; ++num) {
		for(int den = 1; den<4; ++den) {
			input = mpq_class(num,den + (den>=num?1:0));
			for(int k = 1; k < 100; k *= 10) {
				mpq_class eps(1,k);
				std::cout << "cqp::sqrt(" << input << ", " << eps << ") == " << cqp::sqrt(input, eps) << "\t";
			}
		}
		std::cout << "\n";
	}
}

void test_simple_fraction(cqp::scalar a, cqp::scalar b) {
	a.canonicalize(); b.canonicalize();
	std::cout << "cqp::simple_fraction(" << a << ", " << b << ") == " << cqp::simple_fraction(a,b);
}

void test_simple_fraction() {
	cqp::scalar a, b;
	int num, den, num2, den2;
	for(int i=-3; i<4; ++i) {
		num = 5*i*i*i+2*i+cqp::cqp_sign(i)*127;
		int l = std::max(2,std::abs(i));
		for(int j=l; j<l+3; ++j) {
			den = j*(j-l+1)+cqp::cqp_mod(41*j*j*j-20*j*j+j+77,l*l+2);
			den2 = (2*den+(1-2*cqp::cqp_mod(j,2))*den+1)/2;
			int low = (cqp::cqp_mod(num,den)*den2+den2-1)/den;
			num2 = cqp::cqp_div(num,den)*den2+((low+den2+1)/2);
			a = cqp::scalar(num, den);
			b = cqp::scalar(num2, den2);
			test_simple_fraction(a, b);
			std::cout << "\t\t";
		}
		std::cout << '\n';
	}
}

void test_mpq_class() {
	std::cout << "sqrt(mpq_class(100,121)) == " << cqp::sqrt(mpq_class(100,121)) << "\n";
	std::cout << "sqrt(mpq_class(99,120)) == " << cqp::sqrt(mpq_class(99,120)) << "\n";
	std::cout << "sqrt(mpq_class(0,121)) == " << cqp::sqrt(mpq_class(0,121)) << "\n";
	std::cout << "sqrt(mpq_class(-1,121)) == " << cqp::sqrt(mpq_class(-1,121)) << "\n\n";

	std::cout << "sqrt(mpq_class(2)) == " << cqp::sqrt(mpq_class(2)) << "\n";
	std::cout << "sqrt(mpq_class(2),1) == " << cqp::sqrt(mpq_class(2),1) << "\n";
	std::cout << "sqrt(mpq_class(2),2) == " << cqp::sqrt(mpq_class(2),2) << "\n";
	std::cout << "sqrt(mpq_class(2),4) == " << cqp::sqrt(mpq_class(2),4) << "\n";
	std::cout << "sqrt(mpq_class(2),8) == " << cqp::sqrt(mpq_class(2),8) << "\n";
}

void test_NumTraits_mpf_class() {
	std::cout << "************************************test_mpf_class************************************" << "\n";
	std::cout << "mpf_get_default_prec() == " << mpf_get_default_prec() << "\n";
	std::cout << "Eigen::NumTraits<mpf_class>::epsilon() == " << Eigen::NumTraits<mpf_class>::epsilon() << "\n";
	std::cout << "Eigen::NumTraits<mpf_class>::dummy_precision() == " << Eigen::NumTraits<mpf_class>::dummy_precision() << "\n";
	std::cout << "Eigen::NumTraits<mpf_class>::digits10() == " << Eigen::NumTraits<mpf_class>::digits10() << "\n\n";
	
	std::cout << "mpf_set_default_prec(128)\n\n"; mpf_set_default_prec(128);

	std::cout << "mpf_get_default_prec() == " << mpf_get_default_prec() << "\n";
	std::cout << "Eigen::NumTraits<mpf_class>::epsilon() == " << Eigen::NumTraits<mpf_class>::epsilon() << "\n";
	std::cout << "Eigen::NumTraits<mpf_class>::dummy_precision() == " << Eigen::NumTraits<mpf_class>::dummy_precision() << "\n";
	std::cout << "Eigen::NumTraits<mpf_class>::digits10() == " << Eigen::NumTraits<mpf_class>::digits10() << "\n\n";

	std::cout << "std::numeric_limits<long double>::radix == " << std::numeric_limits<long double>::radix << "\n";
	std::cout << "std::numeric_limits<long double>::min_exponent == " << std::numeric_limits<long double>::min_exponent << "\n";
	std::cout << "std::numeric_limits<long double>::max_exponent == " << std::numeric_limits<long double>::max_exponent << "\n";
	std::cout << "std::numeric_limits<long double>::epsilon() == " << std::numeric_limits<long double>::epsilon() << "\n";
	std::cout << "Eigen::NumTraits<long double>::epsilon() == " << Eigen::NumTraits<long double>::epsilon() << "\n";
	std::cout << "Eigen::NumTraits<long double>::dummy_precision() == " << Eigen::NumTraits<long double>::dummy_precision() << "\n";
	std::cout << "Eigen::NumTraits<long double>::digits10() == " << Eigen::NumTraits<long double>::digits10() << "\n\n";

	std::cout << "std::numeric_limits<double>::radix == " << std::numeric_limits<double>::radix << "\n";
	std::cout << "std::numeric_limits<double>::min_exponent == " << std::numeric_limits<double>::min_exponent << "\n";
	std::cout << "std::numeric_limits<double>::max_exponent == " << std::numeric_limits<double>::max_exponent << "\n";
	std::cout << "std::numeric_limits<double>::epsilon() == " << std::numeric_limits<double>::epsilon() << "\n";
	std::cout << "Eigen::NumTraits<double>::epsilon() == " << Eigen::NumTraits<double>::epsilon() << "\n";
	std::cout << "Eigen::NumTraits<double>::dummy_precision() == " << Eigen::NumTraits<double>::dummy_precision() << "\n";
	std::cout << "Eigen::NumTraits<double>::digits10() == " << Eigen::NumTraits<double>::digits10() << "\n\n";

	std::cout << "1/mpf_class(2) == " << 1/mpf_class(2) << "\n";
	std::cout << "mpf_class(-1/2.0) == " << mpf_class(-1/2.0) << "\n";
	std::cout << "sqrt(mpf_class(99)) == " << sqrt(mpf_class(99)) << "\n";
	std::cout << "sqrt(mpf_class(100)) == " << sqrt(mpf_class(100)) << "\n";
	std::cout << "sqrt(mpf_class(101)) == " << sqrt(mpf_class(101)) << "\n";
}

void test_linear_algebra_mpf_class() {
	cqp::float_matrix X = cqp::float_matrix::Zero(2,2);
	std::cout << "X == \n" << X << "\n";
	X << 1,2,3,4;
	std::cout << "X == \n" << X << "\n";
	cqp::float_vector v = cqp::float_vector::Zero(2);
	std::cout << "v == \n" << v << "\n";
	v << 1.7,8.12;
	std::cout << "v == \n" << v << "\n";

	X(0,0) = (1/1.7)*(1/1.7);
	X(1,1) = (1/v(0))*(1/v(0));
	std::cout << "X == \n" << X << "\n";
}


