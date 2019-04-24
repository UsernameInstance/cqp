#ifndef MISC_TESTS_HXX
#define MISC_TESTS_HXX
#include "cqp.hxx"
#include "test_problem.hxx"
#include "test_problem_general_form.hxx"
#include <iostream>
#include <iomanip>
#include <string>
#include <cstdlib>

void test_FullPivLU(const Eigen::Matrix<mpq_class, Eigen::Dynamic, Eigen::Dynamic> &);
void test_FullPivLU(void);
void test_echelon(const cqp::matrix &);
void test_echelon(void);
void test_purify(void);
void test_unconstrained(); //tests for unconstrained problem.
void test_standard_form(int, char**); //tests for problem in standard form, i.e. constraints are in the form {A*x=b , x>=0}
void test_general_form(); //tests for problem in general form, i.e. constraints are in the form {A*x=a , B*x>=b}
void test_empty_eigen();
void test_empty_standard_form();

inline bool is_double(std::string input) {
	char *p;
	std::strtod(input.c_str(), &p);
	return *p==0; //p points to '\0' if success.
}

inline bool is_long(std::string input) {
	char *p;
	std::strtol(input.c_str(), &p, 10);
	return *p==0; //p points to '\0' if success.
}

#endif
