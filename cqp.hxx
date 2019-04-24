#ifndef CQP_HXX
#define CQP_HXX
#include <Eigen/Core> //for Vector and Matrix class
#include <Eigen/LU> //For fullPivLu().solve()
#include <gmpxx.h>

#define USE_FLOATING_POINT_HOMOTOPY_ALGORITHM 0 /*if 1 calls to solve() will use floating point version homotopy algorithm otherwise rational data type version used*/

#define LOG_SOLVE_STANDARD 0
#define LOG_SOLVE_GENERAL 0
#define LOG_HOMOTOPY_ALGORITHM 0

#if LOG_SOLVE_STANDARD || LOG_HOMOTOPY_ALGORITHM || LOG_SOLVE_GENERAL
#include <iostream>
#include <iomanip>
#endif

namespace Eigen { //modified code from Eigen documentation for mpq_class
	template<> struct NumTraits<mpq_class> : GenericNumTraits<mpq_class> {
		typedef mpq_class Real;
		typedef mpq_class NonInteger;
		typedef mpq_class Nested;
		static inline Real epsilon() { return 0; }
		static inline Real dummy_precision() { return 0; }
		static inline int digits10() { return 0; }
		enum {
			IsInteger = 0,
			IsSigned = 1,
			IsComplex = 0,
			RequireInitialization = 1,
			ReadCost = 6,
			AddCost = 350,
			MulCost = 100
		};
	};
	namespace internal {
		template<> struct scalar_score_coeff_op<mpq_class> {
			struct result_type {
				std::size_t len;
				result_type(int i = 0) : len(i) {} // Eigen uses Score(0) and Score()
				result_type(mpq_class const& q) :
					len(mpz_size(q.get_num_mpz_t())+
							mpz_size(q.get_den_mpz_t())-1) {}
				friend bool operator<(result_type x, result_type y) {
					// 0 is the worst possible pivot
					if (x.len == 0) return y.len > 0;
					if (y.len == 0) return false;
					// Prefer a pivot with a small representation
					return x.len > y.len;
				}
				friend bool operator==(result_type x, result_type y) {
					// Only used to test if the score is 0
					return x.len == y.len;
				}
				friend bool operator>(const result_type& x, const result_type& y)  { return y < x; }
				friend bool operator<=(const result_type& x, const result_type& y) { return !(y < x); }
				friend bool operator>=(const result_type& x, const result_type& y) { return !(x < y); }
				friend bool operator!=(const result_type& x, const result_type& y) { return !(x == y); }
			};
			result_type operator()(mpq_class const& x) const { return x; }
		};
	}

	template<> struct NumTraits<mpf_class> : GenericNumTraits<mpf_class> {
		typedef mpf_class Real;
		typedef mpf_class NonInteger;
		typedef mpf_class Nested;
		static inline Real epsilon() { Real retval(2); mpf_div_2exp(retval.get_mpf_t(), retval.get_mpf_t(), mpf_get_default_prec()); return retval; }
		static inline Real dummy_precision() { Real retval(2); mpf_div_2exp(retval.get_mpf_t(), retval.get_mpf_t(), 25*mpf_get_default_prec()/32); return retval; }
		static inline int digits10() { Real retval(mpf_get_default_prec()); retval = 0.30103*retval; if(mpf_fits_sint_p(retval.get_mpf_t())) { return retval.get_si(); } else { return std::numeric_limits<int>::max(); } }
		enum {
			IsInteger = 0,
			IsSigned = 1,
			IsComplex = 0,
			RequireInitialization = 1,
			ReadCost = 1,
			AddCost = 3,
			MulCost = 4 
		};
	};
} //end Eigen namespace


namespace cqp {
	typedef Eigen::Index index; //Eigen index type. Set to std::ptrdiff_t by default.
	typedef Eigen::Matrix<index, Eigen::Dynamic, 1> index_vector;

	typedef mpq_class scalar; //exact arithmetic types
	typedef Eigen::Matrix<scalar, Eigen::Dynamic, 1> vector;
	typedef Eigen::Matrix<scalar, 2, 1> vector2;
	typedef Eigen::Matrix<scalar, 3, 1> vector3;
	typedef Eigen::Matrix<scalar, 1, Eigen::Dynamic> row_vector;
	typedef Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic> matrix;

	typedef mpf_class float_scalar; //floating point types
	typedef Eigen::Matrix<float_scalar, Eigen::Dynamic, Eigen::Dynamic> float_matrix;
	typedef Eigen::Matrix<float_scalar, Eigen::Dynamic, 1> float_vector;

	template <typename T>
	T cqp_abs(T x) {
		return (x<0?-x:x);
	}

	template <typename T>
	T cqp_sign(T x) {
		return (x<0?-1:(x==0?0:1));
	}

	template <typename T>
	T cqp_div(T x, T y) { //integral types x and y with default truncated division returns floor(x/y)
		if(x==0 || y==0) {
			return 0;
		} else if((x<0 && y>0) || (x>0 && y<0)) {
			return -((cqp_abs(y)+cqp_abs(x)-1)/cqp_abs(y));
		} else {
			return x/y;
		}
	}

	template <typename T>
	T cqp_mod(T x, T y) { //integral types x and y with default truncated division returns x modulo y
	       return x - y*cqp_div(x,y); //abs(return value) < abs(y), sign(return value)==sign(y)
	}       

	//return overestimate of absolute value of square root of input
	//input x>=0, eps>0, output y such that 
	//	sqrt(x) <= y < sqrt(x)*(1+eps)
	inline mpq_class sqrt(mpq_class input, mpq_class eps = mpq_class(1,4294967296)) {
		mpz_class num = input.get_num();
		mpz_class den = input.get_den();
		if(num>0 && eps>0) { 
			//find floor(lg(input))
			size_t lg_num = mpz_sizeinbase(num.get_mpz_t(), 2);
			size_t lg_den = mpz_sizeinbase(den.get_mpz_t(), 2);
			mpz_class lg_input = mpz_class(lg_num)-mpz_class(lg_den);
			mpz_class pow2_num, pow2_den;
			mpz_ui_pow_ui(pow2_num.get_mpz_t(), 2, lg_num);
			mpz_ui_pow_ui(pow2_den.get_mpz_t(), 2, lg_den);
			if(num*pow2_den < den*pow2_num) {
				--lg_input;
			}

			//initialize output to 0th iterate
			mpz_class lg_init = 1+cqp_div(lg_input,mpz_class(2));
			mpq_class output(1,1);
			if(lg_init >= 0) {
				mpz_ui_pow_ui(output.get_num().get_mpz_t(), 2, mpz_get_ui(lg_init.get_mpz_t()));
			} else {
				mpz_ui_pow_ui(output.get_den().get_mpz_t(), 2, mpz_get_ui(cqp_abs(lg_init).get_mpz_t()));
			}
			output.canonicalize();

			//initialize worst case absolute and relative error
			mpq_class rel_err;
			rel_err = 1; //abs_err = output*rel_err;

			//iterate until error less than eps
			while(rel_err>=eps) {
				output = (output+input/output)/2; //newton's method quadratic convergence nth iterate rel_err less than 2*(0.5^{2^{n}})
				rel_err = (rel_err*rel_err)/(2*(1+rel_err));
//				abs_err = rel_err*output;
			}
			return output;
		}
		return 0;
	}
	
	inline mpq_class max(mpq_class const & input1, mpq_class const & input2) {
		if(input1>input2) {
			return input1;
		} 
		return input2;
	}
	inline mpq_class min(mpq_class const & input1, mpq_class const & input2) {
		if(input1>input2) {
			return input2;
		} 
		return input1;
	}

	inline scalar scale_factor(const Eigen::Ref<const matrix> & input) {
		mpz_class output = 1;
		for(index i=0; i<input.rows(); ++i) {
			for(index j=0; j<input.cols(); ++j) {
				output = lcm(output, input(i,j).get_den());
			}
		}
		return output;
	}

	inline float_scalar max(float_scalar const & input1, float_scalar const & input2) {
		if(input1>input2) {
			return input1;
		} 
		return input2;
	}
	inline float_scalar min(float_scalar const & input1, float_scalar const & input2) {
		if(input1>input2) {
			return input2;
		} 
		return input1;
	}

	/* Prefer computed value of 
	 * 	(beta*beta + sqrt(n))/(beta + sqrt(n))
	 * to be greater than or equal to mathematically exact value and strictly less than 1. This can be done if scalar is an exact rational number type and sqrt returns a sufficiently accurate rational
	 * approximation of the square root which is greater than or equal to the mathematically exact square root.
	 */
	/*
	 * Problem form:
	 * Primal:
	 * 	min_x x^T 0.5Q x + c^T x
	 * 	s.t.  Ax = b
	 * 	      x >= 0
	 *
	 * Dual:
	 * 	max_(x,y) x^T (-0.5)Q x + b^T y
	 * 	s.t.	  Qx-A^T y+c >= 0;	
	 *
	 * where Q is an n by n symmetric positive semidefinite matrix, c is an n vector, A is an 
	 * m by n matrix with rank m, b is an m vector, and ^T denotes the transpose operator.
	 *
	 * Note: 
	 * If rank(A)<m then a row permutation of A can be written as the block matrix
	 * 	[  B  ]
	 * 	[ C*B ]
	 * where B has full row rank. In this case either there is no solution to Ax = b and the 
	 * problem is infeasible, or the rows corresponding to C*B can be removed from A and b.
	 */

	/* homotopy_algorithm assumptions:
	 * (a) S := {x \in (real numbers)^n | Ax = b, x > 0} is non-empty
	 * (b) {w \in (real numbers)^m | A^T w < c} is non-empty
	 * (c) A has full row rank 
	 *
	 * input restrictions:
	 * (i) 0 < beta < 1 
	 * (ii) 0 < gamma <= eta
	 * (iii) x \in S and there exists a y \in (real numbers)^m such that
	 * 	 ||(eta*I + diag(x)*Q*diag(x))^(-0.5)*(eta*(ones vector) - diag(x)*(Qx+c-A^T*y)||/(eta^(0.5)) <= beta
	 *
	 * note: no safety checks for the mentioned assumptions and restrictions.
	 *
	 * Miscellaneous info: 
	 * After r+1 iterations of the while loop (including the first from the do) the 
	 * complementary slackness condition from the KKT conditions will be approximately 
	 * satisfied. Specifically
	 * 	0 <= diag(x) (Q x + c - A^T y) <= (alpha)^(r-1) * eta * ones_vector
	 * 	0 <= x^T (Q x + c - A^T y) <= n * (alpha)^(r-1) * eta 
	 * where x and y are the respective values of x and y after the loop terminates and 
	 * 	alpha = (beta^2 + sqrt(n))/(beta+sqrt(n)).
	 *
	 * After each loop the upper bound on the complementary slackness condition is multiplied by alpha. The value of beta that minimizes alpha is given by
	 * 	sqrt(n+sqrt(n))-sqrt(n).
	 * In which case 
	 * 	alpha = 2*(sqrt(n+sqrt(n))-sqrt(n))=2*beta.
	 * In the limit as n->infinity sqrt(n+sqrt(n))-sqrt(n) increases towards 0.5.
	 */
	//floating point version
	inline void homotopy_algorithm(const Eigen::Ref<const float_matrix> & Q, const Eigen::Ref<const float_vector> & c, const Eigen::Ref<const float_matrix> & A, 
		       	const Eigen::Ref<const float_vector> & b, float_vector x, 
			float_scalar beta, float_scalar gamma, float_scalar eta, 
			float_vector & solution_primal_part, float_vector & solution_dual_part) {
		index m = A.rows(); index n = A.cols(); 
		/* require 
		 * 	2*(sqrt(n+sqrt(n))-sqrt(n)) <= alpha < 1
		 * and
		 * 	0 < beta <= (alpha+sqrt(alpha^2-4*(1-alpha)*sqrt(n)))/2.
		 */
		float_scalar alpha;
		alpha = (beta*beta + sqrt(float_scalar(n)))/(beta + sqrt(float_scalar(n)));
		/* If 
		 * 	|beta| <= (alpha+sqrt(alpha^2-4*(1-alpha)*sqrt(n)))/2 
		 * then method should still converge as long as 
		 * 	alpha >= 2*(sqrt(n+sqrt(n))-sqrt(n)),
		 * even if 
		 * 	alpha < (beta^2+sqrt(n))/(beta+sqrt(n)). 
		 */
		index max_alpha_step = 1;

		float_matrix M = float_matrix::Zero(m+n, m+n); 
		M.block(0,0,n,n) = Q;
		M.block(0,n,n,m) = -A.transpose();
		M.block(n,0,m,n) = -A;

		float_vector rhs = float_vector::Zero(m+n);
		rhs.block(n,0,m,1) = -b;

		float_vector v = float_vector::Zero(m+n);
		float_vector diff = float_vector::Zero(m+n);
		float_vector res = float_vector::Zero(m+n);
		index refinement_iter = 1;
		index centering_iter = 1;
		float_matrix X; 
		index alpha_step_count = 0;
		float_scalar tolerance = Eigen::NumTraits<float_scalar>::epsilon();
		float_vector y = float_vector::Zero(m);
		float_vector z = float_vector::Zero(n);

		index_vector illi(n); //indices in {1,...,n} of those x_i contributing to ill-conditioning of float_matrix.
		index_vector welli(n); //complement of illi in {1,...,n}
		for(index i=0;i<n;++i) {
			welli(i) = i;
		}
		index welli_size = n;
		index illi_size = 0;

		float_matrix ill_I(n, 0); //columns of identity float_matrix corresponding to those x_i contributing to ill-conditioning of float_matrix.
		float_matrix well_I = float_matrix::Identity(n,n); //the rest of the columns of the identity float_matrix
		float_matrix ill_X_inv_2, well_X_inv_2;
		int count = 0; //higher values mean larger variable values considered to contribute to ill conditioning. cutoff scaled by 2^count
		int max_count = 2; //count increases to at most max_count before giving up.
		float_scalar oalpha(alpha);
		float_scalar lalpha(alpha);

		float_scalar gap(0);
		float_scalar lgap(gap); //gap corresponding to lx, ly
		float_vector lx(x); //last x found which is feasible
		float_vector ly(y); //last y found which is feasible
		bool inner_fail; //inner loop failed to find feasible x and y?
		index max_iter_count = centering_iter; //inner loop max iterations

		float_matrix well_X_2;

#if LOG_HOMOTOPY_ALGORITHM 
		{
			mpq_class mpqeta = mpq_class(eta);
			mpq_class mpqgamma = mpq_class(gamma);
			mpq_class estiter = (mpq_class(mpz_sizeinbase(mpqeta.get_num().get_mpz_t(),2)+mpz_sizeinbase(mpqgamma.get_den().get_mpz_t(),2),1) - mpq_class(mpz_sizeinbase(mpqeta.get_den().get_mpz_t(),2) + mpz_sizeinbase(mpqgamma.get_num().get_mpz_t(),2),1) + 2)*.7/mpq_class(1-alpha);
			estiter = cqp_div(mpz_class(estiter.get_num()+estiter.get_den()-1), estiter.get_den());
			std::cout << "est. max iterations == " << estiter << "\n";
		}
		size_t total_iter_count = 0;
		float_scalar max_r0;
#endif 

		do { 
#if LOG_HOMOTOPY_ALGORITHM
			++total_iter_count;
			std::cout << "eta == " << eta << "\n";
			std::cout << "x max == " << x.maxCoeff() << "\n";
			std::cout << "y max == " << v.tail(m).cwiseAbs().maxCoeff() << "\n";
#endif
			index iter_count=0;
			inner_fail = true;
			do {
				X = x.asDiagonal();
				ill_X_inv_2 = float_matrix::Zero(illi_size, illi_size);
				for(index i=0; i<illi_size; ++i) {
					ill_X_inv_2(i,i) = (1/x(illi(i)))*(1/x(illi(i)));
				}
				well_X_inv_2 = float_matrix::Zero(welli_size, welli_size);
				well_X_2 = float_matrix::Zero(welli_size, welli_size);
				for(index i=0; i<welli_size; ++i) {
					well_X_inv_2(i,i) = (1/x(welli(i)))*(1/x(welli(i)));
					well_X_2(i,i) = x(welli(i))*x(welli(i)); 
				}
	
				//set top left block in a way that attempts to decrease ill conditioning due to variables approaching the boundary of the feasible region.
				M.block(0,0,n,n) = Q*X; //scaling by last computed good x.
				if(welli_size) {
					M.block(0,0,n,n) += eta*well_I*well_X_inv_2*well_I.transpose()*X;
					if(illi_size) { 
						float_matrix well_Q = well_I.transpose()*Q*well_I; //fullPivLu crashes on zero float_matrix input
						/* In exact arithmetic well_Q+eta*well_X_inv_2 is positive definite.*/
						float_matrix nonzero_float_matrix = well_Q+eta*well_X_inv_2; //On the other hand we don't want crashes and fullPivLu crashes on zero float_matrix input.
						if(nonzero_float_matrix.isZero()) { 
							/* If eta*well_X_inv_2 is zero then so is well_Q and so we use well_X_2/eta as the inverse.
							 *
							 * Suppose eta*well_X_inv_2 is nonzero, then it's nonzero entries are positive and on its main diagonal.
							 * well_Q must have some negative main diagonal entries and all other entries zero.
							 * Assume negative diagonal elements of well_Q are approximately zero. Replace well_Q with zero float_matrix
							 * and use well_X_2/eta as the inverse.
							 */
							M.block(0,0,n,n) += ill_I * ( eta*ill_X_inv_2 - 
									ill_I.transpose() * ( Q - 
										Q*well_I * well_X_2*(well_I.transpose()*Q)/eta)*ill_I)*ill_I.transpose()*X;
						} else {
							M.block(0,0,n,n) += ill_I * ( eta*ill_X_inv_2 - 
									ill_I.transpose() * ( Q - 
										Q*well_I * (nonzero_float_matrix).fullPivLu().solve(well_I.transpose()*Q))*ill_I)*ill_I.transpose()*X;
						}
					}
				} else if(illi_size) {
					M.block(0,0,n,n) = ill_I*eta*ill_X_inv_2*ill_I.transpose()*X;
				}
				M.block(n,0,m,n) = -A*X; //bottom left block
				//inverse of this M float_matrix approximates inverse of the float_matrix that would have been obtained if illi_size was 0

				rhs.block(0,0,n,1) = 2*eta*x.cwiseInverse()-c; 

				if(M.isZero(0)) { //uh oh
					v = float_vector::Zero(m+n);
				} else {
					//solve for v in M*v = rhs
					Eigen::FullPivLU<float_matrix> lu(M);
					v = lu.solve(rhs);
					for(index i=0; i<refinement_iter; ++i) { //improve solution quality
						res = rhs-M*v;
						diff = lu.solve(res);
						v = v+diff;
					}
				}

				z = Q*X*v.head(n)+M.block(0,n,n,m)*v.tail(m)+c;

				if(((v.head(n)).array() <= 0).any() || (z.array() < 0).any()) { //ensure primal strict feasibility and dual feasibility. 
					if(alpha_step_count>=max_alpha_step) { //failure
						break;
					} else { //try again with smaller step
						alpha = (1+alpha)/2;
						eta = eta*alpha/lalpha; //closer to previous eta
						lalpha = alpha; //set last alpha
						++alpha_step_count;
						++max_iter_count; //ensures loop executed again
					}
				} else { //feasible but not necessarily good
					x = X*v.head(n); /*invert scaling before assignment.*/
					x = x.cwiseMax(float_vector::Constant(n, Eigen::NumTraits<float_scalar>::epsilon())); //prevent zero entries.
					y = v.tail(m);
					gap = x.dot(z); //duality gap of computed x and y
					inner_fail = false; //success
				}
				++iter_count;
			} while (iter_count < max_iter_count);
			alpha_step_count = 0; //reset to original values
			alpha = oalpha;
			max_iter_count = centering_iter;

			if(abs(gap) >= max((1+tolerance)*n*eta, abs(lgap)) || inner_fail) { //feasible but not good
				float_vector rejected;
				if(inner_fail) {//if inner_fail we use v.head(n) to ensure any x(i) that were negative will be added to set of variables contributing to ill conditioning
					rejected = v.head(n);  
				} else { //otherwise use the scaled feasible x.
					rejected = X.inverse()*x;
				}
				float_scalar bound; mpf_mul_2exp(bound.get_mpf_t(), eta.get_mpf_t(), 7+count); 
				//increased count may not increase bound enough to change previous outcome but sorting float_vector too expensive.
		
				illi_size = 0;
				welli_size = 0;
				for(index i=0; i<n; ++i) {
					if(rejected(i) <= bound) { //if less than bound we predict that corresponding variable approaching boundary and so contributes to ill conditioning
						illi(illi_size++) = i;
					} else {
						welli(welli_size++) = i;
					}
				}
				ill_I.resize(n, illi_size);
				well_I.resize(n, welli_size);
				for(index i=0; i<illi_size; ++i) {
					ill_I.col(i) = float_vector::Unit(n, illi(i));
				}
				for(index i=0; i<welli_size; ++i) {
					well_I.col(i) = float_vector::Unit(n, welli(i));
				}
				++count;
			} else { //feasible and good
				lgap = gap; //update
				lx = x;
				ly = y;
				eta = alpha*eta;

				illi_size = 0; //reset 
				welli_size = n;
				for(index i=0;i<n;++i) {
					welli(i) = i;
				}
				ill_I.resize(n, 0);
				well_I = float_matrix::Identity(n,n);
				count = 0;
#if LOG_HOMOTOPY_ALGORITHM
				max_r0 = x.cwiseProduct(Q*x+c-A.transpose()*v.tail(m)).maxCoeff(); //<= eta;
				std::cout << "max gap component == " << max_r0 << "\n";
#endif
			}
		} while(eta>gamma && count<max_count);
#if LOG_HOMOTOPY_ALGORITHM
		std::cout << "total iter == " << total_iter_count << "\n";
#endif
		solution_primal_part = lx; //return last feasible && good pair
		solution_dual_part = ly;
	} //end of homotopy_algorithm

	inline void float_homotopy_algorithm(const Eigen::Ref<const matrix> & Q, const Eigen::Ref<const vector> & c, const Eigen::Ref<const matrix> & A, //just passes to above floating point version 
		       	const Eigen::Ref<const vector> & b, vector x, 
			scalar beta, scalar gamma, scalar eta, 
			vector & solution_primal_part, vector & solution_dual_part) {
		float_vector sol_x = solution_primal_part.cast<float_scalar>();
		float_vector sol_y = solution_dual_part.cast<float_scalar>();

		homotopy_algorithm(Q.cast<float_scalar>(), c.cast<float_scalar>(), A.cast<float_scalar>(), b.cast<float_scalar>(), x.cast<float_scalar>(),
				float_scalar(beta), float_scalar(gamma), float_scalar(eta),
				sol_x, sol_y);

		solution_primal_part = sol_x.cast<scalar>();
		solution_dual_part = sol_y.cast<scalar>();
	} 


	//get fraction with smallest denominator in [a,b]
	inline scalar simple_fraction(scalar a, scalar b) { //a<=b
		scalar sig = cqp_sign(a);
		if(sig*cqp_sign(b)<=0) {
			return 0;
		}
		if(a == 1 || b == 1) {
			return 1;
		}
		if(a == -1 || b == -1) {
			return -1;
		}
		if(sig<0) { //if both negative swap and make positive then at the end return negative of result
			scalar temp = b;
			b = -a;
			a = -temp;
		}
		mpz_class ai, bi;
		scalar af, bf;
		af = a; bf = b;
		matrix A = matrix::Identity(2,2);
		matrix B = matrix::Identity(2,2);
		
		while(true) {
			ai = cqp_div(af.get_num(), af.get_den()); //integer part
			af = af - scalar(ai); //fractional part
			bi = cqp_div(bf.get_num(), bf.get_den()); 
			bf = bf - scalar(bi);
			A = A*(matrix(2,2) << scalar(ai), 1, 1, 0).finished();
			B = B*(matrix(2,2) << scalar(bi), 1, 1, 0).finished();

			if(ai < bi) {
				if(af == 0) {
					return sig*a;
				}
				return sig*(A(0,0)+A(0,1))/(A(1,0)+A(1,1));
			}
			if(ai>bi) {
				if(bf == 0) {
					return sig*b;
				}
				return sig*(B(0,0)+B(0,1))/(B(1,0)+B(1,1));
			}
			if(af == 0) {
				return sig*a;
			}
			if(bf == 0) {
				return sig*b;
			}

			af = 1/af;
			bf = 1/bf;
		}
	}

	inline size_t get_max_length(const Eigen::Ref<const vector> & x, int base) { //returns exact or at most 2 too large
		size_t max_length_of_coeff = 0;
		size_t length_of_coeff = 0;
		for(index i=0; i<x.size(); ++i) {
			length_of_coeff = mpz_sizeinbase(x(i).get_num().get_mpz_t(), base) + mpz_sizeinbase(x(i).get_den().get_mpz_t(), base);
			if(length_of_coeff > max_length_of_coeff) {
				max_length_of_coeff = length_of_coeff;
			}
		}
		return max_length_of_coeff;
	}

	/* Need to deal with growth of numerator+denominator of rational numbers in exact arithmetic. Depends on ability to find fraction with small denominator+numerator in a given closed interval 
	 * with rational endpoints, as well as the size of that interval, which will be a function of the matrices A and Q, the previous iterates' solution, eta, and beta.
	 *
	 * The fraction with smallest denominator (and numerator) in a closed interval with rational endpoints can be found using the continued fraction expansions of the interval endpoints.
	 *
	 * ---done--- Currently need to finish extending homotopy algorithm to perturbed KKT system in order to find valid nontrivial intervals for acceptable perturbation of coefficients of each iterates solution. 
	 * In other words at each iteration there should be a collection of small intervals each containing a coefficient of the solution such that replacing each coefficient of the solution with any
	 * other number in the associated interval gives a solution to a new system with same LHS side matrix but with RHS altered by a perturbation vector with small magnitude. Perturbation
	 * vector maximum magnitude decreases with each iteration such that when the algorithm ends the magnitude of the final perturbation is sufficiently small (small enough to not cause purification
	 * step to fail when it would otherwise succeed). Will probably require increasing the required number of iterations depending on the desired sizes of the intervals.
	 *
	 * Maximum perturbation for component i of x given by s*previous_x(i).
	 * Don't bother perturbing y.
	 */
	inline void homotopy_algorithm(const Eigen::Ref<const matrix> & Q, const Eigen::Ref<const vector> & c, const Eigen::Ref<const matrix> & A, 
		       	const Eigen::Ref<const vector> & b, vector x, 
			scalar beta, scalar gamma, scalar eta, 
			vector & solution_primal_part, vector & solution_dual_part, scalar max_eq_error = 0) { //max_eq_error denotes bound on infinity norm of A*x-b in solution returned
		index m = A.rows(); index n = A.cols(); 

		if(!n) { //no primal variables
			solution_primal_part = x;
			solution_dual_part = vector::Zero(m);
			return;
		}

		matrix M = matrix::Zero(n+m, n+m); 
		M.block(0,n,n,m) = -A.transpose();
		M.block(n,0,m,n) = -A;

		Eigen::FullPivLU<matrix> luM;
		vector x_inv; 
		vector rhs(n+m);
		rhs.tail(m) = -b;
		vector v = vector::Zero(m+n);
		scalar sqrtn = sqrt(scalar(n));
		scalar low_alpha = (beta*beta + sqrtn)/(beta+sqrtn); //no lower
		scalar high_alpha = (beta*beta + 1.414213563*sqrtn)/(beta+1.414213563*sqrtn); //can replace 1.414213563 with anything >= 1.
		scalar alpha = low_alpha; //choose alpha s.t. low_alpha <= alpha <= high_alpha.
		vector w = x;
		scalar phi = 0; 
		scalar p0, p1, p2, p3, s, s1;

		gamma = gamma;
		scalar next_eta;
		scalar max_r0;
#if LOG_HOMOTOPY_ALGORITHM 
		mpq_class estiter = (mpq_class(mpz_sizeinbase(eta.get_num().get_mpz_t(),2)+mpz_sizeinbase(gamma.get_den().get_mpz_t(),2),1) - mpq_class(mpz_sizeinbase(eta.get_den().get_mpz_t(),2) + mpz_sizeinbase(gamma.get_num().get_mpz_t(),2),1) + 2)*.7/(1-high_alpha); //overestimate of max number iterations
		estiter = cqp_div(mpz_class(estiter.get_num()+estiter.get_den()-1), estiter.get_den());
		size_t iter_count = 0;
		std::cout << "low_alpha == " << float_scalar(low_alpha) << "\n";
		std::cout << "high_alpha == " << float_scalar(high_alpha) << "\n";
		mpz_class s_max_length(0), x_max_length(0), y_max_length(0), eta_max_length(0), alpha_max_length(0), x_length, y_length, eta_length, alpha_length, s_length;
		float_scalar min_r0_step(eta), max_r0_step(-eta), min_alpha(high_alpha), max_alpha(0), min_eta_step(eta), max_eta_step(-eta), float_step(0);
		mpz_class max_eta_length_reduction(0), max_x_length_reduction(0), length_step(0), max_s_length_reduction(0);
		std::cout << "est. max iterations == " << estiter << "\n\n";
#endif 
		do { 
#if LOG_HOMOTOPY_ALGORITHM
			++iter_count;
			next_eta = alpha*eta;
			length_step = (mpz_class(mpz_sizeinbase(next_eta.get_num().get_mpz_t(), 10) + mpz_sizeinbase(next_eta.get_den().get_mpz_t(), 10)));
#endif
			next_eta = simple_fraction(low_alpha*eta, high_alpha*eta);
#if LOG_HOMOTOPY_ALGORITHM
			length_step -= mpz_class(mpz_class(mpz_sizeinbase(next_eta.get_num().get_mpz_t(), 10) + mpz_sizeinbase(next_eta.get_den().get_mpz_t(), 10)));
			if(length_step > max_eta_length_reduction) {
				max_eta_length_reduction = length_step;
			}
#endif
			alpha = next_eta/eta;

			x_inv = x.cwiseInverse();
			rhs.head(n) = 2*eta*x_inv-c;
			M.block(0,0,n,n) = Q+eta*x_inv.cwiseProduct(x_inv).asDiagonal()*matrix::Identity(n,n);
			luM.compute(M);
			v = luM.solve(rhs);
			w = x_inv.cwiseProduct(v.head(n)-x);

			phi = (x.asDiagonal()*Q*x.asDiagonal()).cwiseAbs().rowwise().sum().maxCoeff(); 
			p0 = phi/(alpha*eta);
			p1 = (1-w.minCoeff())/alpha + p0*(1+w.maxCoeff())+sqrt(p0);
			p2 = beta+sqrtn-(sqrtn+w.squaredNorm())/alpha;
			p3 = 1+w.minCoeff();
			/* want nonnegative underestimate s of smallest nonnegative zero of
			 * 	f(x) = p0*(x^2)+p1*x-p2+x/(p3-x)
			 * f has three real roots so no solution in real radicals. 
			 * 0 <= s < p3
			 * Define g by
			 * 	g(x) = p0*x+p1+1/(p3-x).
			 * We have
			 * 	f(x) = g(x)*x-p2
			 * 	f(s) = 0 <--> s = p2/g(s).
			 * define h by
			 * 	h(x) = p2/g(x).
			 * For all real x, h'(x)<=0.
			 * The two vertical asymptotes of h are contained in the complement of (-p1/p0, p3).
			 * If 0<=x<p3 then 
			 * 	0 <= h(x) <= p2*p3/(p1*p3+1) < p3.
			 * If x,y \in [0, p3) then
			 * 	|h(x)-h(y)|/|x-y| = p2*(p0 + 1/((p3-x)*(p3-y)))/(g(x)*g(y)) < p2 <= beta < 1
			 * So s is an attractive fixed point of h and any initial point in [0,p3) will converge to s under iteration by h.
			 * Let s(0) = 0, and s(n+1) = h(s(n)). Then
			 * 	|s(n+1) - s(n)| < ((p2)^n)*s(1) < ((p2)^n)
			 * 	|s(n) - s| < (p2^n)*s(1)/(1-p2)
			 * Furthermore for each nonnegative integer n
			 * 	s(2*n) < s(2*(n+1)) < s < s(2*(n+1)+1) < s(2*n+1).
			 */
			s = 0;
			for(index ind = 0; ind < 3; ++ind) {
				s1 = p2*(p3-s)/((p0*s+p1)*(p3-s)+1);
				s = p2*(p3-s1)/((p0*s1+p1)*(p3-s1)+1);
#if LOG_HOMOTOPY_ALGORITHM
				length_step = mpz_sizeinbase(s.get_num().get_mpz_t(), 10) + mpz_sizeinbase(s.get_den().get_mpz_t(), 10);
#endif
				s = simple_fraction((1-p2*(s1-s))*s, s);
#if LOG_HOMOTOPY_ALGORITHM
				s_length = mpz_sizeinbase(s.get_num().get_mpz_t(), 10) + mpz_sizeinbase(s.get_den().get_mpz_t(), 10);
				if(s_length > s_max_length) {
					s_max_length = s_length;
				}
				length_step -= s_length;
				if(length_step > max_s_length_reduction) {
					max_s_length_reduction = length_step;
				}
#endif
			}
			s = s/sqrtn;
			eta = next_eta;
			for(index i=0; i<n; ++i) {
#if LOG_HOMOTOPY_ALGORITHM
				length_step = mpz_sizeinbase(x(i).get_num().get_mpz_t(), 10) + mpz_sizeinbase(x(i).get_den().get_mpz_t(), 10);
#endif
				//use continued fractions to find simplest fraction in [v(i)-x(i)*s, v(i)+x(i)*s] and set x(i) equal to it.
				x(i) = simple_fraction(v(i)-x(i)*s, v(i)+x(i)*s);
#if LOG_HOMOTOPY_ALGORITHM
				length_step -= mpz_class(mpz_sizeinbase(x(i).get_num().get_mpz_t(), 10) + mpz_sizeinbase(x(i).get_den().get_mpz_t(), 10));
				if(length_step > max_x_length_reduction) {
					max_x_length_reduction = length_step;
				}
#endif
			}
#if LOG_HOMOTOPY_ALGORITHM 
			float_step = float_scalar(alpha);
			if(float_step < min_alpha) {
				min_alpha = float_step;
			}
			if(float_step > max_alpha) {
				max_alpha = float_step;
			}

			float_step = float_scalar(next_eta-eta);
			if(float_step < min_eta_step) {
				min_eta_step = float_step;
			}
			if(float_step > max_eta_step) {
				max_eta_step = float_step;
			}
			x_length = get_max_length(x, 10);
			if(x_length > x_max_length) {
				x_max_length = x_length;
			}

			y_length = get_max_length(v.tail(m), 10);
			if(y_length > y_max_length) {
				y_max_length = y_length;
			}

			eta_length = mpz_sizeinbase(eta.get_num().get_mpz_t(), 10) + mpz_sizeinbase(eta.get_den().get_mpz_t(), 10);
			if(eta_length > eta_max_length) {
				eta_max_length = eta_length;
			}

			alpha_length = mpz_sizeinbase(alpha.get_num().get_mpz_t(), 10) + mpz_sizeinbase(alpha.get_den().get_mpz_t(), 10);
			if(alpha_length > alpha_max_length) {
				alpha_max_length = alpha_length;
			}
#endif 

#if LOG_HOMOTOPY_ALGORITHM
			float_step = float_scalar(max_r0);
#endif	
			max_r0 = x.cwiseProduct(Q*x+c-A.transpose()*v.tail(m)).maxCoeff(); //<= eta;
#if LOG_HOMOTOPY_ALGORITHM
			float_step -= float_scalar(max_r0);
			float_step = -float_step;
			if(float_step < min_r0_step) {
				min_r0_step = float_step;
			}
			if(float_step > max_r0_step) {
				max_r0_step = float_step;
			}
#endif
		} while(max_r0 > gamma); //eta > gamma
#if LOG_HOMOTOPY_ALGORITHM
		std::cout << "iter_count == " << iter_count << "\n\n";

		std::cout << "eta_max_length == " << eta_max_length << "\n";
		std::cout << "x_max_length == " << x_max_length << "\n";
		std::cout << "s_max_length == " << s_max_length << "\n";
		std::cout << "y_max_length == " << y_max_length << "\n";
		std::cout << "alpha_max_length == " << alpha_max_length << "\n\n";

		std::cout << "max_eta_length_reduction == " << max_eta_length_reduction << "\n";
		std::cout << "max_x_length_reduction == " << max_x_length_reduction << "\n";
		std::cout << "max_s_length_reduction == " << max_s_length_reduction << "\n\n";

		std::cout << "min_r0_step == " << min_r0_step << "\n";
		std::cout << "max_r0_step == " << max_r0_step << "\n";

		std::cout << "min_eta_step == " << min_eta_step << "\n";
		std::cout << "max_eta_step == " << max_eta_step << "\n";

		std::cout << "min_alpha == " << min_alpha << "\n";
		std::cout << "max_alpha == " << max_alpha << "\n";
#endif
		/*once more to enforce Ax==b error <= sqrt(gamma)*/
		if((A*x-b).cwiseAbs().maxCoeff() > max_eq_error && !A.isZero(0)) {
			scalar A_infinity = A.cwiseAbs().rowwise().sum().maxCoeff();

			x_inv = x.cwiseInverse();
			rhs.head(n) = 2*eta*x_inv-c;
			M.block(0,0,n,n) = Q+eta*x_inv.cwiseProduct(x_inv).asDiagonal()*matrix::Identity(n,n);
			luM.compute(M);
			v = luM.solve(rhs);

			phi = (x.asDiagonal()*Q*x.asDiagonal()).cwiseAbs().rowwise().sum().maxCoeff(); 
			p0 = phi/(alpha*eta);
			p1 = (1-w.minCoeff())/alpha + p0*(1+w.maxCoeff())+sqrt(p0);
			p2 = beta+sqrtn-(sqrtn+w.squaredNorm())/alpha;
			p3 = 1+w.minCoeff();
			s = 0;
			for(index ind = 0; ind < 3; ++ind) {
				s1 = p2*(p3-s)/((p0*s+p1)*(p3-s)+1);
				s = p2*(p3-s1)/((p0*s1+p1)*(p3-s1)+1);
				s = simple_fraction((1-p2*(s1-s))*s, s);
			}
			s = s/sqrtn;
			scalar s2 = gamma/A_infinity;
			scalar radius;
			for(index i=0; i<n; ++i) {
				radius = min(x(i)*s, s2);
				x(i) = simple_fraction(v(i)-radius, v(i)+radius);
			}
		}
		/*
		x_inv = x.cwiseInverse();
		rhs.head(n) = 2*eta*x_inv-c;
		M.block(0,0,n,n) = Q+eta*x_inv.cwiseProduct(x_inv).asDiagonal()*matrix::Identity(n,n);
		luM.compute(M);
		v = luM.solve(rhs);
		*/

		solution_primal_part = x;
		solution_dual_part = v.tail(m);
	}
	/* purify info
	 * Let M denote the (n+m)x(n+m) block matrix  
	 * 	[Q A^T]
	 * 	[A  0 ]
	 * and let N denote the (2n)x(n+m) block matrix
	 * 	[Q -A^T]
	 * 	[I   0 ]
	 * where I denotes the nxn identity matrix.
	 * Let gamma >= 0 be such that 
	 * 	gamma >= sup { |det(B_2)/det(B_1)| : B_1, B_2 square submatrices of M differing by exactly one row and B_1 invertible }
	 * and delta such that 
	 * 	M(x,-y) + (c,0) >= delta
	 * 	x >= delta 
	 * where for all vectors u and v (u,v) denotes the vector [u^T v^T]^T.
	 *
	 * Suppose that A has full row rank and Ax-b<=epsilon. Under these assumptions this algorithm returns (p,d) satisfying
	 * 	A*p = b
	 * 	N*(p,d) + (c,0) >= (min(epsilon, delta) - (m+n)*max(|epsilon|, |delta|)*gamma)*ones_vector
	 * 	There exists a subset K of {1, ..., 2n} such that if i in K then
	 *		N_i*(p,d) + (c,0)_i <= epsilon + (m+n)*max(|epsilon|, |delta|)*gamma,
	 *	if 
	 * 		N_i*(x,y) + (c,0)_i <= epsilon 
	 * 	then i in K, and the rows of A together with the rows of N indexed by K span the rows of N, where N_i denotes the ith row of N and (c,0)_i denotes the ith entry of (c,0). 
	 */
	inline void purify(const Eigen::Ref<const matrix> & Q, const Eigen::Ref<const vector> & c, const Eigen::Ref<const matrix> & A,
		       	const Eigen::Ref<const vector> & b, vector x, vector y,
			scalar epsilon, 
			vector & solution_primal_part, vector & solution_dual_part) {
		index m = A.rows(); index n = A.cols(); 

		if(n==0) { //off chance someone passes an empty matrix. If generalize to LCP later where arbitrary M matrix allowed then change this to check if M zero.
			solution_primal_part = matrix::Zero(n,1);
			solution_dual_part = matrix::Zero(m,1);
			return;
		}

		matrix M = matrix::Zero(m+2*n, m+n); 
		M.block(0,0,m,n) = A;
		M.block(m,0,n,n) = Q;
		M.block(m,n,n,m) = -A.transpose();
		M.block(m+n,0,n,n) = matrix::Identity(n,n);
		vector q(m+2*n); q << (-b), c, vector::Zero(n);
		/* M ==
		 * 	[A   0 ]
		 *	[Q -A^T]
		 *	[I   0 ]
		 *
		 * q ==
		 * 	[-b ]
		 * 	[ c ]
		 * 	[ 0 ]
		 */
		
		vector v(m+n); v << x, y; //v == (x,y)

		Eigen::VectorXi ip = Eigen::MatrixXi::Constant(m+2*n,1,-1); //row indices permutation
		for(index i=0; i<ip.rows(); ++i) {
			ip(i) = i;
		}
		index s_size = 0; //first s_size indices of ip are indices i such that M.row(i)*v <= epsilon 
		index sc_size = m+2*n;//subsequent sc_size indices are indices i of rows of M such that M.row(i) not in the span of the rows of M index by s. 
		//remaining indices i are such that M.row(i)*v > epsilon but M.row(i) is in the span of rows of M indexed by s.
		
		Eigen::VectorXi sb = Eigen::MatrixXi::Constant(m+n,1,-1); 
		index sb_size = 0;
		//after loop complete sb stores a subset of first s_size indices of ip such that the corresponding rows of M are a basis for the row space of M.

		matrix Mp = M; //Mp == P*M for permutation matrix P with row i of P given by row vector with 1 in ip(i) position and 0 elsewhere.
		vector qp = q; //qp == P*q.
		matrix echelon = Mp.transpose(); //has a submatrix (first k columns for some k) corresponding to echelon form for a submatrix of Mp^T

		vector theta_psc = Mp.block(s_size, 0, sc_size, Mp.cols())*v+qp.block(s_size, 0, sc_size, 1); //Updates at end of loop.
		/* For each n+m vector z let s(z) denote the set of all i in {1, ..., 2n+m} such that M.row(i)*z+q(i)<= epsilon, let S(z) denote the rows of M indexed by s(z), and let span(S(z)) denote the
		 * set of all linear combinations of elements of S(z). 
		 * Pseudocode for the following loop:
		 *
		 * 0. while there exists an i in {1, ...,2n+m} such that M.row(i) is not in span(S(v)) do
		 * 1.  	pick i in {1, ..., 2n+m} such that M.row(i) not in span(S(v));
		 * 2. 	find a vector u in the orthogonal complement of span(S(v)) such that M.row(i).dot(u) = 1;
		 * 3. 	t := max({(epsilon-M.row(j)*v-q.row(j))/(M.row(j)*v) : j in {1, ..., 2n+m} and M.row(j)*v > 0});
		 * 4. 	v := v+t*u;
		 * 5. end
		 *
		 * The actual implementation is more complicated. In 0. and 1. we just check those indices i such that on the previous iteration of the loop M.row(i) was not in the span(S(v)) (the previous 
		 * iteration's v). To handle 0., 1., and 2., on each iteration we partially compute the row echelon form of Mp.transpose() where Mp is obtained from M via a row permutation such that the 
		 * first s_size rows of Mp are the elements of S(v), the following sc_size rows are the rows of M which were not in the span(S(v)) on the previous iteration, and the remaining rows are those
		 * rows which were in span(S(v)) but not in S(v). We keep the partially computed echelon form matrix between iterations carrying out further pieces of the row echelon form computation on it 
		 * and permuting it ('s columns) with Mp ('s rows) so that they match. In step 4. instead of checking the entire index set {1, ..., 2n+m} we just check those indices j such that on the 
		 * previous iteration M.row(j) was not in span(S(v)). The condition check in 0. is part of the implementation's loop body.
		 */

		for(index iteration_number=0; iteration_number<=n+m; ++iteration_number) {//force termination after n+m iterations because I'm paranoid.
			index old_s_size = s_size;
			//out of those i such that on the previous iteration M.row(ip(s_size+i)) was not in the span of the rows indexed by s(v) get those i such that now theta_psc(ip(sc_size+i)) <= epsilon.
			for(index i=0, j=sc_size; i< j; ++i) {
				if(theta_psc(i) <= epsilon) {
					ip.row(s_size).swap(ip.row(old_s_size+i));
					Mp.row(s_size).swap(Mp.row(old_s_size+i));
					qp.row(s_size).swap(qp.row(old_s_size+i));
					echelon.col(s_size).swap(echelon.col(old_s_size+i));
					++s_size;
					--sc_size;
				}
			}

			index ind_row = sb_size; index ind_col = old_s_size;
			index rows = echelon.rows(); index cols = echelon.cols();
			while(ind_row < rows && ind_col < s_size) {//partially compute row echelon form. So that first s_size columns are in row echelon form (without normalized pivots).
				typedef Eigen::internal::scalar_score_coeff_op<scalar> Scoring;
				typedef typename Scoring::result_type Score;
				index ind_max;
				Score val_max = echelon.col(ind_col).tail(rows-ind_row).unaryExpr(Scoring()).maxCoeff(&ind_max); //can replace cwiseAbs() with different function to determine "largest" pivot
				ind_max += ind_row; //since using tail() offsets index
				if(val_max == Score(0)) {
					++ind_col;
					continue;
				}
				if(ind_max != ind_row) {
					echelon.row(ind_row).swap(echelon.row(ind_max));
				}
				index rrows = rows-1-ind_row;
				index rcols = cols-ind_col; //should be able to shrink this to s_size+sc_size-ind_col since we don't care about vectors in span(S(v)) but not in S(v).
				if(0<rrows) {
					echelon.bottomRightCorner(rrows,rcols) -= echelon.col(ind_col).tail(rrows)*(echelon.row(ind_row).tail(rcols)/echelon(ind_row, ind_col));
				}
				++ind_row;
				++ind_col;
			}
			Eigen::VectorXi out = Eigen::MatrixXi::Constant(sc_size,1,-1); //will hold new indices i such that M.row(ip(i)) now in span(S(v)) but not in S(v).
			index out_size = 0;

			for(index i=sb_size; i<rows; ++i) {
				bool no_pivot = true;
				for(index j=old_s_size; j<s_size; ++j) {//from previous iteration we know that if j<old_s_size then value of echelon(i,j) will be zero.
					if(echelon(i,j) != scalar(0)) {
						sb(sb_size++) = ip(j); //update indices for a basis of span(S(v)).
						no_pivot = false;
						break;
					}
				}

				if(no_pivot) {//first s_size entries are zero
					//all remaining rows have all zeros in the first s_size columns because block consisting of first s_size columns is in row echelon form.
					for(index j=s_size+sc_size-1; j>=s_size; --j) {//if j>=s_size+sc_size then column already in span from previous iterations
						//If some entry in last rows-i components of echelon.col(j) nonzero then this column corresponds to row of M not in span(S(v)),
						//otherwise echelon.col(j) corresponds to row of M now in span(S(v)) but not in S(v).
						if(((echelon.col(j).tail(rows-i).array() == 0).all())) {
							out(out_size++) = j; 
						}
					}
					break; //span check done
				}
			}

			if(out_size==sc_size) {
				break; //everything in span so done.
			}  

			//Still stuff not in span. 
			//Update ip, Mp, qp, echelon.
			for(index i=0, end=s_size+sc_size-1; i<out_size; ++i, --end, --sc_size) {
				ip.row(end).swap(ip.row(out(i)));
				Mp.row(end).swap(Mp.row(out(i)));
				qp.row(end).swap(qp.row(out(i)));
				echelon.col(end).swap(echelon.col(out(i)));
				theta_psc.row(end-s_size).swap(theta_psc.row(out(i)-s_size));
			}
			
			//Compute new v.
			matrix N(sb_size+1,M.cols());
			for(index i=0; i<sb_size; ++i) {
				N.row(i) = M.row(sb(i)); //Can eliminate need for M if keep track of basis indices in Mp.
			}
			N.row(sb_size) = Mp.row(s_size); //==M.row(ip(s_size)). ip(s_size) is index of row not in span.
			vector u = N.fullPivLu().solve(vector::Unit(sb_size+1, sb_size));
			scalar t = (epsilon - theta_psc(0));
			for(index i=0; i<sc_size; ++i) {//find max
				scalar temp0 = ((Mp.row(s_size+i)*u).value());
				if(temp0 > 0) {
					scalar temp1 = (epsilon-theta_psc(i))/temp0;
					if(temp1 > t) {
						t = temp1;
					}
				}

			}
			v = v+t*u; 
			
			theta_psc = Mp.block(s_size, 0, sc_size, Mp.cols())*v+qp.block(s_size, 0, sc_size, 1); 
		} 

		//compute final v.
		matrix Msb(sb_size, M.cols()); //submatrix corresponding to indices in sb
		vector qsb(sb_size);
		for(index i=0; i<sb_size; ++i) {
			Msb.row(i) = M.row(sb(i));
			qsb(i) = q(sb(i));
		}
		
		if(sb_size>0) {
			v = Msb.fullPivLu().solve(-qsb);
		} else {
			v.setZero();
		}
	
		solution_primal_part = v.head(n);
		solution_dual_part = v.tail(m);
	}

	/* unconstrained
	 * input problem (P)
	 * 	min  x^T 0.5*Q x + x^T c
	 */
	inline vector2 solve(const Eigen::Ref<const matrix> & Q, const Eigen::Ref<const vector> & c, vector & x) {
		if(Q.rows() == 0) { //no variables
			x = vector::Zero(0); 
			return (vector2() << 0, 0).finished();
		}
		bool isZeroQ = Q.isZero();
		Eigen::FullPivLU<matrix> luQ;
		index rQ;
		if(isZeroQ) {
			rQ = 0;
		} else {
			luQ.compute(Q);
			rQ = luQ.rank(); //rank of Q3FF
		}

		if(rQ==0) { //approximately zero matrix
			x = vector::Zero(Q.rows());
			if(c.isZero()) { //approximately zero
				return (vector2() << 0,0).finished(); 
			} else { //unbounded
				index index_max = 0;
				scalar val_max = 0;
				for(index i=0; i<c.size(); ++i) {
					if(abs(c(i)) > abs(val_max)) {
						val_max = c(i);
						index_max = i;
					}
				}
				x(index_max) = 1; 
				if(val_max > 0) {
					x(index_max) *= -1;
				} 
				return (vector2() << 0, -1).finished();
			}
		}

		matrix PQright;
		PQright = luQ.permutationQ()*matrix::Identity(Q.rows(),Q.rows()); //since Q symmetric PQright.transpose()*Q*PQright has invertible top left rQ by rQ corner.

		if(rQ == Q.cols()) { //Q invertible
			x = luQ.solve(-c);
			return (vector2() << 0, 0).finished();
		}

		/* x == PQright*x1
		 */
		matrix Q1 = PQright.transpose()*Q*PQright;
		vector c1 = PQright.transpose()*c;

		Eigen::FullPivLU<matrix> beta(Q1.topLeftCorner(rQ,rQ)); 

		matrix gamma = beta.solve(Q1.topRightCorner(rQ,Q1.rows()-rQ));
		vector c2 = -gamma.transpose()*c.head(rQ) + c.tail(Q.rows()-rQ);
		vector x2 = vector::Zero(Q1.rows()-rQ);
		/* x1 == [-gamma; Identity(x2.size(),x2.size())]*x2+[-beta.solve(c.head(rQ)) ; 0]
		 */
		vector2 return_value;
		index index_max = 0;
		scalar val_max = 0;
		if(c2.isZero()) {
			return_value << 0,0;
		} else {
			for(index i=0; i<c2.size(); ++i) {
				if(abs(c2(i)) > abs(val_max)) {
					val_max = c2(i);
					index_max = i;
				}
			}
			x2(index_max) = 1;
			if(val_max > 0) {
				x2(index_max) *= -1;
			} 
			return_value << 0,-1;
		}
		if(val_max == 0) {
			x = PQright*(vector(Q1.rows()) << -beta.solve(c.head(rQ)), vector::Zero(Q1.rows()-rQ)).finished();
		} else {
			x = vector::Zero(Q.rows());
			vector col = (matrix(Q1.rows(), Q1.rows()-rQ) << -gamma, matrix::Identity(x2.size(), x2.size())).finished().col(index_max);
			x = PQright*(x2(index_max)*col+(vector(Q1.rows()) << -beta.solve(c.head(rQ)), vector::Zero(Q1.rows()-rQ)).finished());
		}
		return return_value;
	}

	
	/* Returns a fixed size size 2 vector of scalars, call it v. 
	 * 	v(0) == tx.tail(1)
	 * 	v(1) == ty.tail(1)
	 * where tx and ty are the primal and dual variables for the transformed problem used in the algorithm.
	 *	v == (0, 0) implies primal bounded and feasible.				(i) 
	 *	v(0) == 0 && v(1) < 0 implies primal unbounded (dual infeasible).		(ii)
	 *	v(0) > 0 && v(1) == 0 implies primal infeasible (dual unbounded).		(iii)
	 *	v(0) > 0 && v(1) < 0 implies primal unbounded or infeasible.			(iv)
	 * In case of (iv) call solve() again with with same A and b but Q and c guaranteed to make problem bounded, e.g. with Q==0 and c==0. If
	 * end up with (iii) or (iv) on second call then original problem was infeasible otherwise it was unbounded.
	 */
	inline vector2 solve(const Eigen::Ref<const matrix> & Q, const Eigen::Ref<const vector> & c, const Eigen::Ref<const matrix> & A,
		       	const Eigen::Ref<const vector> & b,
			vector & solution_primal_part, vector & solution_dual_part, cqp::scalar gamma = 0x1p-24) { //gamma denotes target maximum allowable duality gap
		index m = A.rows(); index n = A.cols();
		if(n == 0) { //no variables. We treat output of A*x as zero m-vector and objective function as constant 0.
			solution_primal_part = vector::Zero(0); solution_dual_part = vector::Zero(m);
			if(b.isZero() || (A.rows()==0)) {
				return (vector2() << 0, 0).finished(); //feasible
			} else {
				return (vector2() << 1, 0).finished(); //infeasible
			}
		}

		//transform into an equivalent problem meeting restrictions of homotopy_algorithm with easily identifiable strictly feasible point.
		index tm = m+1; index tn = n+2;
		matrix tQ = matrix::Zero(tn, tn); 
		tQ.topLeftCorner(n, n) = Q;
		/* If there is an optimal solution then there is an optimal solution (x,y) which 
		 * is a vertex of the polyhedron defined by
		 * 	[ Q -A^T][x] >= -c
		 * 	[-A   0 ][y] =  -b 
		 * 	[ I   0 ]    >=  0.
		 * We want to find an upper bound on the sizes (absolute values) of components of
		 * basic feasible solutions (vertices) to the above (by, e.g., using Cramer's rule with 
		 * determinant bounds). If the polytope is bounded then all feasible points also
		 * have components satisfying this size bound.
		 * We'll use Hadamard's inequality together with assumption coefficients are rational to find a bound.
		 */
		vector scale_factor_vector = vector::Ones(m+n+3); //for later use with purify() and in obtaining gamma 
		vector row_squared_norms = vector::Zero(m+n+3);
		scalar component_bound = 1;
		scalar scale_factors_product = 1;
		scalar min_nonzero_squared_norm = 1;
		for(index i=0; i<n; ++i) {//since the bottom n rows each have norm 1 we can exclude them from consideration.
			row_vector temp(m+n); //represents row(i) == col(i).transpose()
			temp << Q.row(i), A.col(i).transpose();
			scale_factor_vector(i) = scale_factor(temp);
			if(scale_factor_vector(i)) {
				scale_factors_product *= scale_factor_vector(i);
				row_squared_norms(i) = temp.squaredNorm();
				if(row_squared_norms(i)) {
					component_bound *= row_squared_norms(i);
					if(row_squared_norms(i) < min_nonzero_squared_norm) {
						min_nonzero_squared_norm = row_squared_norms(i);
					}
				}
			}
		}
		for(index i=0; i<m; ++i) {
			row_vector temp(n); 
			temp << A.row(i);
			scale_factor_vector(2+n+i) = scale_factor(temp);
			if(scale_factor_vector(2+n+i)) {
				scale_factors_product *= scale_factor_vector(2+n+i);
				row_squared_norms(2+n+i) = temp.squaredNorm();
				if(row_squared_norms(2+n+i)) {
					component_bound *= row_squared_norms(2+n+i);
					if(row_squared_norms(2+n+i) < min_nonzero_squared_norm) {
						min_nonzero_squared_norm = row_squared_norms(2+n+i);
					}
				}
			}
		}
		scalar max_abs_det_sub_squared = scale_factors_product*scale_factors_product*component_bound; //Square of upper bound for maximum determinant of a submatrix of F*M (M row scaled to make coefficients integer).
		component_bound = component_bound/min_nonzero_squared_norm;
		component_bound *= (vector(m+n) << c, b).finished().squaredNorm();
		component_bound = sqrt(component_bound);
		component_bound *= scale_factors_product;

		vector comp_bounds = vector::Constant(n, component_bound);
		bool is_infeasible = false;
		for(index i=0; i < m; ++i) {
			scalar new_bound;
			if(A.row(i).maxCoeff() <= 0 || A.row(i).minCoeff() >= 0) {
				for(index j=0; j<n; ++j) {
					if(A(i,j) != 0) {
						new_bound = b(i)/A(i,j);
						if(new_bound < comp_bounds(j)) {
							comp_bounds(j) = new_bound;
							if(new_bound < 0) {
								is_infeasible = true;
								break;
							}
						}
					}
				}
			}
		}

		scalar tb_back = comp_bounds.sum();//n*component_bound;
#if LOG_SOLVE_STANDARD
		std::cout << "\ncomp_bounds.sum() == " << float_scalar(tb_back) << "\n";
		std::cout << "n*component_bound == " << float_scalar(n*component_bound);
#endif
		tb_back = simple_fraction(tb_back+0.5, tb_back+1.5);
#if LOG_SOLVE_STANDARD
		std::cout << "\ntb_back == " << float_scalar(tb_back) << "\n";
#endif
		if(is_infeasible) {
			solution_primal_part = comp_bounds;
			solution_dual_part = vector::Zero(m);
			return (vector2() << 1, 0).finished();
		}

		/*
		scalar tb_back = n*component_bound/tn;
		tb_back = scalar(tb_back.get_num()/tb_back.get_den()); //==floor(tb_back)
		tb_back = tn*(1+tb_back);
		*/
		vector tb(tm);
		tb << b, tb_back;

		matrix tA = matrix::Zero(tm, tn);
		tA.topLeftCorner(m, n) = A;
		tA.topRightCorner(m,1) = tn*b/tb_back - A*vector::Ones(n); //choose such that initial strictly feasible point meeting requirements of homotopy algorithm easily identifiable. 
		tA.bottomRows(1) = matrix::Constant(1, tn, 1);

		scalar tc_back; //choose tc_back such that tc_back > tA.topRightCorner(m,1).dot(y) for at least one y such that there exists an x such that (x, y) is an optimal solution to the original dual
		tc_back = tA.topRightCorner(m,1).cwiseAbs().sum()*component_bound;
		tc_back = simple_fraction(tc_back, tc_back+1);
		/*
		tc_back = scalar(tc_back.get_num()/tc_back.get_den())+1; //==1+floor(tc_back)
		*/
		vector tc(tn);
		tc << c, 0, tc_back;

		scale_factor_vector(n+1) = scale_factor((vector(2+m) << tA.topRightCorner(m+1,1), tc_back).finished().transpose());
		/* scale_factor_vector(i) now holds scale_factor for row i of transformed system
		 * 	[ tQ -tA^T][tx]  >= -tc
		 * 	[-tA    0 ][ty]  =  -tb 
		 * 	[  I    0 ]      >=   0.
		 */

		scalar beta = 1/scalar(1+sqrt(scalar(1+sqrt(scalar(1,tn))))); //= sqrt(static_cast<scalar>(tn)+sqrt(static_cast<scalar>(tn)))-sqrt(static_cast<scalar>(tn));

		vector tx = matrix::Constant(tn,1,tb_back/tn); //initial x for homotopy_algorithm
		vector ty = vector::Zero(tm);
		if(!tA.isZero(0)) {
			ty = (tA*tA.transpose()).fullPivLu().solve(tA*(tQ*tx+tc));
		}
		scalar eta = sqrt((tx.cwiseProduct(tQ*tx+tc-tA.transpose()*ty)).squaredNorm())/beta; 
		eta = simple_fraction(eta, eta*scalar(5,4));
		ty(tm-1) += -eta*tn/tb_back;
		
		scalar sqrtgamma, eq_err;
		if(gamma<=scalar(0)) {
			sqrtgamma = 1/((scale_factor_vector.sum()+1)*sqrt(max_abs_det_sub_squared));
			gamma = sqrtgamma*sqrtgamma;
			eq_err = sqrtgamma;
		} else {
			sqrtgamma = sqrt(gamma);
			eq_err = 0;
		}
#if LOG_SOLVE_STANDARD
		int format_width = 27; //testing
		std::cout << "\ntQ == \n" << tQ << "\n\ntc == \n" << tc << "\n\ntA == \n" << tA << "\n\ntb == \n" << tb << "\n\ntx == \n" << tx;
		std::cout << "\n\nscale_factor_vector max == \n" << scale_factor_vector.maxCoeff() << std::endl;

		std::cout << "\n\n" << std::setw(format_width) << std::left << "tn == " << tn
			  << "\n" << std::setw(format_width) << std::left << "tm == " << tm;
		std::cout << "\n" << std::setw(format_width) << std::left << "beta == " << float_scalar(beta)
			  << "\n" << std::setw(format_width) << std::left << "gamma == " << float_scalar(gamma)
			  << "\n" << std::setw(format_width) << std::left << "eta == " << float_scalar(eta)
			  << "\n" << std::setw(format_width) << std::left << "max_abs_det_sub == " << float_scalar(sqrt(max_abs_det_sub_squared)) << std::endl;
		/*
		std::cout << "\n\ntA*tx-tb == \n" << tA*tx-tb << std::endl;
		std::cout << "\n\nscale_factor_vector == \n" << scale_factor_vector.transpose() << std::endl;
		*/
#endif
#if USE_FLOAT_HOMOTOPY_ALGORITHM
		float_homotopy_algorithm(tQ, tc, tA, tb, tx, beta, gamma, eta, tx, ty); 
#else
		homotopy_algorithm(tQ, tc, tA, tb, tx, beta, gamma, eta, tx, ty, eq_err); 
#endif

		vector2 return_value;
		return_value << tx(tn-1), ty(tm-1);

		cqp::vector tw, tz;
		cqp::scalar tgap;
		tz = (tQ*tx+tc-tA.transpose()*ty);
		tw = tA*tx - tb;
		tgap = tx.dot(tz);

		vector backup_tx = tx;
		vector backup_ty = ty;
#if LOG_SOLVE_STANDARD
		/*
		std::cout << "\n\ntx == \n" << tx;
		std::cout << "\n\nty == \n" << ty;
		std::cout << "\n\ntz == \n" << tz;
		std::cout << "\n\ntw == \n" << tw;
		*/

		format_width = 30;
		std::cout << "\n\n" << std::setw(format_width) << std::left << "predicted_gap == tn*gamma == " << float_scalar(tn*gamma);
		std::cout << "\n" << std::setw(format_width) << std::left << "tgap == tx^T*tz == " << float_scalar(tgap);
		cqp::scalar teq_gap = tw.transpose()*ty;
		std::cout << "\n" << std::setw(format_width) << std::left << "teq_gap == tw^T*ty == " << float_scalar(teq_gap);
		std::cout << "\n" << std::setw(format_width) << std::left << "ttotal_gap == tgap+teq_gap == " << float_scalar(tgap+teq_gap) << std::endl;
#endif
		scalar feas_error = max(cqp::matrix::Zero(tn,1).cwiseMax(-tx).maxCoeff(), cqp::matrix::Zero(tn,1).cwiseMax(-tz).maxCoeff());
		feas_error = max(feas_error, tw.cwiseAbs().maxCoeff());
		scalar abs_gap = abs(tgap);

		scalar backup_feas_error = feas_error;
		scalar backup_abs_gap = abs_gap;

#if USE_FLOAT_HOMOTOPY_ALGORITHM
		scalar max_coeff(tx.cwiseProduct(tQ*tx+tc-tA.transpose()*ty).cwiseAbs().maxCoeff());
		if(max_coeff > gamma) { //corrected sqrtgamma
			unsigned long prec = mpf_get_default_prec(); //make sqrt error < 2^{-prec}
			unsigned long prec2 = prec;
			prec2 += 5+std::max(mpz_sizeinbase(max_coeff.get_num().get_mpz_t(), 2),mpz_sizeinbase(max_coeff.get_den().get_mpz_t(), 2))/2;
			if(prec2 >= mpz_sizeinbase(max_coeff.get_den().get_mpz_t(), 2)) {
				prec2 -= mpz_sizeinbase(max_coeff.get_den().get_mpz_t(), 2);
			} else {
				prec2 = 0;
			}
			scalar eps(1,1);
			mpz_ui_pow_ui(eps.get_den().get_mpz_t(), 2, std::min(2*prec, prec2));
			eps.canonicalize();
			sqrtgamma = sqrt(max_coeff, eps);
		}
#endif

#if LOG_SOLVE_STANDARD
		mpz_class tx_length, ty_length;
		tx_length = get_max_length(tx, 10); 
		ty_length = get_max_length(ty, 10); 
		std::cout << "\n\nunpurified\n";
		std::cout << "tx_length == " << tx_length << "\n";
		std::cout << "ty_length == " << ty_length;
		std::cout << "\ntx == \n" << tx;
		std::cout << "\nty == \n" << ty;
		/*
		std::cout << "\n\ntx == \n" << tx;
		std::cout << "\n\nty == \n" << ty;
		std::cout << "\n\ntz == \n" << tz;
		std::cout << "\n\ntw == \n" << tw;
		*/
#endif

		purify(tQ, tc, tA, tb, tx, ty, sqrtgamma, tx, ty); 

		tz = (tQ*tx+tc-tA.transpose()*ty);
		tw = tA*tx - tb;
#if LOG_SOLVE_STANDARD
		tx_length = get_max_length(tx, 10); 
		ty_length = get_max_length(ty, 10); 
		std::cout << "\n\npurified\n";
		std::cout << "tx_length == " << tx_length << "\n";
		std::cout << "ty_length == " << ty_length;
		std::cout << "\ntx == \n" << tx;
		std::cout << "\nty == \n" << ty << "\n";
		/*
		std::cout << "\n\ntx == \n" << tx;
		std::cout << "\n\nty == \n" << ty;
		std::cout << "\n\ntz == \n" << tz;
		std::cout << "\n\ntw == \n" << tw;
		*/
#endif
		abs_gap = tx.dot(tz);
		feas_error = max(cqp::matrix::Zero(tn,1).cwiseMax(-tx).maxCoeff(), cqp::matrix::Zero(tn,1).cwiseMax(-tz).maxCoeff());
		feas_error = max(feas_error, tw.cwiseAbs().maxCoeff());
		if(abs_gap<0) {
			abs_gap = -abs_gap;
		}
#if LOG_SOLVE_STANDARD
		format_width = 21;
		std::cout << '\n';
		std::cout << std::setw(format_width) << std::left << "backup_feas_error == " << float_scalar(backup_feas_error) << std::endl;
		std::cout << std::setw(format_width) << std::left << "backup_abs_gap == " << float_scalar(backup_abs_gap) << std::endl;
		std::cout << std::setw(format_width) << std::left << "feas_error == " << float_scalar(feas_error) << std::endl;
		std::cout << std::setw(format_width) << std::left << "abs_gap == " << float_scalar(abs_gap) << std::endl;
#endif
		if(feas_error > 0) { 
			tx = backup_tx;
			ty = backup_ty;
#if LOG_SOLVE_STANDARD
			std::cout << "\npurify() failed to find better solution\n";
#endif
		} else if(abs_gap > backup_abs_gap && backup_feas_error==0) {
			tx = backup_tx;
			ty = backup_ty;
#if LOG_SOLVE_STANDARD
			std::cout << "\npurify() failed to find better solution\n";
#endif
		} else {
			return_value << 0,0; 
		}
		solution_primal_part = tx.head(n);
		solution_dual_part = ty.head(m);

		return return_value;
	}


	/* general form
	 * input problem (P)
	 * 	min  x^T 0.5*Q x + x^T c
	 * 	s.t. A*x == a
	 * 	     B*x >= b 
	 *
	 * dual problem (D)
	 * 	max_{x,y,z} x^T*0.5*Q*x + x^T*c - y^T*(A*x-a) - z^T*(B*x-b)
	 * 	s.t	    Q*x - A^T*y - B^T*z == -c
	 * 				      z >= 0
	 *
	 * output (x,y,z) solving the KKT system
	 * 	Q*x - A^T*y - B^T*z == -c
	 * 	A*x == a
	 * 	B*x >= b
	 * 	z   >= 0
	 * 	z^T*(B*x - b) == 0 
	 */
	/* To do:
	 * Remove unncessary intermediate matrices and vectors. Most of them are just there to help improve conceptual clarity.
	 */
	inline vector2 solve(const Eigen::Ref<const matrix> & Q, const Eigen::Ref<const vector> & c, 
			const Eigen::Ref<const matrix> & A, const Eigen::Ref<const vector> & a, 
			const Eigen::Ref<const matrix> & B, const Eigen::Ref<const vector> & b, 
			vector & x, vector & y, vector & z, cqp::scalar gamma = 0x1p-24) {
		/* introduce slack variables to get problem (P1)
		 *	min x1^T*0.5*Q1*x1 + x1^T*c1
		 *	s.t. A1*x1 == a1 
		 *		 s >= 0
		 * where
		 * 	Q1 == [Q 0; 0 0]
		 * 	c1 == [c; 0]
		 * 	A1 == [A 0; B -I]
		 * 	x1 == [x; s]
		 * and we're using the MATLAB notation for matrices/block matrices, i.e. semicolon in brackets denotes new row/rows space between identifiers denotes new column/columns.
		 */
		index ns = B.rows(); //number slack variables
		index n0 = Q.rows(); //number original variables, == A.cols(), == B.cols()
		if(n0 == 0) { //no variables. We treat output of A*x and B*x as zero vectors and objective function as constant 0.
			x = vector::Zero(0); y = vector::Zero(A.rows()); z = vector::Zero(B.rows());
			if(((A.rows() > 0 && a.isZero()) || A.rows()==0) && ((B.rows() > 0 && (b.array()<=0).all()) || B.rows()==0)) {
				return (vector2() << 0, 0).finished(); //feasible
			} else {
				return (vector2() << 1, 0).finished(); //infeasible
			}
		}
		index n1 = n0+ns;
		matrix Q1 = matrix::Zero(n1, n1);
		Q1.topLeftCorner(n0, n0) = Q;

		vector c1 = vector::Zero(n1);
		c1.head(n0) = c;

		index m0 = A.rows(), m1 = m0+ns;
		matrix A1 = matrix::Zero(m1, n1);
		A1.topLeftCorner(m0, n0) = A;
		A1.bottomLeftCorner(ns, n0) = B;
		A1.bottomRightCorner(ns, ns) = -matrix::Identity(ns, ns); //B1 == [Zero(ns,n0) Identity(ns,ns)]		b1 == Zero(ns,1)
		vector a1(m1);
		a1 << a, b;
		if(m1==0) { //no constraints
			y = vector::Zero(0);
			z = vector::Zero(0);
			return solve(Q,c,x);
		}
#if LOG_SOLVE_GENERAL
		std::cout << "Q1 == \n" << Q1 << '\n';
		std::cout << "c1 == \n" << c1 << '\n';
		std::cout << "A1 == \n" << A1 << '\n';
		std::cout << "a1 == \n" << a1 << '\n';
#endif
		/* For each positive integer i let (Pi) refer to problem
		 * 	min xi^T*0.5*Qi*xi+xi^T*ci
		 * 	s.t. Ai*xi == ai
		 * 	     Bi*xi >= bi,
		 * where Qi is symmetric positive semidefinite;
		 * let xiG denote the subvector of those components of xi appearing with nonzero coefficient in at least one inequality constraint,
		 * let xiF denote the subvector of those components of xi with 0 coefficient in every inequality constraint,
		 * and let (xi, yi, zi) refer to a solution of the corresponding KKT system (KKTi).
		 *
		 * If R and M are invertible, d a vector (all with appropriate dimensions for the following), i and j positive integers,
		 * Qj = M^T*Qi*M, cj = M^T*(ci+Qi*d), Aj == R*Ai*M, aj = R*(ai-Ai*d), Bj == Bi*M, and bj == bi-Bi*d
		 * then (M*xj+d, R^T*yj, zj) is a solution to KKTi.
		 *
		 * x1 == [x; s]	
		 * y1 == [y; z] 
		 * z1 == z
		 *
		 * where [x; y; z] is a solution to KKT system for original problem (P)
		 */
		Eigen::FullPivLU<matrix> luA1F(A1.leftCols(n0)); //LU factorization of part of A1 corresponding to those variables not appearing in the nonnegativity constraints
		matrix LA1F = matrix::Identity(m1, m1);
		LA1F.block(0,0,m1,std::min(m1,n0)).triangularView<Eigen::StrictlyLower>() = luA1F.matrixLU().block(0,0,m1,std::min(m1,n0));
		matrix UA1F = matrix::Zero(m1,n0);
		UA1F = luA1F.matrixLU().triangularView<Eigen::Upper>();
		index rA1F = luA1F.rank(); //number unbounded variables being removed
		matrix Pleft = matrix::Identity(m1,m1);
		matrix PA1Fright = matrix::Identity(n0,n0); //Pleft*A1F*PA1Fright == L*U
		matrix Pright = matrix::Identity(n1,n1);
		if(rA1F>0) {
			Pleft = luA1F.permutationP()*matrix::Identity(m1,m1);
			PA1Fright = luA1F.permutationQ()*matrix::Identity(n0,n0); //Pleft*A1F*PA1Fright == L*U
			Pright = matrix::Identity(n1,n1);
			Pright.topLeftCorner(n0,n0) = PA1Fright;
		}
		/*
		 * Pright == [PA1Fright 0; 0 I]
		 * x2 == Pright^T*x1 				x1 == Pright*x2
		 * y2 == ((LA1F^-1*Pleft)^T)^-1*y1		y1 == (LA1F^-1*Pleft)^T*y2
		 * z2 == z1					z1 == z2
		 *
		 * A2 == LA1F^-1*Pleft*A1*Pright		a2 == LA1F^-1*Pleft*a1
		 * B2 == B1 == [0_{ns,n0} I_{ns}]	 	b2 == b1 == 0_{ns,1}
		 * Q2 = Pright^T*Q1*Pright			c2 == Pright^T*c1
		 */
		matrix A2 = LA1F.fullPivLu().solve(Pleft*A1*Pright);
		vector a2 = LA1F.fullPivLu().solve(Pleft*a1);
		matrix Q2 = Pright.transpose()*Q1*Pright;
		matrix c2 = Pright.transpose()*c1;
#if LOG_SOLVE_GENERAL
		std::cout << "Q2 == \n" << Q2 << '\n';
		std::cout << "c2 == \n" << c2 << '\n';
		std::cout << "A2 == \n" << A2 << '\n';
		std::cout << "a2 == \n" << a2 << '\n';
#endif
		index n3F = n0 - rA1F; //remaining number of unbounded variables
		index n3 = n1 - rA1F; //remaining number of variables		n3G == ns == n1G
		matrix UA1Fbase = UA1F.topLeftCorner(rA1F, rA1F); //invertible top left block of A2
		matrix H1 = matrix::Zero(m1,rA1F);
		H1.topRows(rA1F) = matrix::Identity(rA1F, rA1F);
		matrix H2 = matrix::Zero(m1,m1-rA1F);
		H2.bottomRows(m1-rA1F) = matrix::Identity(m1-rA1F, m1-rA1F);

		matrix J1 = matrix::Zero(n1,rA1F);
		J1.topRows(rA1F) = matrix::Identity(rA1F, rA1F);
		matrix J2 = matrix::Zero(n1,n1-rA1F);
		J2.bottomRows(n1-rA1F) = matrix::Identity(n1-rA1F, n1-rA1F);

		matrix M = matrix::Zero(n1, n3);
		if(rA1F>0) {
			M.topRows(rA1F) = UA1Fbase.fullPivLu().solve(-A2.topRightCorner(rA1F, n3));
		}
		M.bottomRows(n3) = matrix::Identity(n3,n3);
		vector mu = vector::Zero(n1);
		if(rA1F>0) {
			mu.head(rA1F) = UA1Fbase.fullPivLu().solve(a2.head(rA1F));
		}

		index m3 = m1-rA1F; //remaining number equality constraints
		matrix A3 = (A2*M).bottomRows(m3);
		vector a3 = (a2 - A2*mu).tail(m3);
		matrix Q3 = M.transpose()*Q2*M;
		matrix c3 = M.transpose()*(c2+Q2*mu); //B3 == [Zero(ns,n3F) Identity(ns, ns)]
		/*
		 * x2 == M*x3+mu
		 * y2 == H2*y3 + (rA1F>0 ? H1*UA1Fbase.transpose().inverse()*J1.transpose()*(Q2*(M*x3+mu)-A2.transpose()*H2*y3-B2.transpose()*z3+c2) : 0)
		 * z2 == z3
		 *
		 * In P3 unbounded variables only appear in the objective function not in any of the constraints. 
		 */
		index rQ3FF;
		matrix PQ3FFright;
		if(n3F>0) {//we will disregard this LU factorization because it's left and right permutation matrices are not necessarily transposes of one another.
			Eigen::FullPivLU<matrix> luQ3FF(Q3.topLeftCorner(n3F,n3F)); //call top left n3F by n3F corner Q3FF
			rQ3FF = luQ3FF.rank(); //rank of Q3FF
			PQ3FFright = luQ3FF.permutationQ()*matrix::Identity(n3F,n3F); //since Q3FF symmetric PQ3FFright.transpose()*Q3FF*PQ3FFright has invertible top left rQ3FF by rQ3FF block.
		} else { 
			rQ3FF = 0;
			PQ3FFright = matrix::Identity(0,0);
		}

		matrix PQ3right = matrix::Identity(n3,n3);
		PQ3right.topLeftCorner(n3F,n3F) = PQ3FFright;
#if LOG_SOLVE_GENERAL
		std::cout << "Q3 == \n" << Q3 << '\n';
		std::cout << "c3 == \n" << c3 << '\n';
		std::cout << "A3 == \n" << A3 << '\n';
		std::cout << "a3 == \n" << a3 << '\n';
#endif

		matrix A4 = A3*PQ3right; //n4F == n3F, n4 == n3
		vector a4 = a3;
		matrix Q4 = PQ3right.transpose()*Q3*PQ3right; //Q4 has invertible top left rQ3FF by rQ3FF corner
		vector c4 = PQ3right.transpose()*c3; //B4 == B3		b4 == b3 == 0
		/*
		 * x3 == PQ3right*x4
		 * y3 == y4
		 * z3 == z4
		 */
		Eigen::FullPivLU<matrix> beta;
		if(rQ3FF>0) {
			beta.compute(Q4.topLeftCorner(rQ3FF,rQ3FF));
		}
		index M2cols = n3-rQ3FF; //n4_and_a_half
		matrix M2 = matrix::Zero(n3, M2cols);
		M2.bottomRows(M2cols) = matrix::Identity(M2cols,M2cols);
		vector mu2 = vector::Zero(n3);
		if(rQ3FF>0) {
			M2.topRows(rQ3FF) = beta.solve(-Q4.topRightCorner(rQ3FF,M2cols));
			mu2.head(rQ3FF) = beta.solve(-c4.head(rQ3FF));
		}
#if LOG_SOLVE_GENERAL
		std::cout << "Q4 == \n" << Q4 << '\n';
		std::cout << "c4 == \n" << c4 << '\n';
		std::cout << "A4 == \n" << A4 << '\n';
		std::cout << "a4 == \n" << a4 << '\n';
#endif

		matrix negM2TLC = -M2.topLeftCorner(rQ3FF, n3F-rQ3FF);
		//n3-n3F == number variables in P5 == number nonnegative in P4 == number nonnegative in P1 == ns

		matrix A5 = matrix::Zero(m3+n3F-rQ3FF,ns);
		A5.topRows(n3F-rQ3FF) = negM2TLC.transpose()*Q4.topRightCorner(rQ3FF, ns) - Q4.block(rQ3FF, n3F, n3F-rQ3FF, ns);
		A5.bottomRows(m3) = A4.rightCols(ns);
		vector a5 = vector::Zero(A5.rows());
		a5.head(n3F-rQ3FF) = -negM2TLC.transpose()*c4.head(rQ3FF)+c4.segment(rQ3FF, n3F-rQ3FF);
		a5.tail(m3) = a4;
		matrix Q5, c5;
		if(rQ3FF>0) {
			Q5 = Q4.bottomRightCorner(ns,ns)-Q4.topRightCorner(rQ3FF, ns).transpose()*beta.solve(Q4.topRightCorner(rQ3FF, ns));
			c5 = c4.tail(ns)-Q4.topRightCorner(rQ3FF, ns).transpose()*beta.solve(c4.head(rQ3FF));
		} else {
			Q5 = Q4.bottomRightCorner(ns,ns);
			c5 = c4.tail(ns);
		}
#if LOG_SOLVE_GENERAL 
		std::cout << "Q5 == \n" << Q5 << '\n';
		std::cout << "c5 == \n" << c5 << '\n';
		std::cout << "A5 == \n" << A5 << '\n';
		std::cout << "a5 == \n" << a5 << '\n';
#endif
		vector x5(ns), y5(A5.rows()), z5(ns);
		vector2 return_value(solve(Q5, c5, A5, a5, x5, y5, gamma));
		z5 = Q5*x5 - A5.transpose()*y5+c5;

		/* back substitute
		 * x4 == M2*[y5.head(n3F-rQ3FF); x5] + mu2
		 * y4 == y5.tail(m3)
		 * z4 == z5
		 *
		 * x3 == PQ3right*x4
		 * y3 == y4
		 * z3 == z4
		 *
		 * x2 == M*x3+mu
		 * y2 == H2*y3 + (rA1F>0 ? H1*UA1Fbase.transpose().inverse()*J1.transpose()*(Q2*(M*x3+mu)-A2.transpose()*H2*y3-B2.transpose()*z3+c2) : 0)
		 * z2 == z3
		 *
		 * x1 == Pright*x2
		 * y1 == (LA1F^-1*Pleft)^T*y2
		 * z1 == z2
		 *
		 * x == x1.head(n0)
		 * y == y1.head(m0)
		 * z == z1
		 */
		vector x4, x3, x2, x1, y4, y3, y2, y1;
		x4 = M2*(vector(M2cols) << y5.head(n3F-rQ3FF), x5).finished() + mu2;
		x3 = PQ3right*x4;
		x2 = M*x3+mu;
		x1 = Pright*x2;
		x = x1.head(n0);

		matrix B2 = matrix::Zero(ns,n0+ns);
		B2.rightCols(ns) = matrix::Identity(ns,ns);
		y4 = y5.tail(m3);
		y3 = y4;
		if(rA1F>0) {
			y2 = H2*y3 + H1*UA1Fbase.transpose().fullPivLu().solve(J1.transpose()*(Q2*(M*x3+mu)-A2.transpose()*H2*y3-B2.transpose()*z5+c2));
		} else {
			y2 = H2*y3;
		}
		y1 = Pleft.transpose()*LA1F.transpose().fullPivLu().solve(y2);
		y = y1.head(m0);
		
		z = z5;

		vector primal_feas_error, dual_feas_error;
		primal_feas_error = cqp::vector::Zero(m0+ns);
		primal_feas_error.head(m0) = A*x-a;
		primal_feas_error.tail(ns) = primal_feas_error.tail(ns).cwiseMax(-(B*x-b));
		dual_feas_error = cqp::vector::Zero(n0+ns);
		dual_feas_error.head(n0) = Q*x-A.transpose()*y-B.transpose()*z+c;
		dual_feas_error.tail(ns) = dual_feas_error.tail(ns).cwiseMax(-z);
		return_value << primal_feas_error.cwiseAbs().maxCoeff(), -dual_feas_error.cwiseAbs().maxCoeff();

#if LOG_SOLVE_GENERAL
		matrix B3(ns, ns+n3F); B3 << matrix::Zero(ns,n3F), matrix::Identity(ns, ns); //B3 == B4
		std::cout << "x5 == " << x5.transpose() << '\n';
		std::cout << "y5 == " << y5.transpose() << '\n';
		std::cout << "z5 == " << z5.transpose() << '\n';
		std::cout << "A5*x5-a5 == " << (A5*x5-a5).transpose() << '\n';
		std::cout << "gap5 == " << x5.dot(z5) << "\n\n";

		std::cout << "x4 == " << x4.transpose() << '\n';
		std::cout << "y4 == " << y4.transpose() << '\n';
		std::cout << "z4 == " << z5.transpose() << '\n';
		std::cout << "A4*x4-a4 == " << (A4*x4-a4).transpose() << '\n';
		std::cout << "B4*x4-b4 == " << (B3*x4).transpose() << '\n';
		std::cout << "gap4 == " << z5.transpose()*(B3*x3) + y4.transpose()*(A4*x4-a4) << "\n\n";

		std::cout << "x3 == " << x3.transpose() << '\n';
		std::cout << "y3 == " << y3.transpose() << '\n';
		std::cout << "z3 == " << z5.transpose() << '\n';
		std::cout << "A3*x3-a3 == " << (A3*x3-a3).transpose() << '\n';
		std::cout << "B3*x3-b3 == " << (B3*x3).transpose() << '\n';
		std::cout << "gap3 == " << z5.transpose()*(B3*x3) + y3.transpose()*(A3*x3-a3) << "\n\n";

		std::cout << "x2 == " << x2.transpose() << '\n';
		std::cout << "y2 == " << y2.transpose() << '\n';
		std::cout << "z2 == " << z5.transpose() << '\n';
		std::cout << "A2*x2-a2 == " << (A2*x2-a2).transpose() << '\n';
		std::cout << "B2*x2-b2 == " << (B2*x2).transpose() << '\n';
		std::cout << "gap2 == " << z5.transpose()*(B2*x2) + y2.transpose()*(A2*x2-a2) << "\n\n";

		std::cout << "x1 == " << x1.transpose() << '\n';
		std::cout << "y1 == " << y1.transpose() << '\n';
		std::cout << "z1 == " << z5.transpose() << '\n';
		std::cout << "A1*x1-a1 == " << (A1*x1-a1).transpose() << '\n';
		std::cout << "B1*x1-b1 == " << (B2*x1).transpose() << '\n';
		std::cout << "gap1 == " << z5.transpose()*(B2*x1) + y1.transpose()*(A1*x1-a1) << "\n\n";

		std::cout << "x == " << x.transpose() << '\n';
		std::cout << "y == " << y.transpose() << '\n';
		std::cout << "z == " << z.transpose() << '\n';
		std::cout << "A*x-a == " << (A*x-a).transpose() << '\n';
		std::cout << "B*x-b == " << (B*x-b).transpose() << '\n';
		std::cout << "gap == " << z.transpose()*(B*x-b) + y.transpose()*(A*x-a) << "\n\n";
#endif
		return return_value;
	} //end of solve

} //end of cqp namespace
#endif
