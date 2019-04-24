# cqp
This is the initial version of a convex quadratic program solver.  All algorithms are contained in the single header file cqp.hxx; the rest of the files are just for testing purposes. They rely on the Eigen C++ template library and GNU multiple precision arithmetic library, the specific headers used being <Eigen/Core>, <Eigen/LU>, and <gmpxx.h>. Requires linker flags -lgmpxx and -lgmp.

# Algorithms
Default data type for scalars is mpq_class. Although there is an implementation of a floating point homotopy algorithm which uses the mpf_class enabled via setting the preprocessor macro USE_FLOATING_POINT_HOMOTOPY_ALGORITHM to 1.

## homotopy_algorithm
The basis of this algorithm is [A Simple Polynomial-Time Algorithm for Convex Quadratic Programming](https://core.ac.uk/download/pdf/4380751.pdf) (Tseng, 1988). The complexity of the algorithm is O((n^0.5)L) iterations and O((n^3.5)L) arithmetic operations where n is the number of variables and L is the size of the problem encoding in binary. 

For the rational data type the algorithm was altered in order to reduce growth of rational number numerator/denominator via considering perturbed KKT conditions and using continued fractions to find simple rational numbers in given intervals. 

For the floating point implementation the algorithm was altered in order to identify primal variables near the boundary of the feasible region contributing to ill conditioning. After variables are identified the appropriate matrix block is altered in an attempt to reduce the condition number.

## solve (standard form)
This first transforms the initial problem into a new problem with easily identifiable interior point meeting the requirements of the homotopy_algorithm, before then calling said algorithm. For details see pages 22-28 of [An O(n^3L) Interior Point Algorithm for Convex Quadratic Programming](https://apps.dtic.mil/dtic/tr/fulltext/u2/a186001.pdf) (Monteiro & Adler, 1987) .

## solve (general form)
Converts problem from general form to standard form before passing to solve (standard form).

## purify
Converts an approximate solution with "small enough" error into an exact solution. For details see lemmas 1 and 2 of [An application of the Khachian-Shor algorithm to a class of linear complementarity problems](https://cowles.yale.edu/sites/default/files/files/pub/d05/d0549.pdf) (Adler, et al., 1980, pp. 6-7).

# Notes
Although the linked papers assume integral data when defining L, it's not hard to show the complexity stays in the same class for rational data. 

The implicit constant in the big O notation, after the transformation from solve(), seems to be quite big. The transformation itself seems to cause a lot of slowdown probably due to the large sizes (as in length of representation) of the numbers involved. This could be improved with good error bounds in order to help gain tighter bounds on the location of an optimal solution. Although these bounds exist, for example see [Error bounds for monotone linear complementarity problems](https://apps.dtic.mil/dtic/tr/fulltext/u2/a160975.pdf), computing them seems to be painful as you need to deal with algebraic numbers appearing in matrix square roots, a lot of painful computation would just yield a huge constant multiplier for the residual, which I may due anyway.
