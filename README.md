# cqp
An implementation of Paul Tseng's [A Simple Polynomial-Time Algorithm for Convex Quadratic Programming](https://core.ac.uk/download/pdf/4380751.pdf). This is an interior point method using a logarithmic barrier function. The complexity of the algorithm is O(m^(1/2)L) iterations and O(m^(7/2)L) arithmetic operations where m is the number of variables and L is the size of the problem encoding in binary.