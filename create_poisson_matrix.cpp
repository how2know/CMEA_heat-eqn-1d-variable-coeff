#include "create_poisson_matrix.hpp"

//! Used for filling the sparse matrix.
using Triplet = Eigen::Triplet<double>;

//! Create the 1D Poisson matrix
//! @param[in] N the number of interior points
//! @param[in] a the coefficient function a
//!
//! @returns the Poisson matrix.

SparseMatrix createPoissonMatrix(int N, const std::function<double(double)>& a)
{
    // matrix A to return (sparse form because it contains mostly 0's)
    SparseMatrix A;

// (write your solution here)

    /// Start of my solution ///

    // matrix A is N x N
    A.resize(N, N);

    // vector of triplet to fill the matrix A
    std::vector<Triplet> triplets;

    // number of triplets needed: N * 3 - 2
    triplets.reserve(N * 3 - 2);

    // step size
    double h = 1. / (N+1);

    for (int i = 0; i < N; i++)
    {
        // x = i * h   with   i = 1, 2, ..., N
        double x_i = (i+1) * h;

        // compute the value of the function a(x) at x = x_i
        double a_i = a(x_i);

        // prepare the triplets to fill the matrix A => Triplet(row, column, value)
        // first row
        if (i == 0)
        {
            triplets.push_back(Triplet(i, i, 2.*a_i));
            triplets.push_back(Triplet(i, i+1, -a_i));
        }
        // last row
        else if (i == N-1)
        {
            triplets.push_back(Triplet(i, i, 2.*a_i));
            triplets.push_back(Triplet(i, i-1, -a_i));
        }
        // all rows in between
        else
        {
            triplets.push_back(Triplet(i, i-1, -a_i));
            triplets.push_back(Triplet(i, i, 2.*a_i));
            triplets.push_back(Triplet(i, i+1, -a_i));
        }
    }

    // fill the matrix with the triplets
    A.setFromTriplets(triplets.begin(), triplets.end());

    // multiply each element of A by (1/h^2)
    A = (1 / (h*h)) * A;

    /// End of my solution ///

    return A;
}

