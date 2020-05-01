#include "forward_euler.hpp"
#include "create_poisson_matrix.hpp"

//! Uses the explicit forward Euler method to compute u from time 0 to time T
//!
//! @param[in] u0 the initial data, as column vector (size N+2)
//! @param[in] dt the time step size
//! @param[in] T the final time at which to compute the solution (which we assume to be a multiple of dt)
//! @param[in] N the number of interior grid points
//! @param[in] gL function of time with the Dirichlet condition at left boundary
//! @param[in] gR function of time with the Dirichlet condition at right boundary
//! @param[in] a the coefficient function a
//!
//! @return u at all time steps up to time T, each column corresponding to a time step (including the initial condition as first column)
//!
//! @note the vector returned should include the boundary values!

/*  I tried to use a function to compute G, but for some reasons it didn't compile...

void computeG(Eigen::VectorXd G, int N, double t,
              const std::function<double(double)>& gL,
              const std::function<double(double)>& gR,
              const std::function<double(double)>& a)
{
    // G is a vector of size N
    G.resize(N);

    // fill G with 0's
    G.setZero(N);
    //G = Eigen::VectorXd::Zero(N);

    // spatial step size
    double h = 1. / (N + 1);

    // compute the first and the last elements of G with the boundary condition
    G[0] = a(h) * gL(t);
    G[N-1] = a(1-h) * gR(t);
}
*/

std::pair<Eigen::MatrixXd, Eigen::VectorXd> forwardEuler(const Eigen::VectorXd& u0,
                                                         double dt, double T, int N,
                                                         const std::function<double(double)>& gL,
                                                         const std::function<double(double)>& gR,
                                                         const std::function<double(double)>& a)
{
    // time step size
    const int nsteps = int(round(T / dt));

    // spacial step size
    const double h = 1. / (N + 1);

    // matrix that stores the vectors u(t) from t = 0 to t = T
    Eigen::MatrixXd u;
    u.resize(N + 2, nsteps + 1);

    // vector with all time steps from t = 0 to t = T
    Eigen::VectorXd time;
    time.resize(nsteps + 1);

// (write your solution here)

    /// Start of my solution ///

    // create the matrix A
    SparseMatrix A = createPoissonMatrix(N, a);

    // initialize vector G
    Eigen::VectorXd G;
    G.resize(N);
    G.setZero(N);

    // vector for the linear algebra
    Eigen::VectorXd u_k;
    u_k.resize(N);
    u_k = u0.segment(1, N);

    // store the initial value
    u.col(0) = u0;
    time(0) = 0.;

    for (int k = 0; k < nsteps; k++)
    {
        // next time
        time(k + 1) = (k+1) * dt;

        // copmute the first and the last element of U at the next time
        u(0, k+1) = gL(time(k+1));
        u(N+1, k+1) = gR(time(k+1));

        // computeG(G, N, t_k, gL, gR, a);   this was my attempt to compute G with a function...

        // compute the first and the last element of G
        G[0] = a(h) * gL(time(k)) / (h*h);
        G[N-1] = a(N*h) * gR(time(k)) / (h*h);

        // compute the vector u_k at the next time step and store it in U
        u_k = u_k - dt * A * u_k + dt * G;
        u.col(k+1).segment(1,N) = u_k;
    }

    /// End of my solution ///

    return std::make_pair(u, time);
}

