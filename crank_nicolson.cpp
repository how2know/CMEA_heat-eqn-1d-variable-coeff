#include "crank_nicolson.hpp"

//! Uses the Crank-Nicolson method to compute u from time 0 to time T
//!
//! @param[in] u0 the initial data, as column vector (size N+2)
//! @param[in] dt the time step size
//! @param[in] T the final time at which to compute the solution (which we assume to be a multiple of dt)
//! @param[in] N the number of interior grid points
//! @param[in] gL function of time with the Dirichlet condition at left boundary
//! @param[in] gR function of time with the Dirichlet condition at right boundary
//! @param[in] a the coefficient function a
//!
//! @note the vector returned should include the boundary values!

std::pair<Eigen::MatrixXd, Eigen::VectorXd> crankNicolson(const Eigen::VectorXd& u0,
                                                          double dt, double T, int N,
                                                          const std::function<double(double)>& gL,
                                                          const std::function<double(double)>& gR,
                                                          const std::function<double(double)>& a)
{
    Eigen::VectorXd time;
    Eigen::MatrixXd u;

// (write your solution here)

    /// Start of my solution ///

    // time step size
    const int nsteps = int(round(T / dt));

    // spacial step size
    const double h = 1. / (N + 1);

    // matrix "u" stores the vectors u(t) from t = 0 to t = T
    u.resize(N + 2, nsteps + 1);

    // vector "time" stores all time steps from t = 0 to t = T
    time.resize(nsteps + 1);

    // create the matrix A
    SparseMatrix A = createPoissonMatrix(N, a);

    // identity matrix
    SparseMatrix I(N,N);
    I.setIdentity();

    // vector G at the current time
    Eigen::VectorXd G_current;
    G_current.resize(N);
    G_current.setZero(N);

    // vector G at the next time
    Eigen::VectorXd G_next;
    G_next.resize(N);
    G_next.setZero(N);

    // store the initial value
    u.col(0) = u0;
    time(0) = 0.;

    // vector u with at the current time
    Eigen::VectorXd u_current;
    u_current.resize(N);
    u_current = u0.segment(1, N);

    // vector u with at the next time
    Eigen::VectorXd u_next;
    u_next.resize(N);

    for (int k = 0; k < nsteps; k++)
    {
        // next time
        time(k + 1) = (k+1) * dt;

        // copmute the first and the last element of U at the next time
        u(0, k+1) = gL(time(k+1));
        u(N+1, k+1) = gR(time(k+1));


        // compute the first and the last element of G at the current time and at the next time
        G_current[0] = a(h) * gL(time(k)) / (h*h);
        G_current[N-1] = a(N*h) * gR(time(k)) / (h*h);
        G_next[0] = a(h) * gL(time(k+1)) / (h*h);
        G_next[N-1] = a(N*h) * gR(time(k+1)) / (h*h);

        // we now want to solve X * u_next = Y

        // matrix X
        SparseMatrix X = I + (dt/2.) * A;

        // vector Y that depends on u_current
        Eigen::VectorXd Y = (I - (dt/2.) * A) * u_current + (dt/2.) * (G_next + G_current);

        // solve for u_next

        // matrix to use the solver
        Eigen::SparseLU<SparseMatrix> solver;
        solver.compute(X);

        // solve X * u_next = Y for u_next
        u_next = solver.solve(Y);

        // store the next u in the matrix
        u.col(k+1).segment(1,N) = u_next;

        // prepare the next step
        u_current = u_next;
    }

    /// End of my solution ///

    return std::make_pair(u, time);
}
