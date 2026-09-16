// Direct static equilibrium solver for the discrete Cosserat rod.
//
// The XPBD relaxation in solver.h reaches static equilibrium by damped
// dynamics, and Gauss-Seidel moves information one element per sweep, so the
// budget a converged relaxation needs grows quadratically with segment count.
// That is the right tool for checking that the SOLVER finds equilibrium, and
// the wrong one for checking that the DISCRETIZATION converges to the
// continuum: those cases only need the equilibrium itself.
//
// This solves it directly. The unknowns are the free particle positions and a
// body-frame rotation vector per free segment, the same perturbation the XPBD
// Jacobians use. Equilibrium is
//
//     r = J^T alpha^-1 C  -  f_ext  =  0
//
// with f_ext the particle forces (plus gravity) and the applied torques
// expressed in each segment's body frame. The torques are dead loads fixed in
// the world, so r is not the gradient of any potential and its Jacobian is not
// symmetric; the linear solve is a banded LU with partial pivoting.
//
// Every constraint couples at most three neighbouring elements, so with DOFs
// interleaved particle/segment along the rod the Jacobian is banded with a
// half-bandwidth of 8 -- the block-tridiagonal structure of the rod. It is
// built by central differences of the exact residual, perturbing every DOF
// that is more than a bandwidth apart at once, so one Jacobian costs a fixed
// number of residual evaluations regardless of rod length. Each Newton step is
// O(n).
//
// Large deflections are reached by load continuation: the loads are ramped
// from zero, and a load step whose Newton iteration fails is retried at half
// the size.
#pragma once

#include "rod.h"

namespace crs {

struct StaticParams {
    Vec3 gravity = Vec3();
    // Newton stops when the largest update component falls below this (metres
    // for positions, radians for rotations).
    Real stepTol = Real(1e-12);
    int maxNewtonIterations = 30;
    // Largest rotation, in radians, any single Newton update may apply to a
    // segment. The update is scaled down uniformly to respect it.
    Real maxRotationPerIteration = Real(0.5);
    int maxLoadSteps = 4096;
};

struct StaticReport {
    bool converged = false;
    int loadSteps = 0;         // accepted continuation steps
    int newtonIterations = 0;  // total, over all accepted and rejected steps
    Real residual = 0;         // inf-norm of r at the returned state [N or N m]
};

// Solve for the static equilibrium under the rod's extForce, extTorque and
// `gravity`, starting from the current configuration. The rod must be
// restrained against rigid motion (pinned particles or fixed frames), or the
// Jacobian is singular. On return velocities are zero and prev = current.
StaticReport solveStatic(Rod& rod, const StaticParams& params = StaticParams());

// Residual of the static equilibrium, one Vec3 per particle and per segment
// (zero for pinned ones). Exposed for tests and diagnostics.
void staticResidual(const Rod& rod, Vec3 gravity, Real loadScale, std::vector<Vec3>& forces,
                    std::vector<Vec3>& torques);

}  // namespace crs
