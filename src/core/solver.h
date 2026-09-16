// XPBD solver for the discrete Cosserat rod.
//
// Substepped XPBD (Macklin et al. 2019): one Gauss-Seidel sweep per substep by
// default, because substepping buys accuracy far more cheaply than iterating.
// Compliance is scaled as alpha/h^2 with h the SUBSTEP size, which is what makes
// the effective stiffness independent of both timestep and segment count.
#pragma once

#include "collision.h"
#include "coloring.h"
#include "rod.h"

namespace crs {

struct SolverParams {
    Real dt = Real(1e-3);
    int substeps = 10;
    int iterations = 1;  // constraint sweeps per substep
    Vec3 gravity = Vec3(0, 0, Real(-9.81));

    // Exponential velocity decay, expressed as a rate [1/s] so the amount of
    // damping per unit time does not change when dt or substeps change.
    Real linearDamping = Real(0);
    Real angularDamping = Real(0);

    // Regenerate contacts every N substeps. 1 is the accurate choice and what
    // the validation cases use; larger values trade contact freshness for the
    // broadphase cost, which is the knob that matters once this is on a GPU.
    int contactInterval = 1;

    // Optional constraint orderings. When set, each Gauss-Seidel sweep visits
    // constraints colour by colour instead of in index order. The result is a
    // different (red-black) iteration, not a faster one -- on the CPU this
    // exists so the reference can reproduce exactly what the GPU does.
    const Coloring* stretchColoring = nullptr;
    const Coloring* bendColoring = nullptr;
};

// Per-rod scratch that persists across steps so the contact arrays and hash
// tables are not reallocated every frame.
struct SolverContext {
    ContactSet contacts;
    SpatialHash hash;
};

// Choose a substep count so that the unconstrained predictor never displaces a
// particle by more than `fraction` of the shortest element in one substep, nor
// spins a frame by more than `maxAngle` radians.
//
// What this is, precisely: a conservative guard used by the static relaxations
// in the validation suite, where loads are applied suddenly to a rod at rest.
// Its angular half is load-bearing for ACCURACY and was measured to be: at
// maxAngle = 0.1 the pure-moment case converges to a state carrying 0.77% excess
// moment in every joint, and at 0.00625 that bias disappears.
//
// What this is not: the dynamic accuracy limit. That is measured separately
// (the `timestep-envelope` case): keeping strain error under 1% requires
// material to move no more than a few percent of an element length per substep,
// which is a kinematic bound, not the h^2 f / m group this function bounds.
int stableSubsteps(const Rod& rod, Vec3 gravity, Real dt, Real fraction = Real(0.05),
                   Real maxAngle = Real(0.1));

// One full step of `params.dt`, internally substepped.
void step(Rod& rod, const SolverParams& params);

// Same, with collision against `world` and rod self-collision.
void step(Rod& rod, const SolverParams& params, const CollisionWorld& world, SolverContext& ctx);

// Single substep of size h. Exposed so tests can drive it directly.
void substep(Rod& rod, const SolverParams& params, Real h, ContactSet* contacts = nullptr);

// Constraint projection only (one Gauss-Seidel sweep over both families).
void projectConstraints(Rod& rod, Real h, const Coloring* stretchColoring = nullptr,
                        const Coloring* bendColoring = nullptr);

// One sweep over the contact set: unilateral non-penetration followed by
// position-level Coulomb friction, both against the multipliers accumulated so
// far in this substep.
void projectContacts(Rod& rod, ContactSet& contacts);

// Residual of the equilibrium search: the largest particle speed and segment
// angular speed currently in the state.
struct Residual {
    Real maxSpeed = 0;
    Real maxAngularSpeed = 0;
};
Residual residual(const Rod& rod);

// Drive the rod to static equilibrium by stepping with heavy damping until the
// residual falls below tolerance. Returns the number of steps taken; a return
// of `maxSteps` means it did not converge and the caller should say so.
struct RelaxReport {
    int steps = 0;
    bool converged = false;
    Residual final;
};
RelaxReport relaxToEquilibrium(Rod& rod, SolverParams params, int maxSteps,
                               Real speedTol = Real(1e-7));
RelaxReport relaxToEquilibrium(Rod& rod, SolverParams params, const CollisionWorld& world,
                               SolverContext& ctx, int maxSteps, Real speedTol = Real(1e-7));

}  // namespace crs
