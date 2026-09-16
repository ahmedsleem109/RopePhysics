// Application: will a cable stay on a hook?
//
// A robot (or a person) drapes a cable over a horizontal bar and lets go. The
// decision it controls is WHERE it grasps the cable, which sets how much hangs
// on each side; the property it cannot control is friction. Whether the cable
// stays is the answer the robot needs before it acts.
//
// For a perfectly flexible rope this has a classical answer. The two legs
// pull with tensions proportional to their lengths, and the capstan equation
// says the rope holds while
//
//     long / short <= exp(mu * pi)          (half a turn of wrap)
//
// so the critical friction for a placement is mu* = ln(long / short) / pi. A
// real cable has bending stiffness, which the formula ignores; how far a stiff
// cable departs from it is exactly what a simulator is for.
//
// This header is the single definition of the task, used by the GPU sweep, its
// CPU cross-check, and the demo scene, so they cannot drift apart.
#pragma once

#include "../core/collision.h"
#include "../core/solver.h"

namespace crs {
namespace apps {

struct HangingCable {
    Real length = Real(1.0);      // cable length [m]
    int segments = 100;           // 1 cm elements: shorter than the cable's diameter
    Real barRadius = Real(0.05);  // [m]
    Real barHeight = Real(0.8);   // bar axis height above the floor [m]
    Real barHalfLength = Real(0.3);
    // Simulated seconds after release. Long, because this solver's friction
    // creeps: even at twice the capstan friction a draped cable creeps about
    // 2 mm/s, and just above the threshold that creep turns into a slide after
    // several seconds (mu = 0.28 on a 2:1 drape held at 8 s and fell by 12 s).
    // A 3 s window matched theory to 0.006 only because it stopped watching.
    Real duration = Real(8.0);

    // A soft rubber-jacketed cable, 8 mm across. Youngs modulus is what the
    // sweep varies.
    RodMaterial material() const;

    // The cable released at rest, draped over the bar centred at `barCentre`:
    // the short leg on -x, half a turn over the top, the long leg on +x, with
    // long/short = legRatio. Its rest shape is straight -- it is a cable, not a
    // pre-bent hook -- so the drape is stressed from the start.
    Rod build(Real legRatio, Real youngs, Vec3 barCentre = Vec3()) const;

    // The bar and the floor, both with the given friction coefficient.
    CollisionWorld world(Real friction, Vec3 barCentre = Vec3()) const;

    SolverParams params() const;

    // How far gravity moves a particle in one substep, in units of the float
    // spacing at the highest coordinate in the scene. Below ~1 a resting cable
    // cannot start moving in single precision (see params()).
    double floatMotionMargin() const;
    int steps() const;

    // Ideal flexible-rope answer: the smallest friction that holds legRatio.
    static Real capstanFriction(Real legRatio);
};

// Did the cable stay? It slid off if its short end ever rose by more than
// kSlipTolerance: a cable that holds settles by millimetres, one that slips
// drags its short end up and over the bar.
struct HangOutcome {
    static constexpr Real kSlipTolerance = Real(0.05);

    Real startZ = 0;
    Real maxRise = 0;

    explicit HangOutcome(Real shortEndZ) : startZ(shortEndZ) {}
    void observe(Real shortEndZ) { maxRise = std::max(maxRise, shortEndZ - startZ); }
    bool held() const { return maxRise <= kSlipTolerance; }
};

}  // namespace apps
}  // namespace crs
