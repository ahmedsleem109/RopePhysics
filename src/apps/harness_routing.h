// Application: a robot routes a wire harness.
//
// Car and aircraft wiring looms are still largely routed by hand: a worker
// pulls each cable along a board, around pegs and into clips. Automating it is
// hard because the cable is the part you cannot predict -- how stiff it is,
// how it drags on the board -- and a motion that works for one cable fails for
// the next. This is the kind of skill that is learned in simulation, across
// thousands of randomized cables, before a robot attempts it.
//
// The board: a connector on the left holds one end of the cable. The gripper
// holds the other end and must lay the cable BELOW peg A, ABOVE peg B, and
// THROUGH a clip on the right. The robot's decision is its motion: a few
// waypoints the gripper passes through on a fixed schedule. What it cannot
// control is the cable's stiffness and the board's friction.
//
// This header is the single definition of the task, shared by the learning
// loop on the GPU, its CPU cross-check and the demo scenes.
#pragma once

#include <array>
#include <vector>

#include "../core/collision.h"
#include "../core/solver.h"

namespace crs {
namespace apps {

struct HarnessTask {
    // Board layout, metres; the board surface is z = 0.
    Vec3 connector = Vec3(0, 0, 0);
    Vec3 pegA = Vec3(Real(0.25), 0, 0);
    Vec3 pegB = Vec3(Real(0.55), 0, 0);
    Vec3 clip = Vec3(Real(0.85), 0, 0);
    Real pegRadius = Real(0.012);
    Real pegHeight = Real(0.06);
    Real clipPostRadius = Real(0.006);
    Real clipHalfGap = Real(0.018);  // post centre to clip centreline
    Real clipHeight = Real(0.04);
    Real boardFriction = Real(0.5);

    // Cable: 8 mm across, 1.2 m, 1 cm elements (shorter than its diameter).
    Real cableLength = Real(1.2);
    int segments = 120;
    Real cableYoungs = Real(5e6);  // the reference; batches scale it per rod

    // A policy is where the gripper goes: waypoints in the board plane, visited
    // on the fixed schedule below. Starting from the cable's free end, the
    // gripper passes through waypoint[0..3] and ends at the last one.
    static constexpr int kWaypoints = 5;
    using Policy = std::array<Vec3, kWaypoints>;
    // Seconds spent reaching each waypoint, then holding still at the end.
    std::array<Real, kWaypoints> legSeconds = {Real(2.4), Real(0.7), Real(0.7), Real(0.8),
                                               Real(0.7)};
    Real holdSeconds = Real(1.0);

    RodMaterial material() const;
    // The cable lying loose off the board edge: from the connector straight
    // towards -y. Particle 0 is plugged in (pinned); the last particle is held
    // by the gripper (pinned, driven kinematically).
    Rod build(Real youngsScale = Real(1)) const;
    CollisionWorld world(Real frictionScale = Real(1)) const;
    SolverParams params() const;
    int steps() const;

    // A reasonable first guess a person would program.
    Policy handWrittenPolicy() const;

    // Gripper position at time t, and the velocity to apply over [t, t + dt).
    Vec3 gripperAt(const Policy& policy, Real t) const;
    Vec3 gripperVelocity(const Policy& policy, Real t) const;
    // The times at which the gripper velocity changes (start of each leg and
    // of the hold): the only moments a batch must upload new velocities.
    std::vector<Real> legStarts() const;

    // How well a cable shape fulfils the routing.
    struct Outcome {
        bool belowA = false, aboveB = false, throughClip = false;
        // Smooth score for learning: each requirement contributes up to 1 by
        // how far the cable sits on the correct side (capped), so a near miss
        // scores better than a wild one.
        double score = 0;
        double sideA = 0, sideB = 0, clipMiss = 0;  // diagnostics, metres
        bool success() const { return belowA && aboveB && throughClip; }
    };
    Outcome evaluate(const std::vector<Vec3>& cable) const;
};

}  // namespace apps
}  // namespace crs
