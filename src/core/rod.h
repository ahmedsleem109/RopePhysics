// Discrete Cosserat rod: state, material, and constraint topology.
//
// Discretization (Kugelstadt & Schoemer 2016, "Position and Orientation Based
// Cosserat Rods", in the XPBD setting of Macklin et al. 2016/2019):
//
//   * N particles carry the centerline:            x_i, mass m_i
//   * N-1 segments carry the material frame:       q_j, body-frame inertia
//
//   Segment j spans particles j and j+1. Its body z-axis e3 is the tangent, so
//   d3 = R(q_j) e3 is the material direction and d1, d2 span the cross-section.
//
// Layout is structure-of-arrays throughout, because the CUDA port reads these
// same arrays and wants coalesced access.
#pragma once

#include <cstddef>
#include <vector>

#include "math3.h"

namespace crs {

// ---------------------------------------------------------------- material

// Isotropic linearly-elastic circular cross-section.
struct RodMaterial {
    Real youngs = Real(1e7);    // E   [Pa]
    Real poisson = Real(0.35);  // nu  [-]
    Real density = Real(1000);  // rho [kg/m^3]
    Real radius = Real(5e-3);   // r   [m]

    Real shear() const { return youngs / (Real(2) * (Real(1) + poisson)); }  // G
    Real area() const { return kPi * radius * radius; }               // A
    Real secondMoment() const {                                              // I
        const Real r2 = radius * radius;
        return kPi * r2 * r2 / Real(4);
    }
    Real polarMoment() const { return Real(2) * secondMoment(); }  // J = 2I
    // Timoshenko shear correction factor for a circular section.
    Real shearCorrection() const {
        return Real(6) * (Real(1) + poisson) / (Real(7) + Real(6) * poisson);
    }
    Real bendStiffness() const { return youngs * secondMoment(); }  // EI
    Real twistStiffness() const { return shear() * polarMoment(); } // GJ
};

// ---------------------------------------------------------------- state

struct RodState {
    // Particles.
    std::vector<Vec3> x, xPrev, v;
    std::vector<Real> invMass, mass;
    std::vector<Vec3> extForce;  // external force per particle [N], zero by default

    // Segments (orientation elements). omega is in the BODY frame, matching the
    // right-multiplied rotation increment the solver uses.
    std::vector<Quat> q, qPrev;
    std::vector<Vec3> omega;
    std::vector<Vec3> invInertia, inertia;  // body-frame diagonals
    std::vector<Vec3> extTorque;            // external torque per segment [N m], WORLD frame

    std::size_t numParticles() const { return x.size(); }
    std::size_t numSegments() const { return q.size(); }
};

// ---------------------------------------------------------------- constraints
//
// Stretch/shear, one per segment, coupling the two end particles to the frame:
//     C_s = (1/l) (x_{i+1} - x_i) - R(q_j) e3          [3 components]
// Bend/twist, one per interior joint, from the discrete Darboux vector:
//     C_b = (2/lbar) Im(conj(q_a) q_b) - Omega_0       [3 components]
//
// Compliance is the inverse of the energy Hessian per unit length times the
// element length, which is what makes stiffness independent of resolution:
//     alpha_s = diag(1/(ks G A), 1/(ks G A), 1/(E A)) / l
//     alpha_b = diag(1/(E I),    1/(E I),    1/(G J))  / lbar

struct StretchConstraints {
    std::vector<int> p0, p1, seg;
    std::vector<Real> restLength;
    std::vector<Vec3> compliance, lambda;

    std::size_t size() const { return seg.size(); }
    void clearMultipliers() { lambda.assign(size(), Vec3()); }
};

struct BendConstraints {
    std::vector<int> segA, segB;
    std::vector<Real> restLength;  // lbar: distance between segment midpoints
    std::vector<Vec3> restDarboux, compliance, lambda;

    std::size_t size() const { return segA.size(); }
    void clearMultipliers() { lambda.assign(size(), Vec3()); }
};

// ---------------------------------------------------------------- rod

struct Rod {
    RodState state;
    StretchConstraints stretch;
    BendConstraints bend;
    RodMaterial material;

    // Boundary conditions.
    void pinParticle(int i);          // fix position
    void clampSegment(int j);         // fix orientation
    void freeParticle(int i, Real m); // restore a finite mass

    // Diagnostics.
    Real kineticEnergy() const;
    Real gravitationalEnergy(Vec3 gravity) const;
    Real elasticEnergy() const;
    Vec3 linearMomentum() const;
    Vec3 angularMomentum() const;  // about the origin, particles + segment spin
    Real totalMass() const;
    Real restLength() const;  // sum of segment rest lengths
};

// Build a straight rod of `numSegments` elements from `origin` along unit
// `direction`. Particle masses are lumped half-and-half from the adjacent
// segments; the rest configuration is recorded as the zero of both constraints.
Rod makeStraightRod(int numSegments, Real length, const RodMaterial& material,
                    Vec3 origin = Vec3(0, 0, 0), Vec3 direction = Vec3(1, 0, 0));

// Build a rod along an arbitrary polyline. Material frames follow the discrete
// parallel transport of `frameSeed` (projected perpendicular to the first
// tangent), so the rest configuration is twist-free by construction; the
// per-joint rest twist can then be set explicitly with `twistPerLength`.
Rod makeRodFromCenterline(const std::vector<Vec3>& points, const RodMaterial& material,
                          Vec3 frameSeed = Vec3(0, 0, 1), Real twistPerLength = Real(0));

// Compliance of one element, given its length. Exposed because boundary
// elements are shorter than interior ones and need their own values.
Vec3 stretchCompliance(const RodMaterial& m, Real elementLength);
Vec3 bendCompliance(const RodMaterial& m, Real elementLength);

// Append a fixed orientation element: a kinematic boundary frame with no
// inertia. It never moves, carries no kinetic energy, and is referenced by
// constraints exactly like a real segment.
int addFixedFrame(Rod& rod, Quat orientation);

// Append a bend/twist constraint between two orientation elements.
void addBendConstraint(Rod& rod, int segA, int segB, Real elementLength, Vec3 restDarboux);

// Clamp one end of the rod: pin the end particle and tie the end segment to a
// fixed ghost frame sitting at the very end of the rod, through a HALF-length
// bend element. Returns the index of the ghost frame, so a caller can drive it
// (rotating it about the tangent imposes a twist; about a transverse axis
// imposes a bend).
//
// The obvious alternative -- freezing segment 0's own orientation -- puts the
// clamp at s = l/2, because that is where the first material frame lives. The
// effective cantilever length is then L - l/2 and every static deflection
// carries an O(h) error of about -1.5 h / L. The half element moves the
// boundary condition to where the continuum problem puts it.
int clampEndExact(Rod& rod, bool tip);

// Convenience for the common case: clamp s = 0.
void clampRootExact(Rod& rod);

// Recompute rest quantities (rest lengths, rest Darboux vectors) from the
// current configuration, i.e. declare "this pose is unstressed".
void setRestFromCurrent(Rod& rod);

// Discrete Darboux vector of a joint, matching the constraint definition.
Vec3 darboux(Quat qa, Quat qb, Real restLen);

}  // namespace crs
