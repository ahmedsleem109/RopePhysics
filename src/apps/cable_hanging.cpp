#include "cable_hanging.h"

#include <algorithm>
#include <cmath>

namespace crs {
namespace apps {

RodMaterial HangingCable::material() const {
    RodMaterial m;
    m.radius = Real(0.004);
    m.density = Real(1100);
    m.poisson = Real(0.35);
    m.youngs = Real(1e6);
    return m;
}

Rod HangingCable::build(Real legRatio, Real youngs, Vec3 barCentre) const {
    RodMaterial mat = material();
    mat.youngs = youngs;

    const Real wrapRadius = barRadius + mat.radius;
    const Real arc = kPi * wrapRadius;
    const Real legs = length - arc;
    const Real longLeg = legs * legRatio / (Real(1) + legRatio);
    const Real shortLeg = legs - longLeg;
    const Real top = barCentre.z + barHeight;

    std::vector<Vec3> pts;
    const Real h = length / segments;
    for (int i = 0; i <= segments; ++i) {
        const Real s = h * i;
        Vec3 p;
        if (s < shortLeg) {
            p = Vec3(-wrapRadius, 0, top - (shortLeg - s));
        } else if (s < shortLeg + arc) {
            const Real phi = (s - shortLeg) / wrapRadius;
            p = Vec3(-wrapRadius * std::cos(phi), 0, top + wrapRadius * std::sin(phi));
        } else {
            p = Vec3(wrapRadius, 0, top - (s - shortLeg - arc));
        }
        pts.push_back(Vec3(barCentre.x, barCentre.y, 0) + p);
    }
    Rod rod = makeRodFromCenterline(pts, mat, Vec3(0, 1, 0));
    for (Vec3& rest : rod.bend.restDarboux) rest = Vec3();
    return rod;
}

CollisionWorld HangingCable::world(Real friction, Vec3 barCentre) const {
    CollisionWorld w;
    const Vec3 axis(barCentre.x, barCentre.y, barCentre.z + barHeight);
    w.primitives.push_back(Primitive::makeCapsule(axis - Vec3(0, barHalfLength, 0),
                                                  axis + Vec3(0, barHalfLength, 0), barRadius,
                                                  friction));
    w.primitives.push_back(Primitive::makePlane(Vec3(0, 0, barCentre.z), Vec3(0, 0, 1), friction));
    return w;
}

SolverParams HangingCable::params() const {
    SolverParams p;
    // 8 substeps x 2 sweeps, not 16 x 1. On the GPU's single precision a
    // cable at rest near 0.8 m can only move in steps of ~6e-8 m, and gravity
    // moves it g h^2 per substep: 3.8e-8 m at 16 substeps, which rounds away.
    // Velocity is recovered from positions, so the cable could never start to
    // slide -- float rounding acting as static friction. It made cables that
    // slip in double precision (and in theory) hold on the GPU. At 8 substeps
    // g h^2 is 2.5x the float spacing and both precisions agree.
    p.dt = Real(1e-3);
    p.substeps = 8;
    p.iterations = 2;
    p.gravity = Vec3(0, 0, Real(-9.81));
    return p;
}

double HangingCable::floatMotionMargin() const {
    const SolverParams p = params();
    const double h = double(p.dt) / p.substeps;
    const double highest = double(barHeight + barRadius) + 0.05;
    const double spacing = double(std::nextafter(float(highest), 2.0f) - float(highest));
    return std::abs(double(p.gravity.z)) * h * h / spacing;
}

int HangingCable::steps() const { return int(std::lround(duration / params().dt)); }

Real HangingCable::capstanFriction(Real legRatio) { return std::log(legRatio) / kPi; }

}  // namespace apps
}  // namespace crs
