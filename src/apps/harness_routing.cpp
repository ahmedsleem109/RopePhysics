#include "harness_routing.h"

#include <algorithm>
#include <cmath>

namespace crs {
namespace apps {

RodMaterial HarnessTask::material() const {
    RodMaterial m;
    m.radius = Real(0.004);
    m.density = Real(1100);
    m.poisson = Real(0.35);
    m.youngs = cableYoungs;
    return m;
}

Rod HarnessTask::build(Real youngsScale) const {
    RodMaterial mat = material();
    mat.youngs *= youngsScale;
    const Vec3 start = connector + Vec3(0, 0, mat.radius);
    Rod rod = makeStraightRod(segments, cableLength, mat, start, Vec3(0, -1, 0));
    rod.pinParticle(0);
    rod.pinParticle(segments);
    return rod;
}

CollisionWorld HarnessTask::world(Real frictionScale) const {
    const Real mu = boardFriction * frictionScale;
    CollisionWorld w;
    w.primitives.push_back(Primitive::makePlane(Vec3(), Vec3(0, 0, 1), mu));
    for (const Vec3& peg : {pegA, pegB})
        w.primitives.push_back(
            Primitive::makeCapsule(peg, peg + Vec3(0, 0, pegHeight), pegRadius, mu));
    for (Real side : {Real(-1), Real(1)}) {
        const Vec3 post = clip + Vec3(0, side * clipHalfGap, 0);
        w.primitives.push_back(
            Primitive::makeCapsule(post, post + Vec3(0, 0, clipHeight), clipPostRadius, mu));
    }
    return w;
}

SolverParams HarnessTask::params() const {
    SolverParams p;
    // 4 substeps x 4 sweeps. Coordinates reach ~1.2 m, where a float can only
    // move in steps of ~1.2e-7 m; gravity moves a particle g h^2 = 6e-7 m per
    // substep here, a 5x margin. (At 8 substeps it would be 1.3x, too close:
    // see HangingCable::params for what happens below 1.)
    p.dt = Real(1e-3);
    p.substeps = 4;
    p.iterations = 4;
    p.gravity = Vec3(0, 0, Real(-9.81));
    return p;
}

int HarnessTask::steps() const {
    Real total = holdSeconds;
    for (Real s : legSeconds) total += s;
    return int(std::lround(total / params().dt));
}

HarnessTask::Policy HarnessTask::handWrittenPolicy() const {
    // Straight at the targets: under peg A, over peg B, through the clip.
    return {Vec3(pegA.x, pegA.y - Real(0.08), 0), Vec3(Real(0.40), 0, 0),
            Vec3(pegB.x, pegB.y + Real(0.08), 0),
            Vec3(clip.x - Real(0.10), 0, 0), Vec3(clip.x + Real(0.18), 0, 0)};
}

Vec3 HarnessTask::gripperAt(const Policy& policy, Real t) const {
    Vec3 from(connector.x, connector.y - cableLength, 0);
    Real t0 = 0;
    for (int k = 0; k < kWaypoints; ++k) {
        const Real t1 = t0 + legSeconds[k];
        if (t < t1) {
            // Smoothstep within each leg, so the gripper does not jerk.
            Real u = (t - t0) / legSeconds[k];
            u = u * u * (Real(3) - Real(2) * u);
            const Vec3 p = from + (policy[k] - from) * u;
            return Vec3(p.x, p.y, material().radius);
        }
        from = policy[k];
        t0 = t1;
    }
    return Vec3(from.x, from.y, material().radius);
}

Vec3 HarnessTask::gripperVelocity(const Policy& policy, int step) const {
    const double period = double(params().dt) * stepsPerControl;
    const int tick = step / stepsPerControl;
    return (gripperAt(policy, Real((tick + 1) * period)) - gripperAt(policy, Real(tick * period))) /
           Real(period);
}

HarnessTask::Outcome HarnessTask::evaluate(const std::vector<Vec3>& cable) const {
    Outcome out;
    const Real r = material().radius;

    // Which side of a peg the cable passes: where it crosses the line x = peg.x,
    // taking the crossing closest to the peg, if one comes within reach. (The
    // cable point nearest the peg is not enough: after wrapping under a peg the
    // nearest point can be on the stretch already climbing towards the next.)
    // Returns the signed offset of that crossing, 0 if the cable never crosses.
    auto sideOf = [&](const Vec3& peg) {
        Real best = 1e9;
        Real side = 0;
        for (std::size_t i = 0; i + 1 < cable.size(); ++i) {
            const Vec3 p = cable[i], q = cable[i + 1];
            if ((p.x - peg.x) * (q.x - peg.x) > 0) continue;
            const Real u = std::abs(q.x - p.x) > 1e-9 ? (peg.x - p.x) / (q.x - p.x) : Real(0);
            const Real y = p.y + (q.y - p.y) * u;
            if (std::abs(y - peg.y) < best) {
                best = std::abs(y - peg.y);
                side = y - peg.y;
            }
        }
        return side;
    };
    // Laid against the peg: a route that passes the right side of a peg with a
    // slack loop is not a routed harness.
    const Real touching = pegRadius + r;
    const Real slack = Real(0.015);
    const Real a = sideOf(pegA);
    const Real b = sideOf(pegB);
    out.belowA = a < 0 && -a <= touching + slack;
    out.aboveB = b > 0 && b <= touching + slack;

    // Through the clip: some segment crosses the clip's x between its posts,
    // down on the board.
    Real clipMiss = 1e9;
    for (std::size_t i = 0; i + 1 < cable.size(); ++i) {
        const Vec3 p = cable[i], q = cable[i + 1];
        if ((p.x - clip.x) * (q.x - clip.x) > 0) continue;
        const Real u = std::abs(q.x - p.x) > 1e-9 ? (clip.x - p.x) / (q.x - p.x) : Real(0);
        const Vec3 c = p + (q - p) * u;
        if (c.z > clipHeight) continue;
        clipMiss = std::min(clipMiss, std::abs(c.y - clip.y));
    }
    // Between the post centres. (Resting against a post from inside it sits at
    // clipHalfGap - clipPostRadius - r, less when pulled hard into the post;
    // from outside it could not be closer than clipHalfGap + clipPostRadius.)
    out.throughClip = clipMiss < clipHalfGap;
    out.sideA = a;
    out.sideB = b;
    out.clipMiss = clipMiss;

    auto clamp01 = [](double v) { return std::max(0.0, std::min(1.0, v)); };
    auto pegScore = [&](double offset) {  // offset > 0: on the correct side
        if (offset <= 0) return 0.0;
        return 1.0 - 0.5 * clamp01((offset - double(touching + slack)) / 0.1);
    };
    out.score = pegScore(-a) + pegScore(b) +
                (out.throughClip ? 1.0 : clamp01(1.0 - (clipMiss - clipHalfGap) / 0.1));
    return out;
}

}  // namespace apps
}  // namespace crs
