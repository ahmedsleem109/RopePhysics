#include "rod.h"

#include <cassert>

namespace crs {

namespace {

// Body-frame inertia diagonal of a solid cylinder of length l about its centre,
// with the symmetry axis along e3.
Vec3 cylinderInertia(Real mass, Real radius, Real length) {
    const Real transverse = mass * (Real(3) * radius * radius + length * length) / Real(12);
    const Real axial = Real(0.5) * mass * radius * radius;
    return {transverse, transverse, axial};
}

Vec3 safeInverse(Vec3 d) {
    return {d.x > Real(0) ? Real(1) / d.x : Real(0), d.y > Real(0) ? Real(1) / d.y : Real(0),
            d.z > Real(0) ? Real(1) / d.z : Real(0)};
}

}  // namespace

Vec3 darboux(Quat qa, Quat qb, Real restLen) {
    Quat rel = conj(qa) * qb;
    // Stay on the hemisphere of the identity: the other representative encodes
    // the same rotation but flips the sign of the curvature we measure.
    rel = sameHemisphere(rel, Quat());
    return rel.im() * (Real(2) / restLen);
}

// ---------------------------------------------------------------- boundary

void Rod::pinParticle(int i) {
    state.invMass[i] = Real(0);
    state.v[i] = Vec3();
}

void Rod::clampSegment(int j) {
    state.invInertia[j] = Vec3();
    state.omega[j] = Vec3();
}

void Rod::freeParticle(int i, Real m) {
    state.mass[i] = m;
    state.invMass[i] = Real(1) / m;
}

// ---------------------------------------------------------------- diagnostics

Real Rod::kineticEnergy() const {
    Real e = 0;
    for (std::size_t i = 0; i < state.numParticles(); ++i)
        e += Real(0.5) * state.mass[i] * norm2(state.v[i]);
    for (std::size_t j = 0; j < state.numSegments(); ++j) {
        const Vec3 w = state.omega[j];
        e += Real(0.5) * dot(w, cwise(state.inertia[j], w));
    }
    return e;
}

Real Rod::gravitationalEnergy(Vec3 gravity) const {
    Real e = 0;
    for (std::size_t i = 0; i < state.numParticles(); ++i)
        e -= state.mass[i] * dot(gravity, state.x[i]);
    return e;
}

Real Rod::elasticEnergy() const {
    // E = 1/2 C^T alpha^-1 C, summed over both constraint families. The element
    // length already lives inside `compliance`, so this needs no extra factors.
    Real e = 0;
    for (std::size_t k = 0; k < stretch.size(); ++k) {
        const Vec3 d = state.x[stretch.p1[k]] - state.x[stretch.p0[k]];
        const Vec3 C = rotateInv(state.q[stretch.seg[k]], d) / stretch.restLength[k] - Vec3(0, 0, 1);
        const Vec3 a = stretch.compliance[k];
        e += Real(0.5) * (C.x * C.x / a.x + C.y * C.y / a.y + C.z * C.z / a.z);
    }
    for (std::size_t k = 0; k < bend.size(); ++k) {
        const Vec3 C = darboux(state.q[bend.segA[k]], state.q[bend.segB[k]], bend.restLength[k]) -
                       bend.restDarboux[k];
        const Vec3 a = bend.compliance[k];
        e += Real(0.5) * (C.x * C.x / a.x + C.y * C.y / a.y + C.z * C.z / a.z);
    }
    return e;
}

Vec3 Rod::linearMomentum() const {
    Vec3 p;
    for (std::size_t i = 0; i < state.numParticles(); ++i) p += state.v[i] * state.mass[i];
    return p;
}

Vec3 Rod::angularMomentum() const {
    Vec3 L;
    for (std::size_t i = 0; i < state.numParticles(); ++i)
        L += cross(state.x[i], state.v[i] * state.mass[i]);
    // Segment spin: I_body * omega_body rotated into world.
    for (std::size_t j = 0; j < state.numSegments(); ++j)
        L += rotate(state.q[j], cwise(state.inertia[j], state.omega[j]));
    return L;
}

Real Rod::totalMass() const {
    Real m = 0;
    for (Real mi : state.mass) m += mi;
    return m;
}

Real Rod::restLength() const {
    Real l = 0;
    for (Real li : stretch.restLength) l += li;
    return l;
}

// ---------------------------------------------------------------- build

void setRestFromCurrent(Rod& rod) {
    const RodState& s = rod.state;
    for (std::size_t k = 0; k < rod.stretch.size(); ++k)
        rod.stretch.restLength[k] = norm(s.x[rod.stretch.p1[k]] - s.x[rod.stretch.p0[k]]);
    const int nSeg = static_cast<int>(rod.stretch.size());
    for (std::size_t k = 0; k < rod.bend.size(); ++k) {
        const int a = rod.bend.segA[k], b = rod.bend.segB[k];
        // Interior joints span midpoint to midpoint. Boundary elements involve a
        // ghost frame with no segment of its own, so they keep the length they
        // were given and only have their rest curvature refreshed.
        if (a < nSeg && b < nSeg)
            rod.bend.restLength[k] =
                Real(0.5) * (rod.stretch.restLength[a] + rod.stretch.restLength[b]);
        rod.bend.restDarboux[k] = darboux(s.q[a], s.q[b], rod.bend.restLength[k]);
    }
}

Vec3 stretchCompliance(const RodMaterial& m, Real l) {
    const Real A = m.area();
    const Real EA = m.youngs * A;
    const Real GA = m.shearCorrection() * m.shear() * A;
    return {Real(1) / (GA * l), Real(1) / (GA * l), Real(1) / (EA * l)};
}

Vec3 bendCompliance(const RodMaterial& m, Real l) {
    const Real EI = m.bendStiffness();
    const Real GJ = m.twistStiffness();
    return {Real(1) / (EI * l), Real(1) / (EI * l), Real(1) / (GJ * l)};
}

int addFixedFrame(Rod& rod, Quat orientation) {
    RodState& s = rod.state;
    const Quat q = normalize(orientation);
    s.q.push_back(q);
    s.qPrev.push_back(q);
    s.omega.push_back(Vec3());
    s.extTorque.push_back(Vec3());
    s.inertia.push_back(Vec3());     // zero inertia: contributes no kinetic energy
    s.invInertia.push_back(Vec3());  // zero inverse inertia: never moves
    return static_cast<int>(s.q.size()) - 1;
}

void addBendConstraint(Rod& rod, int segA, int segB, Real elementLength, Vec3 restDarboux) {
    rod.bend.segA.push_back(segA);
    rod.bend.segB.push_back(segB);
    rod.bend.restLength.push_back(elementLength);
    rod.bend.restDarboux.push_back(restDarboux);
    rod.bend.compliance.push_back(bendCompliance(rod.material, elementLength));
    rod.bend.lambda.push_back(Vec3());
}

int clampEndExact(Rod& rod, bool tip) {
    const int lastSeg = static_cast<int>(rod.stretch.size()) - 1;
    const int seg = tip ? lastSeg : 0;
    const int particle = tip ? static_cast<int>(rod.state.numParticles()) - 1 : 0;
    const Real half = Real(0.5) * rod.stretch.restLength[seg];

    rod.pinParticle(particle);
    const int ghost = addFixedFrame(rod, rod.state.q[seg]);
    // Keep the constraint oriented along the rod, so the sign of the Darboux
    // vector means the same thing at both ends.
    if (tip)
        addBendConstraint(rod, seg, ghost, half, Vec3());
    else
        addBendConstraint(rod, ghost, seg, half, Vec3());
    return ghost;
}

void clampRootExact(Rod& rod) { clampEndExact(rod, false); }

namespace {

// Fill compliance from rest lengths and material.
void assignCompliance(Rod& rod) {
    for (std::size_t k = 0; k < rod.stretch.size(); ++k)
        rod.stretch.compliance[k] = stretchCompliance(rod.material, rod.stretch.restLength[k]);
    for (std::size_t k = 0; k < rod.bend.size(); ++k)
        rod.bend.compliance[k] = bendCompliance(rod.material, rod.bend.restLength[k]);
}

// Shared tail of both builders: given centerline points and segment frames,
// build masses, inertias, constraint topology, rest state and compliance.
Rod finishBuild(const std::vector<Vec3>& pts, const std::vector<Quat>& frames,
                const RodMaterial& mat) {
    const int nP = static_cast<int>(pts.size());
    const int nS = nP - 1;
    assert(nS >= 1 && static_cast<int>(frames.size()) == nS);

    Rod rod;
    rod.material = mat;
    RodState& s = rod.state;

    s.x = pts;
    s.xPrev = pts;
    s.v.assign(nP, Vec3());
    s.extForce.assign(nP, Vec3());
    s.mass.assign(nP, Real(0));
    s.invMass.assign(nP, Real(0));

    s.q = frames;
    s.qPrev = frames;
    s.omega.assign(nS, Vec3());
    s.extTorque.assign(nS, Vec3());
    s.inertia.assign(nS, Vec3());
    s.invInertia.assign(nS, Vec3());

    const Real rho = mat.density, A = mat.area();
    for (int j = 0; j < nS; ++j) {
        const Real l = norm(pts[j + 1] - pts[j]);
        const Real segMass = rho * A * l;
        // Lump half of each segment's mass onto each of its end particles.
        s.mass[j] += Real(0.5) * segMass;
        s.mass[j + 1] += Real(0.5) * segMass;
        s.inertia[j] = cylinderInertia(segMass, mat.radius, l);
        s.invInertia[j] = safeInverse(s.inertia[j]);
    }
    for (int i = 0; i < nP; ++i) s.invMass[i] = Real(1) / s.mass[i];

    rod.stretch.p0.resize(nS);
    rod.stretch.p1.resize(nS);
    rod.stretch.seg.resize(nS);
    rod.stretch.restLength.assign(nS, Real(0));
    rod.stretch.compliance.assign(nS, Vec3());
    rod.stretch.lambda.assign(nS, Vec3());
    for (int j = 0; j < nS; ++j) {
        rod.stretch.p0[j] = j;
        rod.stretch.p1[j] = j + 1;
        rod.stretch.seg[j] = j;
    }

    const int nB = nS - 1;
    rod.bend.segA.resize(nB);
    rod.bend.segB.resize(nB);
    rod.bend.restLength.assign(nB, Real(0));
    rod.bend.restDarboux.assign(nB, Vec3());
    rod.bend.compliance.assign(nB, Vec3());
    rod.bend.lambda.assign(nB, Vec3());
    for (int j = 0; j < nB; ++j) {
        rod.bend.segA[j] = j;
        rod.bend.segB[j] = j + 1;
    }

    setRestFromCurrent(rod);
    assignCompliance(rod);
    return rod;
}

}  // namespace

Rod makeStraightRod(int numSegments, Real length, const RodMaterial& material, Vec3 origin,
                    Vec3 direction) {
    const Vec3 dir = normalize(direction);
    const Real l = length / numSegments;

    std::vector<Vec3> pts(numSegments + 1);
    for (int i = 0; i <= numSegments; ++i) pts[i] = origin + dir * (l * i);

    // Every segment shares the frame that takes e3 onto the tangent: for a
    // straight rod this is already the parallel-transported frame.
    const Quat frame = quatFromTo(Vec3(0, 0, 1), dir);
    std::vector<Quat> frames(numSegments, frame);

    return finishBuild(pts, frames, material);
}

Rod makeRodFromCenterline(const std::vector<Vec3>& points, const RodMaterial& material,
                          Vec3 frameSeed, Real twistPerLength) {
    const int nS = static_cast<int>(points.size()) - 1;
    assert(nS >= 1);

    std::vector<Vec3> tangents(nS);
    for (int j = 0; j < nS; ++j) tangents[j] = normalize(points[j + 1] - points[j]);

    // Seed frame: orthogonalize the requested d1 against the first tangent.
    Vec3 d1 = frameSeed - tangents[0] * dot(frameSeed, tangents[0]);
    if (norm2(d1) < Real(1e-20)) {
        d1 = Vec3(1, 0, 0) - tangents[0] * dot(Vec3(1, 0, 0), tangents[0]);
        if (norm2(d1) < Real(1e-20)) d1 = Vec3(0, 1, 0) - tangents[0] * dot(Vec3(0, 1, 0), tangents[0]);
    }
    d1 = normalize(d1);

    std::vector<Quat> frames(nS);
    frames[0] = quatFromMat3(Mat3::fromCols(d1, cross(tangents[0], d1), tangents[0]));
    // Discrete parallel transport: rotate the previous frame by the smallest
    // rotation carrying the previous tangent onto the next one.
    for (int j = 1; j < nS; ++j)
        frames[j] = normalize(quatFromTo(tangents[j - 1], tangents[j]) * frames[j - 1]);

    if (twistPerLength != Real(0)) {
        Real s = 0;
        for (int j = 0; j < nS; ++j) {
            const Real l = norm(points[j + 1] - points[j]);
            const Real sMid = s + Real(0.5) * l;
            // Twist is about the material tangent, i.e. a body-frame rotation.
            frames[j] = normalize(frames[j] * quatAxisAngle(Vec3(0, 0, 1), twistPerLength * sMid));
            s += l;
        }
    }

    return finishBuild(points, frames, material);
}

}  // namespace crs
