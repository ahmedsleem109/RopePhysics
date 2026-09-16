#include "statics.h"

#include <algorithm>
#include <cmath>

namespace crs {

namespace {

const Vec3 kE3(0, 0, 1);

// Free DOFs, interleaved particle 0, segment 0, particle 1, segment 1, ... so
// that every constraint touches a contiguous window. -1 marks a restrained
// element. Fixed ghost frames are appended after the real segments and are
// always restrained, so they never widen the band.
struct DofMap {
    std::vector<int> particle, segment;
    int size = 0;
    int halfBand = 2;

    explicit DofMap(const Rod& rod) {
        const RodState& s = rod.state;
        const int nP = static_cast<int>(s.numParticles());
        const int nS = static_cast<int>(s.numSegments());
        particle.assign(nP, -1);
        segment.assign(nS, -1);
        for (int k = 0; k < std::max(nP, nS); ++k) {
            if (k < nP && s.invMass[k] > Real(0)) {
                particle[k] = size;
                size += 3;
            }
            if (k < nS && norm2(s.invInertia[k]) > Real(0)) {
                segment[k] = size;
                size += 3;
            }
        }
        auto widen = [&](std::initializer_list<int> bases) {
            int lo = size, hi = -1;
            for (int b : bases) {
                if (b < 0) continue;
                lo = std::min(lo, b);
                hi = std::max(hi, b + 2);
            }
            if (hi >= 0) halfBand = std::max(halfBand, hi - lo);
        };
        for (std::size_t k = 0; k < rod.stretch.size(); ++k)
            widen({particle[rod.stretch.p0[k]], particle[rod.stretch.p1[k]],
                   segment[rod.stretch.seg[k]]});
        for (std::size_t k = 0; k < rod.bend.size(); ++k)
            widen({segment[rod.bend.segA[k]], segment[rod.bend.segB[k]]});
    }
};

void assemble(const Rod& rod, const DofMap& dofs, Vec3 gravity, Real loadScale,
              std::vector<Vec3>& f, std::vector<Vec3>& t, std::vector<Real>& r) {
    staticResidual(rod, gravity, loadScale, f, t);
    r.assign(dofs.size, Real(0));
    for (std::size_t i = 0; i < f.size(); ++i)
        if (dofs.particle[i] >= 0)
            for (int c = 0; c < 3; ++c) r[dofs.particle[i] + c] = f[i][c];
    for (std::size_t j = 0; j < t.size(); ++j)
        if (dofs.segment[j] >= 0)
            for (int c = 0; c < 3; ++c) r[dofs.segment[j] + c] = t[j][c];
}

// Band matrix with room for the fill-in partial pivoting creates: row i stores
// columns [i - b, i + 2b].
struct BandMatrix {
    int n, b, width;
    std::vector<Real> a;

    BandMatrix(int n_, int b_) : n(n_), b(b_), width(3 * b_ + 1), a(std::size_t(n_) * width, 0) {}
    Real& at(int i, int j) { return a[std::size_t(i) * width + (j - i + b)]; }

    // Solve A x = rhs in place (rhs becomes x). Destroys A.
    bool solve(std::vector<Real>& rhs) {
        for (int k = 0; k < n; ++k) {
            const int rowEnd = std::min(n - 1, k + b);
            const int colEnd = std::min(n - 1, k + 2 * b);
            int p = k;
            for (int i = k + 1; i <= rowEnd; ++i)
                if (std::abs(at(i, k)) > std::abs(at(p, k))) p = i;
            if (at(p, k) == Real(0)) return false;
            if (p != k) {
                for (int j = k; j <= colEnd; ++j) std::swap(at(k, j), at(p, j));
                std::swap(rhs[k], rhs[p]);
            }
            const Real inv = Real(1) / at(k, k);
            for (int i = k + 1; i <= rowEnd; ++i) {
                const Real m = at(i, k) * inv;
                if (m == Real(0)) continue;
                at(i, k) = 0;
                for (int j = k + 1; j <= colEnd; ++j) at(i, j) -= m * at(k, j);
                rhs[i] -= m * rhs[k];
            }
        }
        for (int k = n - 1; k >= 0; --k) {
            Real sum = rhs[k];
            for (int j = k + 1; j <= std::min(n - 1, k + 2 * b); ++j) sum -= at(k, j) * rhs[j];
            rhs[k] = sum / at(k, k);
        }
        return true;
    }
};

// Displace every free DOF by delta: positions additively, orientations by a
// body-frame rotation vector, matching the perturbation the Jacobian uses.
void applyUpdate(Rod& rod, const DofMap& dofs, const std::vector<Real>& delta) {
    RodState& s = rod.state;
    for (std::size_t i = 0; i < s.numParticles(); ++i) {
        const int d = dofs.particle[i];
        if (d >= 0) s.x[i] += Vec3(delta[d], delta[d + 1], delta[d + 2]);
    }
    for (std::size_t j = 0; j < s.numSegments(); ++j) {
        const int d = dofs.segment[j];
        if (d >= 0) s.q[j] = applyBodyDelta(s.q[j], Vec3(delta[d], delta[d + 1], delta[d + 2]));
    }
}

// Central-difference Jacobian. Columns further apart than twice the half-band
// have disjoint row support, so they are perturbed together: 2 (2b + 1)
// residual evaluations however long the rod is.
void buildJacobian(Rod& rod, const DofMap& dofs, Vec3 gravity, Real loadScale, BandMatrix& J) {
    const int n = dofs.size, b = dofs.halfBand;
    const int stride = 2 * b + 1;

    Real lMin = Real(1);
    for (std::size_t k = 0; k < rod.stretch.size(); ++k)
        lMin = k == 0 ? rod.stretch.restLength[k] : std::min(lMin, rod.stretch.restLength[k]);
    // The residual is linear in positions, so their step only has to beat
    // roundoff; orientations enter nonlinearly and the O(eps^2) truncation
    // error sets theirs.
    const Real epsX = Real(1e-5) * lMin, epsQ = Real(1e-5);

    std::vector<Real> epsOf(n);
    for (std::size_t i = 0; i < dofs.particle.size(); ++i)
        if (dofs.particle[i] >= 0)
            for (int c = 0; c < 3; ++c) epsOf[dofs.particle[i] + c] = epsX;
    for (std::size_t j = 0; j < dofs.segment.size(); ++j)
        if (dofs.segment[j] >= 0)
            for (int c = 0; c < 3; ++c) epsOf[dofs.segment[j] + c] = epsQ;

    const std::vector<Vec3> x0 = rod.state.x;
    const std::vector<Quat> q0 = rod.state.q;
    std::vector<Vec3> f, t;
    std::vector<Real> rPlus, rMinus, delta(n);

    for (int group = 0; group < std::min(stride, n); ++group) {
        std::fill(delta.begin(), delta.end(), Real(0));
        for (int d = group; d < n; d += stride) delta[d] = epsOf[d];

        applyUpdate(rod, dofs, delta);
        assemble(rod, dofs, gravity, loadScale, f, t, rPlus);
        rod.state.x = x0;
        rod.state.q = q0;

        for (int d = group; d < n; d += stride) delta[d] = -epsOf[d];
        applyUpdate(rod, dofs, delta);
        assemble(rod, dofs, gravity, loadScale, f, t, rMinus);
        rod.state.x = x0;
        rod.state.q = q0;

        for (int d = group; d < n; d += stride) {
            const Real inv = Real(0.5) / epsOf[d];
            for (int i = std::max(0, d - b); i <= std::min(n - 1, d + b); ++i)
                J.at(i, d) = (rPlus[i] - rMinus[i]) * inv;
        }
    }
}

enum class NewtonResult { Converged, Failed };

NewtonResult newton(Rod& rod, const DofMap& dofs, const StaticParams& p, Real loadScale,
                    int& iterations) {
    const int n = dofs.size;
    std::vector<Vec3> f, t;
    std::vector<Real> r;
    Real prevStep = 0;

    for (int it = 0; it < p.maxNewtonIterations; ++it) {
        ++iterations;
        assemble(rod, dofs, p.gravity, loadScale, f, t, r);
        BandMatrix J(n, dofs.halfBand);
        buildJacobian(rod, dofs, p.gravity, loadScale, J);
        for (Real& v : r) v = -v;
        if (!J.solve(r)) return NewtonResult::Failed;

        Real maxRot = 0, maxStep = 0;
        for (Real v : r) {
            if (!std::isfinite(v)) return NewtonResult::Failed;
            maxStep = std::max(maxStep, std::abs(v));
        }
        for (int d : dofs.segment)
            if (d >= 0) maxRot = std::max(maxRot, norm(Vec3(r[d], r[d + 1], r[d + 2])));
        if (maxRot > p.maxRotationPerIteration)
            for (Real& v : r) v *= p.maxRotationPerIteration / maxRot;

        applyUpdate(rod, dofs, r);

        if (maxStep < p.stepTol) return NewtonResult::Converged;
        // Once quadratic convergence has run into roundoff the update stops
        // shrinking; that floor is the answer, not a failure.
        if (it > 0 && maxStep < Real(1e-7) && maxStep > Real(0.5) * prevStep)
            return NewtonResult::Converged;
        prevStep = maxStep;
    }
    return NewtonResult::Failed;
}

Real infNorm(const Rod& rod, const DofMap& dofs, Vec3 gravity) {
    std::vector<Vec3> f, t;
    std::vector<Real> r;
    assemble(rod, dofs, gravity, Real(1), f, t, r);
    Real m = 0;
    for (Real v : r) m = std::max(m, std::abs(v));
    return m;
}

}  // namespace

void staticResidual(const Rod& rod, Vec3 gravity, Real loadScale, std::vector<Vec3>& f,
                    std::vector<Vec3>& t) {
    const RodState& s = rod.state;
    f.assign(s.numParticles(), Vec3());
    t.assign(s.numSegments(), Vec3());

    // Elastic generalized forces J^T alpha^-1 C, with the Jacobians of
    // solver.cpp: dC/dx0 = -R^T/l, dC/dx1 = R^T/l, dC/dtheta = skew(u).
    for (std::size_t k = 0; k < rod.stretch.size(); ++k) {
        const int i0 = rod.stretch.p0[k], i1 = rod.stretch.p1[k], j = rod.stretch.seg[k];
        const Real l = rod.stretch.restLength[k];
        const Vec3 u = rotateInv(s.q[j], s.x[i1] - s.x[i0]) / l;
        const Vec3 C = u - kE3;
        const Vec3 a = rod.stretch.compliance[k];
        const Vec3 sigma(C.x / a.x, C.y / a.y, C.z / a.z);
        const Vec3 world = rotate(s.q[j], sigma) / l;
        f[i0] -= world;
        f[i1] += world;
        t[j] += cross(sigma, u);
    }
    // dC/dtheta_a = (-p_w I + skew(p_v)) / lbar, dC/dtheta_b = (p_w I + skew(p_v)) / lbar.
    for (std::size_t k = 0; k < rod.bend.size(); ++k) {
        const int a = rod.bend.segA[k], b = rod.bend.segB[k];
        const Real lbar = rod.bend.restLength[k];
        const Quat p = sameHemisphere(conj(s.q[a]) * s.q[b], Quat());
        const Vec3 C = p.im() * (Real(2) / lbar) - rod.bend.restDarboux[k];
        const Vec3 al = rod.bend.compliance[k];
        const Vec3 sigma(C.x / al.x, C.y / al.y, C.z / al.z);
        const Vec3 pvxs = cross(p.im(), sigma);
        t[a] += (sigma * -p.w - pvxs) / lbar;
        t[b] += (sigma * p.w - pvxs) / lbar;
    }

    // Dead loads, ramped together. Torques are stored in world and act on the
    // body-frame rotation vector, the same conversion the dynamics make.
    for (std::size_t i = 0; i < s.numParticles(); ++i)
        f[i] -= (s.extForce[i] + gravity * s.mass[i]) * loadScale;
    for (std::size_t j = 0; j < s.numSegments(); ++j)
        t[j] -= rotateInv(s.q[j], s.extTorque[j]) * loadScale;
}

StaticReport solveStatic(Rod& rod, const StaticParams& p) {
    const DofMap dofs(rod);
    StaticReport report;

    Real lam = 0, dLam = 1;
    int attempts = 0;
    while (lam < Real(1) && attempts < p.maxLoadSteps) {
        ++attempts;
        const Real target = std::min(Real(1), lam + dLam);
        const std::vector<Vec3> x0 = rod.state.x;
        const std::vector<Quat> q0 = rod.state.q;
        int its = 0;
        const NewtonResult res = newton(rod, dofs, p, target, its);
        report.newtonIterations += its;
        if (res == NewtonResult::Converged) {
            lam = target;
            ++report.loadSteps;
            if (its <= 6) dLam *= 2;
        } else {
            rod.state.x = x0;
            rod.state.q = q0;
            dLam *= Real(0.5);
            if (dLam < Real(1e-8)) break;
        }
    }
    report.converged = lam >= Real(1);

    RodState& s = rod.state;
    for (Quat& q : s.q) q = normalize(q);
    s.xPrev = s.x;
    s.qPrev = s.q;
    std::fill(s.v.begin(), s.v.end(), Vec3());
    std::fill(s.omega.begin(), s.omega.end(), Vec3());
    report.residual = infNorm(rod, dofs, p.gravity);
    return report;
}

}  // namespace crs
