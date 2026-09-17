// Shared plumbing for the validation cases: CSV output, formatting, and the
// reference material and solver settings every static case uses.
#pragma once

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "../core/solver.h"

namespace crs {

// Minimal CSV writer. Silently does nothing when the output directory is empty,
// so cases can be run with --no-out without branching everywhere.
struct Csv {
    std::ofstream out;
    bool open = false;

    Csv(const std::string& dir, const std::string& file, const std::string& header) {
        if (dir.empty()) return;
        out.open(dir + "/" + file);
        if (!out) return;
        open = true;
        out << header << "\n";
        out.precision(12);
    }

    template <typename... Args>
    void row(Args... args) {
        if (!open) return;
        writeRow(args...);
    }

   private:
    void writeRow() { out << "\n"; }
    template <typename T, typename... Rest>
    void writeRow(T first, Rest... rest) {
        out << first;
        if constexpr (sizeof...(rest) > 0) out << ",";
        writeRow(rest...);
    }
};

inline std::string fmt(const char* f, double a, double b = 0, double c = 0) {
    char buf[256];
    std::snprintf(buf, sizeof(buf), f, a, b, c);
    return buf;
}

// The reference material for the elastic cases: soft enough that the rod
// relaxes in a few seconds of simulated time, slender enough (L/r = 200) that
// shear and axial compliance stay well below the discretization error.
inline RodMaterial referenceMaterial() {
    RodMaterial m;
    m.youngs = Real(1e7);
    m.poisson = Real(0.35);
    m.density = Real(1000);
    m.radius = Real(5e-3);
    return m;
}

// First bending eigenfrequency of a cantilever. Damping at about 2*omega_1
// approaches the static state fast without overdamping it into a crawl: the
// equilibrium does not depend on the damping, but the time to reach it does,
// and both too little and too much cost thousands of steps.
inline Real cantileverOmega1(const RodMaterial& m, Real L) {
    return Real(3.516) * std::sqrt(m.bendStiffness() / (m.density * m.area() * L * L * L * L));
}

// Sweep budget for a static solve.
//
// Gauss-Seidel moves information one element per sweep, and the operator it is
// inverting here is the fourth-order beam operator, whose conditioning grows
// like n^4. The consequence is visible in the data: an under-solved rod relaxes
// to a stationary state that is genuinely too SOFT -- the bending constraint is
// never fully satisfied -- and it reports itself converged, because the
// velocities really have gone to zero. n = 32 settles at about 256 sweeps per
// step; n = 64 does not settle at 512. A quadratic budget tracks it over the
// range the suite covers.
//
// Substeps are the other half: below about 4, no sweep budget converges,
// because alpha/h^2 is then too small to make the system diagonally dominant.
//
// The angular limit is tight on purpose. XPBD's static equilibrium is only
// h-independent once alpha/h^2 dominates J M^-1 J^T, and a torque-loaded rod
// reaches that regime late: at maxAngle = 0.1 every joint in the pure-moment
// case settles carrying 0.77% more moment than was applied -- a converged,
// mesh-independent bias that no amount of extra sweeps removes. At
// maxAngle = 0.00625 the same joints match the applied moment to 6e-5 and the
// case converges at second order. Force-loaded cases are unaffected: with no
// applied torque this limit never binds.
inline SolverParams staticParams(const Rod& rod, Real damping) {
    SolverParams p;
    p.dt = Real(0.01);
    p.gravity = Vec3();
    p.substeps = std::max(4, stableSubsteps(rod, p.gravity, p.dt, Real(0.05), Real(0.00625)));
    const int n = static_cast<int>(rod.stretch.size());
    const int budget = std::max(256, n * n);
    p.iterations = std::max(64, (budget + p.substeps - 1) / p.substeps);
    p.linearDamping = damping;
    p.angularDamping = damping;
    return p;
}

// Reference solution of the inextensible planar elastica under a transverse tip
// load, by shooting on the exact BVP rather than elliptic integrals:
//
//   EI theta'' = -P cos(theta),   theta(0) = 0,   theta'(L) = 0
//
// theta is measured from the undeformed axis toward the load. Integrating the
// tangent afterwards gives the tip position.
struct ElasticaRef {
    double tipX = 0, tipZ = 0, tipAngle = 0;
};

//
// With EA and kGA given (0 = rigid), it is Reissner's extensible, shearable
// elastica instead -- the continuum the Cosserat rod actually discretizes. The
// internal force is the tip load throughout, so the axial strain is
// P sin(theta) / EA and the shear strain -P cos(theta) / kGA, the centreline
// tangent is r' = (1 + eps) t + gamma n with t = (cos, -sin), n = (sin, cos),
// and moment balance reads EI theta'' = -P x'.
inline ElasticaRef solveElasticaRef(double P, double L, double EI, double EA = 0, double kGA = 0,
                             int steps = 20000) {
    const double a = P / EI;
    auto dxds = [&](double th) {
        const double eps = EA > 0 ? P * std::sin(th) / EA : 0.0;
        const double gamma = kGA > 0 ? -P * std::cos(th) / kGA : 0.0;
        return (1 + eps) * std::cos(th) + gamma * std::sin(th);
    };
    auto dzds = [&](double th) {
        const double eps = EA > 0 ? P * std::sin(th) / EA : 0.0;
        const double gamma = kGA > 0 ? -P * std::cos(th) / kGA : 0.0;
        return -(1 + eps) * std::sin(th) + gamma * std::cos(th);
    };
    auto shoot = [&](double kappa0, ElasticaRef* out) {
        const double ds = L / steps;
        double th = 0, dth = kappa0, x = 0, z = 0;
        for (int i = 0; i < steps; ++i) {
            // RK4 on (theta, theta') with theta'' = -a x'(theta).
            const double k1t = dth, k1d = -a * dxds(th);
            const double k2t = dth + 0.5 * ds * k1d, k2d = -a * dxds(th + 0.5 * ds * k1t);
            const double k3t = dth + 0.5 * ds * k2d, k3d = -a * dxds(th + 0.5 * ds * k2t);
            const double k4t = dth + ds * k3d, k4d = -a * dxds(th + ds * k3t);
            // Trapezoid the tangent over the same interval for x and z.
            const double thNext = th + ds * (k1t + 2 * k2t + 2 * k3t + k4t) / 6.0;
            x += 0.5 * ds * (dxds(th) + dxds(thNext));
            z += 0.5 * ds * (dzds(th) + dzds(thNext));
            dth += ds * (k1d + 2 * k2d + 2 * k3d + k4d) / 6.0;
            th = thNext;
        }
        if (out) {
            out->tipX = x;
            out->tipZ = z;
            out->tipAngle = th;
        }
        return dth;  // theta'(L), zero at the solution
    };

    // theta'(L) grows monotonically with the shooting parameter, so bisect.
    double lo = 0, hi = 10 * a * L + 10;
    for (int i = 0; i < 64; ++i) {  // halves a bracket of O(10) past double precision
        const double mid = 0.5 * (lo + hi);
        if (shoot(mid, nullptr) > 0)
            hi = mid;
        else
            lo = mid;
    }
    ElasticaRef ref;
    shoot(0.5 * (lo + hi), &ref);
    return ref;
}

// Least-squares slope of log10(y) against log10(x). Used to report the observed
// convergence order; reported, never asserted away.
inline double logLogSlope(const std::vector<double>& xs, const std::vector<double>& ys) {
    const std::size_t n = xs.size();
    if (n < 2) return 0;
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const double lx = std::log10(xs[i]), ly = std::log10(ys[i]);
        sx += lx;
        sy += ly;
        sxx += lx * lx;
        sxy += lx * ly;
    }
    const double d = n * sxx - sx * sx;
    return d == 0 ? 0 : (n * sxy - sx * sy) / d;
}

// Radius, pitch and turns of a rod that should be a helix, measured from the
// centerline alone -- never from the constraint that produced it. `lastSeg` is
// the last real segment (ghost frames are appended after it).
struct HelixShape {
    double radius = 0, pitch = 0, turns = 0;
};

inline HelixShape measureHelix(const Rod& rod, int lastSeg) {
    // Axis: a rod with a constant Darboux vector is a rigid screw, so the
    // material frame rotates about the helix axis and nothing else. The axis of
    // the world-frame rotation carrying the first frame onto the last is
    // therefore the helix axis exactly.
    //
    // (Averaging the tangents also "works", but only over a whole number of
    // turns: over 1.78 turns it tilts the axis enough to bias a circle fit by
    // 8%, which is how this measurement first went wrong.)
    const Quat rel = sameHemisphere(rod.state.q[lastSeg] * conj(rod.state.q[0]), Quat());
    Vec3 axis = normalize(rel.im());

    // Two in-plane basis vectors perpendicular to the axis.
    Vec3 e1 = std::abs(axis.x) < Real(0.9) ? Vec3(1, 0, 0) : Vec3(0, 1, 0);
    e1 = normalize(e1 - axis * dot(e1, axis));
    const Vec3 e2 = cross(axis, e1);

    // Algebraic (Kasa) circle fit of the projected centerline.
    double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0, sxz = 0, syz = 0, sz = 0;
    std::vector<double> us, vs, ws;
    const std::size_t nP = rod.state.numParticles();
    for (std::size_t i = 0; i < nP; ++i) {
        const Vec3 x = rod.state.x[i];
        const double u = dot(x, e1), v = dot(x, e2);
        us.push_back(u);
        vs.push_back(v);
        ws.push_back(dot(x, axis));
        const double zz = u * u + v * v;
        sx += u; sy += v; sxx += u * u; syy += v * v; sxy += u * v;
        sxz += u * zz; syz += v * zz; sz += zz;
    }
    const double N = double(nP);
    const double a11 = 2 * (sxx - sx * sx / N), a12 = 2 * (sxy - sx * sy / N);
    const double a22 = 2 * (syy - sy * sy / N);
    const double b1 = sxz - sx * sz / N, b2 = syz - sy * sz / N;
    const double det = a11 * a22 - a12 * a12;
    const double cu = (b1 * a22 - b2 * a12) / det, cv = (a11 * b2 - a12 * b1) / det;
    double radius = 0;
    for (std::size_t i = 0; i < nP; ++i)
        radius += std::sqrt((us[i] - cu) * (us[i] - cu) + (vs[i] - cv) * (vs[i] - cv));
    radius /= N;

    // Pitch: unwrap the angle about the axis and regress axial position on it.
    double prev = 0, unwrapped = 0;
    std::vector<double> phis;
    for (std::size_t i = 0; i < nP; ++i) {
        const double ang = std::atan2(vs[i] - cv, us[i] - cu);
        if (i > 0) {
            double d = ang - prev;
            while (d > kPi) d -= 2 * kPi;
            while (d < -kPi) d += 2 * kPi;
            unwrapped += d;
        }
        prev = ang;
        phis.push_back(unwrapped);
    }
    double pp = 0, pw = 0, ppp = 0, ppw = 0;
    for (std::size_t i = 0; i < nP; ++i) {
        pp += phis[i]; pw += ws[i]; ppp += phis[i] * phis[i]; ppw += phis[i] * ws[i];
    }
    const double slope = (N * ppw - pp * pw) / (N * ppp - pp * pp);
    HelixShape shape;
    shape.radius = radius;
    shape.pitch = std::abs(slope) * 2 * kPi;
    shape.turns = std::abs(phis.back()) / (2 * kPi);
    return shape;
}

}  // namespace crs
