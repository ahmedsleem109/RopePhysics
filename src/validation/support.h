// Shared plumbing for the validation cases: CSV output, formatting, and the
// reference material and solver settings every static case uses.
#pragma once

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>

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

}  // namespace crs
