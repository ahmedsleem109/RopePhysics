// Exploratory probe: is the twist-buckling threshold biased by how long the
// case watches for growth, and does measuring the growth rate avoid it?
//
// Same rod, twist and seed as the twist-buckling case.
//
//   rodtwist window <segments> <seconds> [growth factor = 100]
//       bisect on "amplitude grew by the factor within the window", as the case
//       does. For an undamped supercritical pitchfork the growth rate goes like
//       sqrt(Phi - Phi_crit), so a finite window overshoots by roughly 1/T^2.
//   rodtwist rate <segments> <overshoot %>...
//       measure the exponential growth rate sigma above threshold at each twist
//       (Phi_ref times 1 + overshoot), and extrapolate sigma^2, linear in Phi
//       near onset, to zero. No window enters.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "core/solver.h"
#include "validation/support.h"

using namespace crs;

namespace {

const RodMaterial kMat = referenceMaterial();
const Real kL = Real(1);
const Real kSeed = Real(1e-7);

double phiCritRef() { return 8.986819 * kMat.bendStiffness() / kMat.twistStiffness(); }

Rod twistedRod(int n, double phi) {
    Rod rod = makeStraightRod(n, kL, kMat, Vec3(0, 0, 0), Vec3(0, 0, 1));
    clampEndExact(rod, false);
    const int tipGhost = clampEndExact(rod, true);
    const Real l = rod.stretch.restLength.front();
    const int nSeg = static_cast<int>(rod.stretch.size());
    for (int j = 0; j < nSeg; ++j) {
        const Real sMid = (Real(j) + Real(0.5)) * l;
        rod.state.q[j] = normalize(quatAxisAngle(Vec3(0, 0, 1), Real(phi) * sMid / kL) * rod.state.q[j]);
    }
    rod.state.q[tipGhost] = normalize(quatAxisAngle(Vec3(0, 0, 1), Real(phi)) * rod.state.q[tipGhost]);
    rod.state.qPrev = rod.state.q;
    for (std::size_t i = 1; i + 1 < rod.state.numParticles(); ++i)
        rod.state.x[i].x += kSeed * std::sin(kPi * rod.state.x[i].z / kL);
    rod.state.xPrev = rod.state.x;
    return rod;
}

SolverParams params() {
    SolverParams p;
    p.dt = Real(1e-3);
    p.substeps = std::getenv("CRS_SUBSTEPS") ? std::atoi(std::getenv("CRS_SUBSTEPS")) : 256;
    p.iterations = 1;
    p.gravity = Vec3();
    return p;
}

double lateral(const Rod& rod) {
    double amp = 0;
    for (const Vec3& x : rod.state.x) amp = std::max(amp, std::sqrt(x.x * x.x + x.y * x.y));
    return amp;
}

int window(int n, double seconds, double growth) {
    const int steps = int(std::lround(seconds / 1e-3));
    auto buckles = [&](double phi) {
        Rod rod = twistedRod(n, phi);
        const SolverParams p = params();
        for (int i = 0; i < steps; ++i) {
            step(rod, p);
            if (lateral(rod) > growth * double(kSeed)) return true;
        }
        return false;
    };
    const double ref = phiCritRef();
    double lo = 0.9 * ref, hi = 1.2 * ref;
    for (int it = 0; it < 10; ++it) {
        const double mid = 0.5 * (lo + hi);
        (buckles(mid) ? hi : lo) = mid;
    }
    const double phi = 0.5 * (lo + hi);
    std::printf("window n=%d T=%.1fs growth=%.0f: phi_crit %.5f vs %.5f (%+.3f%%)\n", n, seconds, growth,
                phi, ref, 100 * (phi - ref) / ref);
    return 0;
}

// Growth rate from the time the amplitude takes to go from 1e-5 to 1e-3 m: past
// the seed's transient, well short of the ~1e-2 m the axial constraint
// saturates it at. 0 if it never gets there in 30 s.
double growthRate(int n, double phi) {
    Rod rod = twistedRod(n, phi);
    const SolverParams p = params();
    double t1 = -1;
    for (int i = 1; i <= 30000; ++i) {
        step(rod, p);
        const double a = lateral(rod);
        if (t1 < 0 && a > 1e-5) t1 = i * 1e-3;
        if (a > 1e-3) return std::log(100.0) / (i * 1e-3 - t1);
    }
    return 0;
}

int rate(int n, const std::vector<double>& overshoots) {
    const double ref = phiCritRef();
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    int m = 0;
    for (double o : overshoots) {
        const double phi = ref * (1 + o / 100);
        const double sigma = growthRate(n, phi);
        std::printf("  n=%d phi=%.5f (+%.1f%%): sigma %.4f 1/s\n", n, phi, o, sigma);
        std::fflush(stdout);
        if (sigma <= 0) continue;
        const double y = sigma * sigma;
        sx += phi, sy += y, sxx += phi * phi, sxy += phi * y;
        ++m;
    }
    if (m < 2) return 1;
    const double slope = (m * sxy - sx * sy) / (m * sxx - sx * sx);
    const double icept = (sy - slope * sx) / m;
    const double phiCrit = -icept / slope;
    std::printf("rate n=%d: phi_crit %.5f vs %.5f (%+.3f%%)\n", n, phiCrit, ref, 100 * (phiCrit - ref) / ref);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 4 && std::strcmp(argv[1], "window") == 0)
        return window(std::atoi(argv[2]), std::atof(argv[3]), argc > 4 ? std::atof(argv[4]) : 100.0);
    if (argc >= 4 && std::strcmp(argv[1], "rate") == 0) {
        std::vector<double> overshoots;
        for (int i = 3; i < argc; ++i) overshoots.push_back(std::atof(argv[i]));
        return rate(std::atoi(argv[2]), overshoots);
    }
    std::fprintf(stderr, "usage: rodtwist window <segments> <seconds> [growth]\n"
                         "       rodtwist rate <segments> <overshoot %%>...\n");
    return 2;
}
