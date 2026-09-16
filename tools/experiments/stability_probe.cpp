// Exploratory probe: what actually makes this solver blow up?
//
// Not a validation case -- a tool for finding out which hypothesis the
// validation case should test. Prints, for each configuration, the largest
// timestep that stays bounded, using an energy criterion: a rod that ends up
// with far more kinetic energy than gravity and the applied loads could ever
// have supplied has had energy injected by the integrator, which is what
// "unstable" means. NaN and runaway positions also count.
//
//   rodexp <experiment>     gravity | tipload | iterations
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

#include "core/solver.h"

using namespace crs;

namespace {

struct Setup {
    int n = 32;
    Real youngs = Real(1e7);
    int substeps = 1;
    int iterations = 1;
    Real tipLoadAlpha = 0;  // P L^2 / EI; 0 = gravity only
    bool gravity = true;
    Real simTime = Real(2);
    int minSteps = 400;
};

// Energy the loads could possibly have put in: potential drop of the whole rod
// through its own length, plus the tip load working over twice the length.
double energyBudget(const Rod& rod, const Setup& s, Real P) {
    const double g = s.gravity ? 9.81 : 0.0;
    return double(rod.totalMass()) * g * 1.0 + double(P) * 2.0 + 1e-9;
}

bool bounded(Setup s, Real dt) {
    RodMaterial mat;
    mat.youngs = s.youngs;
    const Real L = Real(1);
    Rod rod = makeStraightRod(s.n, L, mat, Vec3(0, 0, 0), Vec3(1, 0, 0));
    clampRootExact(rod);
    const Real P = s.tipLoadAlpha * mat.bendStiffness() / (L * L);
    if (P > 0) rod.state.extForce.back() = Vec3(0, 0, -P);

    SolverParams p;
    p.dt = dt;
    p.substeps = s.substeps;
    p.iterations = s.iterations;
    p.gravity = s.gravity ? Vec3(0, 0, Real(-9.81)) : Vec3();

    const double budget = energyBudget(rod, s, P);
    // Bounded both ways: at least minSteps so a large dt is really exercised,
    // at most a few thousand so a tiny dt does not run for hours.
    const int steps = std::min(5000, std::max(s.minSteps, int(s.simTime / dt)));
    for (int i = 0; i < steps; ++i) {
        step(rod, p);
        if ((i & 7) == 0) {
            const double ke = double(rod.kineticEnergy());
            if (!std::isfinite(ke) || ke > 10 * budget) return false;
        }
    }
    return std::isfinite(double(rod.kineticEnergy()));
}

double largestBoundedDt(const Setup& s) {
    double lo = 1e-4, hi = 0.5;
    if (bounded(s, Real(hi))) return hi;  // never broke inside the probed range
    if (!bounded(s, Real(lo))) return 0;
    for (int i = 0; i < 16; ++i) {
        const double mid = std::sqrt(lo * hi);
        if (bounded(s, Real(mid)))
            lo = mid;
        else
            hi = mid;
    }
    return lo;
}

void report(const char* label, const Setup& s) {
    const double dt = largestBoundedDt(s);
    const double h = dt / s.substeps;
    const double l = 1.0 / s.n;
    std::printf("%-28s n=%-4d E=%-8.0e sub=%-3d it=%-3d alpha=%-5.1f  max dt=%-10.4g h=%-10.4g %s\n",
                label, s.n, double(s.youngs), s.substeps, s.iterations, double(s.tipLoadAlpha), dt,
                h, dt >= 0.5 ? "(never broke)" : "");
    std::fflush(stdout);
    (void)l;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string what = argc > 1 ? argv[1] : "gravity";

    if (what == "gravity") {
        for (double E : {1e6, 1e8, 1e10})
            for (int n : {16, 64})
                for (int sub : {1, 8}) {
                    Setup s;
                    s.n = n;
                    s.youngs = Real(E);
                    s.substeps = sub;
                    report("gravity cantilever", s);
                }
    } else if (what == "tipload") {
        for (double alpha : {1.0, 10.0, 100.0})
            for (int n : {16, 64}) {
                Setup s;
                s.n = n;
                s.gravity = false;
                s.tipLoadAlpha = Real(alpha);
                report("tip-loaded cantilever", s);
            }
    } else if (what == "iterations") {
        for (int it : {1, 8, 64})
            for (int n : {16, 64}) {
                Setup s;
                s.n = n;
                s.iterations = it;
                s.gravity = false;
                s.tipLoadAlpha = Real(10);
                report("tip load, iteration sweep", s);
            }
    } else {
        std::printf("usage: rodexp gravity|tipload|iterations\n");
        return 2;
    }
    return 0;
}
