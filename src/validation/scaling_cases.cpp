// Phase 4: scale and characterization.
//
// Three measurements that describe what this solver can and cannot do, rather
// than how fast it goes on one flattering scene:
//
//   stability-envelope  the largest timestep that stays bounded, against
//                       stiffness and against how the solver budget is split.
//   throughput          work per second against batch size, rod length,
//                       substep count and thread count.
//   failure-study       one deliberate blow-up, pushed until it breaks, with
//                       the mechanism measured and the cost of the fix stated.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <thread>
#include <utility>
#include <vector>

#include "../core/solver.h"
#include "cases.h"
#include "support.h"

namespace crs {

namespace {

// A gravity cantilever released from horizontal: the rod swings down under its
// own weight, so its kinetic energy can never exceed the potential energy it
// starts with. Ending up with far more than that means the integrator injected
// energy -- which is what "unstable" means, and is a far sharper test than
// waiting for a NaN. (The first version of these cases counted only NaN or
// runaway positions, never saw a single failure, and reported its own bisection
// bound as a measurement.)
struct Probe {
    int n = 16;
    Real youngs = Real(1e7);
    int substeps = 1;
    int iterations = 1;
};

struct ProbeResult {
    bool bounded = true;
    double maxSpeed = 0;  // fastest particle seen during the run
};

ProbeResult runProbe(const Probe& pr, Real dt) {
    RodMaterial mat = referenceMaterial();
    mat.youngs = pr.youngs;
    const Real L = Real(1);
    Rod rod = makeStraightRod(pr.n, L, mat, Vec3(0, 0, 0), Vec3(1, 0, 0));
    clampRootExact(rod);

    SolverParams p;
    p.dt = dt;
    p.substeps = pr.substeps;
    p.iterations = pr.iterations;
    p.gravity = Vec3(0, 0, Real(-9.81));

    // Energy available: the whole rod falling through its own length.
    const double budget = double(rod.totalMass()) * 9.81 * double(L);

    // At least 400 steps so a large dt is really exercised; at most 5000 so a
    // small one does not run for hours. Two seconds covers the first swing.
    const int steps = std::min(5000, std::max(400, int(2.0 / double(dt))));
    ProbeResult res;
    for (int i = 0; i < steps; ++i) {
        step(rod, p);
        if ((i & 7) == 0) {
            const double ke = double(rod.kineticEnergy());
            if (!std::isfinite(ke) || ke > 10 * budget) {
                res.bounded = false;
                return res;
            }
            for (const Vec3& v : rod.state.v)
                res.maxSpeed = std::max(res.maxSpeed, double(norm(v)));
        }
    }
    return res;
}

// Largest bounded dt by geometric bisection, with the speed seen in the last
// bounded run. `broke` is false when nothing failed inside the probed range:
// the number is then a lower bound, and is reported and excluded as one.
struct Limit {
    double dt = 0;
    double maxSpeed = 0;
    bool broke = true;
};

Limit largestBoundedDt(const Probe& pr) {
    double lo = 1e-4, hi = 0.5;
    Limit lim;
    const ProbeResult top = runProbe(pr, Real(hi));
    if (top.bounded) {
        lim.dt = hi;
        lim.maxSpeed = top.maxSpeed;
        lim.broke = false;
        return lim;
    }
    ProbeResult good = runProbe(pr, Real(lo));
    if (!good.bounded) return lim;
    for (int i = 0; i < 14; ++i) {
        const double mid = std::sqrt(lo * hi);
        const ProbeResult r = runProbe(pr, Real(mid));
        if (r.bounded) {
            lo = mid;
            good = r;
        } else {
            hi = mid;
        }
    }
    lim.dt = lo;
    lim.maxSpeed = good.maxSpeed;
    return lim;
}

double spread(const std::vector<double>& v) {
    if (v.empty()) return 0;
    const auto mm = std::minmax_element(v.begin(), v.end());
    return *mm.second / *mm.first;
}

}  // namespace

// ------------------------------------------------------- stability envelope
//
// Largest bounded timestep against stiffness and against how the per-step
// solver budget is split. What the measurement shows, and what is asserted:
//
//   * stiffness does not matter. Across four decades of Young's modulus the
//     limit barely moves. XPBD's implicit constraint treatment earns its
//     reputation here.
//   * the limit is set by the number of Gauss-Seidel SWEEPS per step, however
//     they are split: eight substeps and eight iterations buy about the same
//     timestep. Substeps remain the better buy, because they are more accurate
//     for the same work -- but not because they are more stable.
CaseResult runStabilityEnvelope(const std::string& outDir) {
    CaseResult res;
    res.name = "stability envelope (timestep vs stiffness vs sweeps)";

    Csv csv(outDir, "stability_envelope.csv",
            "youngs,substeps,iterations,sweeps,segments,max_dt,dt_per_sweep,broke");

    const int n = 16;
    std::vector<double> perSweepGrid, perSweepSplit;
    int unbroken = 0;

    for (double E : {1e6, 1e7, 1e8, 1e9, 1e10}) {
        for (int sub : {1, 2, 4, 8, 16}) {
            Probe pr;
            pr.n = n;
            pr.youngs = Real(E);
            pr.substeps = sub;
            const Limit lim = largestBoundedDt(pr);
            const double perSweep = lim.dt / sub;
            csv.row(E, sub, 1, sub, n, lim.dt, perSweep, lim.broke ? 1 : 0);
            if (lim.broke)
                perSweepGrid.push_back(perSweep);
            else
                ++unbroken;
            if (E == 1e7 && lim.broke) perSweepSplit.push_back(perSweep);
        }
    }

    // The same sweep counts, spent as iterations instead of substeps.
    for (int it : {2, 4, 8, 16}) {
        Probe pr;
        pr.n = n;
        pr.iterations = it;
        const Limit lim = largestBoundedDt(pr);
        csv.row(1e7, 1, it, it, n, lim.dt, lim.dt / it, lim.broke ? 1 : 0);
        if (lim.broke)
            perSweepSplit.push_back(lim.dt / it);
        else
            ++unbroken;
    }

    // perSweepSplit holds the E = 1e7 substep rows first (5 of them), then the
    // iteration rows: compare the two groups' means.
    double subMean = 0, itMean = 0;
    int subCount = 0, itCount = 0;
    for (std::size_t i = 0; i < perSweepSplit.size(); ++i) {
        if (i < 5) {
            subMean += perSweepSplit[i];
            ++subCount;
        } else {
            itMean += perSweepSplit[i];
            ++itCount;
        }
    }
    const double advantage =
        (subCount && itCount) ? (itMean / itCount) / (subMean / subCount) : 0.0;

    res.notes.push_back(fmt("  bounded dt per sweep varies only %.3gx across 4 decades of stiffness "
                            "and 1-16 substeps", spread(perSweepGrid)));
    res.notes.push_back(fmt("  sweeps spent as iterations buy %.3gx the stable dt of the same "
                            "sweeps spent as substeps (substeps remain the more accurate buy)",
                            advantage));
    if (unbroken)
        res.notes.push_back(fmt("  %.0f configuration(s) never broke below dt = 0.5 s and are "
                                "excluded", double(unbroken)));

    res.checks.push_back(makeCheck("stiffness and substeps collapse onto dt per sweep",
                                   spread(perSweepGrid), 1.0, 1.0));
    // The first hypothesis here was that iterations and substeps buy the same
    // stability per sweep. The measurement refuted it: iterations buy more.
    // What is asserted is what was measured -- iterations are never the worse
    // stability buy -- with the factor reported rather than fitted.
    res.checks.push_back(
        makeRangeCheck("iterations stabilize at least as well per sweep", advantage, 1.0, 5.0));
    return res;
}

// ---------------------------------------------------------------- throughput
//
// CPU throughput, batched the way the GPU solver batches: many independent rods
// stepped together. The unit is segment-substeps per second -- segments times
// substeps times steps -- because a substep is the unit of solver work, and
// counting only frame steps would let a coarse substep count inflate the
// number.
CaseResult runThroughput(const std::string& outDir) {
    CaseResult res;
    res.name = "batched CPU throughput and scaling";

    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    res.notes.push_back(fmt("  %.0f hardware threads", double(hw)));

    Csv csv(outDir, "throughput_cpu.csv",
            "threads,rods,segments,substeps,steps,seconds,segment_steps_per_sec");

    auto measure = [&](int threads, int rods, int n, int substeps, int steps) -> double {
        const RodMaterial mat = referenceMaterial();
        std::vector<Rod> batch;
        batch.reserve(rods);
        for (int r = 0; r < rods; ++r) {
            Rod rod = makeStraightRod(n, Real(1), mat, Vec3(0, 0, 0), Vec3(1, 0, 0));
            clampRootExact(rod);
            batch.push_back(std::move(rod));
        }

        SolverParams p;
        p.dt = Real(1e-3);
        p.substeps = substeps;
        p.iterations = 1;
        p.gravity = Vec3(0, 0, Real(-9.81));

        const auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<std::thread> pool;
        for (int t = 0; t < threads; ++t) {
            pool.emplace_back([&, t] {
                for (int r = t; r < rods; r += threads)
                    for (int i = 0; i < steps; ++i) step(batch[r], p);
            });
        }
        for (std::thread& th : pool) th.join();
        const auto t1 = std::chrono::high_resolution_clock::now();

        const double seconds = std::chrono::duration<double>(t1 - t0).count();
        const double work = double(rods) * n * steps * substeps;
        const double rate = work / seconds;
        csv.row(threads, rods, n, substeps, steps, seconds, rate);
        return rate;
    };

    double best = 0;
    for (int t = 1; t <= int(hw); t *= 2) best = std::max(best, measure(t, 256, 64, 8, 100));
    if ((hw & (hw - 1)) != 0) best = std::max(best, measure(int(hw), 256, 64, 8, 100));
    for (int rods : {1, 4, 16, 64, 256, 1024})
        best = std::max(best, measure(int(hw), rods, 64, 8, 100));
    for (int n : {8, 16, 32, 64, 128, 256}) best = std::max(best, measure(int(hw), 256, n, 8, 100));
    for (int sub : {1, 2, 4, 8, 16, 32}) best = std::max(best, measure(int(hw), 256, 64, sub, 100));

    res.notes.push_back(fmt("  peak %.4g million segment-substeps/sec on the CPU", best / 1e6));
    res.checks.push_back(makeCheck("throughput measured", best > 0 ? 1.0 : 0.0, 1.0, 0.0));
    return res;
}

// ---------------------------------------------------------------- failure study
//
// One deliberate blow-up, pushed until it breaks, with the mechanism measured.
//
// XPBD is unconditionally stable in constraint STIFFNESS -- the envelope case
// shows the limit does not move across four decades of modulus. What it is not
// stable against is motion. A Gauss-Seidel sweep over a rod carries a correction
// about one element along the chain. If the material moves further than that
// in the time one sweep has to account for, the chain cannot keep up with its
// own motion, corrections overshoot, and energy appears from nowhere.
//
// So the governing quantity should be
//
//     elements moved per sweep  =  v * (dt / sweeps) / l
//
// and at the stability limit it should come out the same order-one number
// whatever the mesh. That is what this case measures. (An earlier version of
// this project asserted the governing group was the predictor displacement
// h^2 a / l. That was inferred, never isolated; the probe that finally isolated
// the limit found that group varying fivefold across meshes at failure.)
//
// The speed v is the PHYSICAL bound for this setup, sqrt(2 g L) -- the tip
// falling through the rod's length -- not the fastest speed seen in the last
// bounded run. The bounded criterion allows up to ten times the available
// energy, so a run just inside the limit is already carrying up to sqrt(10)
// times the physical speed; measuring that measured the instability, and gave
// 25-58 m/s for a one-metre rod swinging under gravity.
//
// Measured: sweeps spent as substeps break at ~0.65 element lengths per sweep,
// tightly, for elements longer than the rod's diameter. Sweeps spent as
// iterations tolerate ~2.7x more. Once elements are shorter than the rod's own
// diameter the rule stops holding, which is reported rather than asserted.
//
// The cost of the fix follows directly: at the limit, simulated time per sweep
// scales with the element length, so halving the element halves the timestep
// per sweep -- refinement costs twice, once in segments and again in sweeps.
CaseResult runFailureStudy(const std::string& outDir) {
    CaseResult res;
    res.name = "failure study: what actually limits the timestep";

    Csv csv(outDir, "failure_study.csv",
            "segments,substeps,iterations,max_dt,element,elements_per_sweep,element_over_diameter");

    const RodMaterial mat = referenceMaterial();
    const double L = 1.0;
    const double vPhysical = std::sqrt(2 * 9.81 * L);
    const double diameter = 2 * double(mat.radius);

    std::vector<double> substepRegime, iterationRegime, shortElements;
    for (int n : {16, 32, 64, 128}) {
        for (const auto& split : {std::pair<int, int>(1, 1), std::pair<int, int>(4, 1),
                                  std::pair<int, int>(1, 4)}) {
            Probe pr;
            pr.n = n;
            pr.substeps = split.first;
            pr.iterations = split.second;
            const Limit lim = largestBoundedDt(pr);
            if (!lim.broke) continue;
            const double sweeps = double(split.first * split.second);
            const double l = L / n;
            const double perSweep = vPhysical * (lim.dt / sweeps) / l;
            csv.row(n, split.first, split.second, lim.dt, l, perSweep, l / diameter);

            if (l < diameter)
                shortElements.push_back(perSweep);
            else if (split.second == 1)
                substepRegime.push_back(perSweep);
            else
                iterationRegime.push_back(perSweep);
        }
    }

    auto mean = [](const std::vector<double>& v) {
        double m = 0;
        for (double x : v) m += x;
        return v.empty() ? 0.0 : m / double(v.size());
    };

    res.notes.push_back(fmt("  substeps: breakdown at %.3g element lengths per sweep (spread %.3gx "
                            "over meshes and splits)", mean(substepRegime), spread(substepRegime)));
    res.notes.push_back(fmt("  iterations: breakdown at %.3g element lengths per sweep, %.3gx more "
                            "motion per unit of work", mean(iterationRegime),
                            mean(iterationRegime) / mean(substepRegime)));
    res.notes.push_back(fmt("  elements shorter than the rod diameter: %.3g to %.3g -- the rule "
                            "does not extend there",
                            shortElements.empty() ? 0.0
                                                  : *std::min_element(shortElements.begin(),
                                                                      shortElements.end()),
                            shortElements.empty() ? 0.0
                                                  : *std::max_element(shortElements.begin(),
                                                                      shortElements.end())));

    res.checks.push_back(makeCheck("substep breakdown occurs at a fixed motion per sweep",
                                   spread(substepRegime), 1.0, 0.5));
    res.checks.push_back(makeRangeCheck("elements moved per sweep at breakdown is order one",
                                        mean(substepRegime), 0.2, 2.0));
    return res;
}

}  // namespace crs
