// Phase 4: scale and characterization.
//
// Two measurements that describe what this solver can and cannot do, rather
// than how fast it goes on one flattering scene:
//
//   timestep-envelope   the largest timestep that keeps the rod accurate, what
//                       sets it, and how the solver budget is best spent.
//   throughput          work per second against batch size, rod length,
//                       substep count and thread count.
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

// ---------------------------------------------------------------- timestep probe
//
// A cantilever released from horizontal swings down under gravity for one
// second. The measure of accuracy is the worst stretch/shear strain anywhere in
// the rod at any time. The physical strain of this motion is small (7e-4 at
// E = 1e8, 5e-5 at 1e9), so strain far above it is constraint error the solver
// failed to remove: the rod is behaving like rubber rather than like the
// material it was given.
//
// This replaced an energy criterion ("kinetic energy never exceeds ten times
// what gravity could supply"). That one only detects eventual explosions, and
// it was badly behaved: whether a run exploded was not monotone in dt, and
// runs it called stable carried 50-250% strain. Worst strain rises smoothly
// with dt, so bisection on it is meaningful.
struct Probe {
    int n = 16;
    Real youngs = Real(1e8);
    int substeps = 1;
    int iterations = 1;
};

const double kStrainTol = 0.01;

double worstStrain(const Probe& pr, double dt) {
    RodMaterial mat = referenceMaterial();
    mat.youngs = pr.youngs;
    Rod rod = makeStraightRod(pr.n, Real(1), mat, Vec3(0, 0, 0), Vec3(1, 0, 0));
    clampRootExact(rod);

    SolverParams p;
    p.dt = Real(dt);
    p.substeps = pr.substeps;
    p.iterations = pr.iterations;
    p.gravity = Vec3(0, 0, Real(-9.81));

    const int steps = int(std::ceil(1.0 / dt));
    double worst = 0;
    for (int i = 0; i < steps; ++i) {
        step(rod, p);
        for (std::size_t k = 0; k < rod.stretch.size(); ++k) {
            const Vec3 d = rod.state.x[rod.stretch.p1[k]] - rod.state.x[rod.stretch.p0[k]];
            const Vec3 u = rotateInv(rod.state.q[rod.stretch.seg[k]], d) / rod.stretch.restLength[k];
            worst = std::max(worst, double(norm(u - Vec3(0, 0, 1))));
        }
        // Far past the tolerance: stop early, the answer is already "too big".
        if (!(worst < 1.0)) return 1e9;
    }
    return worst;
}

// Largest substep h (= dt / substeps) keeping the worst strain within
// tolerance. Bracketed around the kinematic estimate h ~ 0.02 l / v, then
// bisected geometrically to 2%.
double largestAccurateSubstep(const Probe& pr) {
    const double v = std::sqrt(2 * 9.81);
    const double guess = 0.02 / (pr.n * v);
    double lo = guess / 16, hi = guess * 16;
    auto ok = [&](double h) { return worstStrain(pr, h * pr.substeps) <= kStrainTol; };
    if (!ok(lo)) return 0;
    if (ok(hi)) return hi;
    while (hi / lo > 1.02) {
        const double mid = std::sqrt(lo * hi);
        (ok(mid) ? lo : hi) = mid;
    }
    return lo;
}

double spread(const std::vector<double>& v) {
    if (v.empty()) return 0;
    const auto mm = std::minmax_element(v.begin(), v.end());
    return *mm.second / *mm.first;
}

}  // namespace

// ------------------------------------------------------------ timestep envelope
//
// How large a timestep keeps the rod accurate, and what sets that limit.
// Measured with the probe above, across stiffness (E = 1e7..1e9), mesh
// (16..64 segments) and how a fixed solver budget is spent.
//
// Only the substep h matters, not how substeps are grouped into frames: a frame
// of four substeps is exactly four substeps. So the envelope is stated in h.
//
// Findings:
//
//   * Iterations versus substeps depends on stiffness. Four sweeps spent as
//     iterations allow up to ~2x the dt of four substeps on a soft rod and well
//     under half on a stiff one. Reported, not asserted either way.
//   * The limit is kinematic. At the limit, material moves about 1-4% of an
//     element length per substep (v h / l with v = sqrt(2 g L)), within a 3x
//     spread across meshes and two decades of stiffness. Refining the rod
//     therefore shrinks the usable substep in proportion to the element length.
//     Asserted: the spread, and that the value is of that order.
//
// E = 1e6 is excluded on purpose: this swing strains that rod by 4-10%
// physically, so a 1% tolerance cannot be met at any timestep.
CaseResult runTimestepEnvelope(const std::string& outDir) {
    CaseResult res;
    res.name = "timestep envelope (largest substep keeping strain under 1%)";

    Csv csv(outDir, "timestep_envelope.csv",
            "youngs,segments,substeps,iterations,max_dt,max_substep,max_dt_per_sweep,"
            "elements_per_substep");

    const double v = std::sqrt(2 * 9.81);
    double iterGainMin = 1e9, iterGainMax = 0;
    std::vector<double> motion;
    bool allMeasured = true;

    for (double E : {1e7, 1e8, 1e9}) {
        for (int n : {16, 32, 64}) {
            double perSweepSubsteps = 0;
            for (const auto& split : {std::pair<int, int>(1, 1), std::pair<int, int>(1, 4)}) {
                Probe pr;
                pr.n = n;
                pr.youngs = Real(E);
                pr.substeps = split.first;
                pr.iterations = split.second;
                const double h = largestAccurateSubstep(pr);
                if (h <= 0) allMeasured = false;
                const double dt = h * split.first;
                const double sweeps = double(split.first * split.second);
                const double elementsPerSubstep = v * h * n;
                csv.row(E, n, split.first, split.second, dt, h, dt / sweeps, elementsPerSubstep);

                if (split.second == 1) {
                    motion.push_back(elementsPerSubstep);
                    perSweepSubsteps = dt / sweeps;
                } else {
                    // Four sweeps per frame spent as iterations, against four
                    // single-sweep substeps (same dt per sweep as one substep).
                    const double gain = (dt / sweeps) / perSweepSubsteps;
                    iterGainMin = std::min(iterGainMin, gain);
                    iterGainMax = std::max(iterGainMax, gain);
                }
            }
        }
    }

    res.notes.push_back(fmt("  four sweeps as iterations allow %.3gx to %.3gx the dt of four "
                            "substeps (softer rods favour iterations)", iterGainMin, iterGainMax));
    res.notes.push_back(fmt("  at the limit material moves %.3g to %.3g element lengths per "
                            "substep",
                            *std::min_element(motion.begin(), motion.end()),
                            *std::max_element(motion.begin(), motion.end())));

    res.checks.push_back(
        makeCheck("every configuration bracketed", allMeasured ? 1.0 : 0.0, 1.0, 0.0));
    res.checks.push_back(makeRangeCheck("motion per substep at the limit spreads under 4x",
                                        spread(motion), 1.0, 4.0));
    res.checks.push_back(makeRangeCheck("motion per substep at the limit is a few percent of an "
                                        "element",
                                        *std::max_element(motion.begin(), motion.end()), 0.005,
                                        0.1));
    return res;
}

// ---------------------------------------------------- timestep across motions
namespace {

enum class Motion { kSwing, kDrop, kWhip };

const char* motionName(Motion m) {
    switch (m) {
        case Motion::kSwing: return "swing";
        case Motion::kDrop: return "drop";
        default: return "whip";
    }
}

struct MotionRun {
    double worstStrain = 0;
    double peakSpeed = 0;
};

// One second of a motion at substep h (one sweep per substep), 32 segments.
//   swing  the envelope's cantilever, released from horizontal;
//   drop   a free rod falling 0.5 m, tilted 20 degrees, onto a plane with
//          friction: one end lands first and the rest slaps down after it;
//   whip   a rod pinned at one end, that end shaken vertically at 4 Hz with
//          5 cm amplitude under gravity.
MotionRun runMotion(Motion motion, double youngs, double h) {
    RodMaterial mat = referenceMaterial();
    mat.youngs = Real(youngs);
    const int n = 32;
    CollisionWorld world;
    Rod rod;
    if (motion == Motion::kDrop) {
        const Real tilt = Real(20.0 * kPi / 180.0);
        rod = makeStraightRod(n, Real(1), mat, Vec3(0, 0, Real(0.5)),
                              Vec3(std::cos(tilt), 0, std::sin(tilt)));
        world.primitives.push_back(Primitive::makePlane(Vec3(), Vec3(0, 0, 1), Real(0.3)));
    } else {
        rod = makeStraightRod(n, Real(1), mat, Vec3(0, 0, 0), Vec3(1, 0, 0));
        if (motion == Motion::kSwing)
            clampRootExact(rod);
        else
            rod.pinParticle(0);
    }

    SolverParams p;
    p.dt = Real(h);
    p.substeps = 1;
    p.iterations = 1;
    p.gravity = Vec3(0, 0, Real(-9.81));
    SolverContext ctx;

    const double amplitude = 0.05, omega = 2 * kPi * 4.0;
    MotionRun out;
    const int steps = int(std::ceil(1.0 / h));
    for (int i = 0; i < steps; ++i) {
        if (motion == Motion::kWhip)
            rod.state.kinematicVelocity.front() =
                Vec3(0, 0, Real(amplitude * omega * std::cos(omega * (i + 0.5) * h)));
        step(rod, p, world, ctx);
        for (std::size_t k = 0; k < rod.stretch.size(); ++k) {
            const Vec3 d = rod.state.x[rod.stretch.p1[k]] - rod.state.x[rod.stretch.p0[k]];
            const Vec3 u = rotateInv(rod.state.q[rod.stretch.seg[k]], d) / rod.stretch.restLength[k];
            out.worstStrain = std::max(out.worstStrain, double(norm(u - Vec3(0, 0, 1))));
        }
        for (const Vec3& v : rod.state.v) out.peakSpeed = std::max(out.peakSpeed, double(norm(v)));
        if (!(out.worstStrain < 1.0)) {
            out.worstStrain = 1e9;
            return out;
        }
    }
    return out;
}

}  // namespace

// The envelope measured on one motion, a gravity swing, found the accuracy limit
// is kinematic: a few percent of an element moved per substep. This asks whether
// that holds for motions that are not a swing -- contact (a tilted rod landing
// on the floor) and a whip (one end shaken) -- by finding the largest accurate
// substep for each and stating the limit in each run's own peak particle speed:
// v_peak h / l. The swing's v = sqrt(2 g L) in the envelope is replaced by the
// measured speed, so the three are comparable.
CaseResult runTimestepMotions(const std::string& outDir) {
    CaseResult res;
    res.name = "timestep envelope across motions (swing, drop onto floor, whip)";

    Csv csv(outDir, "timestep_motions.csv",
            "motion,youngs,segments,max_substep,peak_speed,elements_per_substep,worst_strain");
    std::vector<double> motionPerSubstep;
    bool allMeasured = true;
    for (Motion m : {Motion::kSwing, Motion::kDrop, Motion::kWhip}) {
        // An impact at speed v strains a rod by about v / c, c = sqrt(E / rho):
        // 1% for this 3 m/s landing at E = 1e8, over tolerance at any timestep.
        // The drop therefore uses stiffer rods, where that strain is 0.3% and 0.1%.
        const std::vector<double> stiffness =
            m == Motion::kDrop ? std::vector<double>{1e9, 1e10} : std::vector<double>{1e8, 1e9};
        for (double E : stiffness) {
            // Geometric bisection on h, as largestAccurateSubstep.
            double lo = 2e-6, hi = 2e-3;
            auto ok = [&](double h) { return runMotion(m, E, h).worstStrain <= kStrainTol; };
            if (!ok(lo) || ok(hi)) {
                // Say which: a physical strain above tolerance, or no limit found.
                res.notes.push_back(std::string("  ") + motionName(m) +
                                    fmt(", E = %.0e: not bracketed (worst strain %.3g at h = 2e-6 s)",
                                        E, runMotion(m, E, lo).worstStrain));
                allMeasured = false;
                continue;
            }
            while (hi / lo > 1.02) {
                const double mid = std::sqrt(lo * hi);
                (ok(mid) ? lo : hi) = mid;
            }
            const MotionRun run = runMotion(m, E, lo);
            const double elements = run.peakSpeed * lo * 32;
            motionPerSubstep.push_back(elements);
            csv.row(motionName(m), E, 32, lo, run.peakSpeed, elements, run.worstStrain);
            res.notes.push_back(std::string("  ") + motionName(m) +
                                fmt(", E = %.0e: largest accurate substep %.3g s", E, lo) +
                                fmt(", peak speed %.3g m/s, %.3g elements per substep",
                                    run.peakSpeed, elements));
        }
    }
    res.checks.push_back(
        makeCheck("every motion bracketed", allMeasured ? 1.0 : 0.0, 1.0, 0.0));
    if (motionPerSubstep.empty()) return res;
    const double lo = *std::min_element(motionPerSubstep.begin(), motionPerSubstep.end());
    const double hi = *std::max_element(motionPerSubstep.begin(), motionPerSubstep.end());
    res.notes.push_back(fmt("  across motions: %.3g to %.3g elements per substep (%.2gx spread)", lo,
                            hi, spread(motionPerSubstep)));
    // Contact inflates the drop's peak speed well past its 3.1 m/s landing
    // speed; measured against the landing speed its limit is 3-5x tighter.
    res.notes.push_back("  (the drop's peak speed is set by contact resolution, not the fall)");
    res.checks.push_back(
        makeRangeCheck("slowest limit is at least half a percent of an element", lo, 0.005, 0.1));
    res.checks.push_back(makeRangeCheck("fastest limit is under a tenth of an element", hi, 0.005, 0.1));
    return res;
}

// --------------------------------------------------------- timestep convergence
//
// Mesh refinement is established elsewhere; this is its counterpart in time.
// The same swinging cantilever as the envelope probe, simulated for a quarter
// of a second with one sweep per substep, the substep halved from 1/4 ms to
// 1/1024 ms. There is no closed-form trajectory, so convergence is measured
// against itself: d_k is the largest particle distance between the runs at h_k
// and h_k / 2, and the observed order is log2(d_k / d_{k+1}).
//
// Checked: the trajectory converges (every refinement moves it less than the
// last) at a consistent order, and the error that remains at the substep the
// envelope calls accurate is small next to the motion itself.
CaseResult runTimestepConvergence(const std::string& outDir) {
    CaseResult res;
    res.name = "timestep convergence (swinging cantilever, substep halved 8 times)";

    RodMaterial mat = referenceMaterial();
    const int n = 16;
    const double duration = 0.25, dt = 1e-3;

    auto simulate = [&](int substeps) {
        Rod rod = makeStraightRod(n, Real(1), mat, Vec3(0, 0, 0), Vec3(1, 0, 0));
        clampRootExact(rod);
        SolverParams p;
        p.dt = Real(dt);
        p.substeps = substeps;
        p.iterations = 1;
        p.gravity = Vec3(0, 0, Real(-9.81));
        const int steps = int(std::lround(duration / dt));
        for (int i = 0; i < steps; ++i) step(rod, p);
        return rod.state.x;
    };

    std::vector<int> substeps;
    for (int s = 4; s <= 1024; s *= 2) substeps.push_back(s);
    std::vector<std::vector<Vec3>> shapes;
    for (int s : substeps) shapes.push_back(simulate(s));

    Csv csv(outDir, "timestep_convergence.csv", "substeps,substep,difference_to_half,order");
    std::vector<double> diffs, orders;
    for (std::size_t k = 0; k + 1 < shapes.size(); ++k) {
        double d = 0;
        for (std::size_t i = 0; i < shapes[k].size(); ++i)
            d = std::max(d, double(norm(shapes[k][i] - shapes[k + 1][i])));
        diffs.push_back(d);
    }
    for (std::size_t k = 0; k < diffs.size(); ++k) {
        const double order = k + 1 < diffs.size() ? std::log2(diffs[k] / diffs[k + 1]) : 0.0;
        if (k + 1 < diffs.size()) orders.push_back(order);
        csv.row(substeps[k], dt / substeps[k], diffs[k], order);
        res.notes.push_back(fmt("  h = %.3g s: moves %.3g m when halved", dt / substeps[k],
                                diffs[k]) +
                            (k + 1 < diffs.size() ? fmt(" (order %.2f)", order) : ""));
    }

    // At first order the remaining error at h_k is about d_k (the halvings sum
    // geometrically to it). Compare the coarsest run's with how far the rod
    // actually moved, which is what the error is an error in.
    double moved = 0;
    const Rod start = makeStraightRod(n, Real(1), mat, Vec3(0, 0, 0), Vec3(1, 0, 0));
    for (std::size_t i = 0; i < shapes.back().size(); ++i)
        moved = std::max(moved, double(norm(shapes.back()[i] - start.state.x[i])));
    bool shrinking = true;
    for (std::size_t k = 1; k < diffs.size(); ++k) shrinking = shrinking && diffs[k] < diffs[k - 1];
    const double asymptotic = orders.back();
    res.notes.push_back(fmt("  the rod moves %.3g m; error at h = %.3g s is %.3g of that", moved,
                            dt / substeps.front(), diffs.front() / moved));

    res.checks.push_back(makeCheck("every halving moves the trajectory less", shrinking ? 1.0 : 0.0,
                                   1.0, 0.0));
    res.checks.push_back(makeRangeCheck("observed order in the substep (asymptotic)", asymptotic,
                                        0.9, 1.1));
    res.checks.push_back(makeRangeCheck("error at a 1/4 ms substep relative to the motion",
                                        diffs.front() / moved, 0.0, 0.01));
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

}  // namespace crs
