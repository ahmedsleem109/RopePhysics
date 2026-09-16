// GPU validation and measurement cases.
//
// The Phase 3 gate is that the GPU reproduces the CPU. Making that a number
// rather than a feeling needs one piece of care: a colored sweep is not the
// same iteration as a sequential one. Red-black Gauss-Seidel and sequential
// Gauss-Seidel converge to the same fixed point but take different paths, so a
// correct GPU port will *not* match a sequentially-swept CPU trajectory, and
// chasing that difference is a good way to waste a week. The parity case
// therefore runs the CPU in the GPU's own colour order, and separately reports
// how much the ordering alone costs, so the two effects are never confused.
#include <algorithm>
#include <chrono>
#include <cmath>

#include "../core/coloring.h"
#include "../core/solver.h"
#include "../validation/cases.h"
#include "../validation/support.h"
#include "gpu_solver.h"

namespace crs {

namespace {

// A rod that is representative of what the batch cases run: long enough to be
// interesting, short enough to fit in shared memory.
Rod makeBenchmarkRod(int segments) {
    RodMaterial mat = referenceMaterial();
    Rod rod = makeStraightRod(segments, Real(1), mat, Vec3(0, 0, 0), Vec3(1, 0, 0));
    clampRootExact(rod);
    return rod;
}

// Only the GPU cases compare trajectories and build batches.
#ifdef CRS_WITH_CUDA
double maxPositionDifference(const Rod& a, const Rod& b) {
    double worst = 0;
    for (std::size_t i = 0; i < a.state.numParticles(); ++i)
        worst = std::max(worst, double(norm(a.state.x[i] - b.state.x[i])));
    return worst;
}

double maxOrientationDifference(const Rod& a, const Rod& b) {
    double worst = 0;
    for (std::size_t j = 0; j < a.state.numSegments(); ++j) {
        Quat qb = sameHemisphere(b.state.q[j], a.state.q[j]);
        const Quat d(a.state.q[j].w - qb.w, a.state.q[j].x - qb.x, a.state.q[j].y - qb.y,
                     a.state.q[j].z - qb.z);
        worst = std::max(worst, std::sqrt(double(dot(d, d))));
    }
    return worst;
}

SolverParams benchmarkParams() {
    SolverParams p;
    p.dt = Real(1e-3);
    p.substeps = 8;
    p.iterations = 1;
    p.gravity = Vec3(0, 0, Real(-9.81));
    return p;
}

gpu::BatchParams toBatchParams(const SolverParams& p) {
    gpu::BatchParams b;
    b.dt = float(p.dt);
    b.substeps = p.substeps;
    b.iterations = p.iterations;
    b.gravityX = float(p.gravity.x);
    b.gravityY = float(p.gravity.y);
    b.gravityZ = float(p.gravity.z);
    b.linearDamping = float(p.linearDamping);
    b.angularDamping = float(p.angularDamping);
    return b;
}

#endif  // CRS_WITH_CUDA

}  // namespace

// ---------------------------------------------------------------- coloring
//
// The coloring is the thing that makes the GPU solver correct, so it gets its
// own case rather than being assumed. Two claims: the coloring is valid (no
// colour contains two constraints that touch the same state), and a rod needs
// exactly two colours per family -- which is what makes a chain such a good fit
// for this scheme in the first place.
CaseResult runColoring(const std::string& outDir) {
    CaseResult res;
    res.name = "constraint graph coloring";

    Csv csv(outDir, "coloring.csv", "segments,stretch_colors,bend_colors,stretch_valid,bend_valid");

    bool allValid = true;
    int worstStretch = 0, worstBend = 0;
    for (int n : {8, 32, 128, 512}) {
        Rod rod = makeBenchmarkRod(n);
        const Coloring sc = colorStretchConstraints(rod);
        const Coloring bc = colorBendConstraints(rod);

        std::vector<std::vector<int>> sres(rod.stretch.size()), bres(rod.bend.size());
        const int frameBase = static_cast<int>(rod.state.numParticles());
        for (std::size_t k = 0; k < rod.stretch.size(); ++k)
            sres[k] = {rod.stretch.p0[k], rod.stretch.p1[k], frameBase + rod.stretch.seg[k]};
        for (std::size_t k = 0; k < rod.bend.size(); ++k)
            bres[k] = {frameBase + rod.bend.segA[k], frameBase + rod.bend.segB[k]};

        const bool sok = verifyColoring(sc, sres), bok = verifyColoring(bc, bres);
        allValid = allValid && sok && bok;
        worstStretch = std::max(worstStretch, sc.numColors());
        worstBend = std::max(worstBend, bc.numColors());
        csv.row(n, sc.numColors(), bc.numColors(), sok ? 1 : 0, bok ? 1 : 0);
    }

    res.notes.push_back(fmt("  %.0f stretch colours and %.0f bend colours at every resolution",
                            double(worstStretch), double(worstBend)));
    res.checks.push_back(makeCheck("coloring is conflict-free", allValid ? 1.0 : 0.0, 1.0, 0.0));
    res.checks.push_back(makeCheck("stretch colours", worstStretch, 2.0, 0.0));
    res.checks.push_back(makeCheck("bend colours", worstBend, 2.0, 0.0));
    return res;
}

#ifdef CRS_WITH_CUDA

// ---------------------------------------------------------------- parity
CaseResult runGpuParity(const std::string& outDir) {
    CaseResult res;
    res.name = "GPU vs CPU trajectory parity";

    if (!gpu::cudaAvailable()) {
        res.skipped = true;
        res.skipReason = gpu::diagnostic();
        return res;
    }
    res.notes.push_back("  device: " + gpu::deviceName());

    // What "matches" can mean. The GPU runs single precision against a
    // double-precision reference, and this trajectory (a gravity-released
    // cantilever, one sweep per substep) amplifies rounding: the SAME colour-
    // ordered CPU code built with float instead of double drifts from the
    // double build by 5e-9 m after one step, 3e-7 m after ten and 2.5e-4 m
    // after two hundred, at every mesh (measured with CRS_REAL_FLOAT). A single
    // tolerance at step 200 cannot tell a kernel bug from that. So parity is
    // checked at several horizons: tightly after one step, where rounding has
    // had no time to grow and any kernel bug shows at full size, and against
    // the measured float drift at step 200. The two strategies run the same
    // kernel code in a different launch structure and must agree exactly.
    Csv csv(outDir, "gpu_parity.csv",
            "segments,steps,strategy,max_dx,max_dq,ordering_only_dx");

    const std::vector<int> checkpoints = {1, 10, 50, 200};
    double worstFirst = 0, worstFinal = 0, strategyGap = 0;
    for (int n : {16, 48, 128}) {
        const Rod prototype = makeBenchmarkRod(n);
        const SolverParams p = benchmarkParams();

        const Coloring sc = colorStretchConstraints(prototype);
        const Coloring bc = colorBendConstraints(prototype);
        SolverParams pc = p;
        pc.stretchColoring = &sc;
        pc.bendColoring = &bc;

        // CPU in the GPU's order, and in index order to size the ordering effect.
        Rod cpuColored = prototype, cpuSequential = prototype;

        gpu::Batch multi, fused;
        const bool haveMulti = multi.create(prototype, 1, gpu::Strategy::kMultiKernel);
        const bool haveFused = fused.create(prototype, 1, gpu::Strategy::kFused);
        if (!haveFused) res.notes.push_back(fmt("  n=%.0f does not fit the fused path", double(n)));
        const gpu::BatchParams bp = toBatchParams(p);

        int done = 0;
        for (int cp : checkpoints) {
            for (; done < cp; ++done) {
                step(cpuColored, pc);
                step(cpuSequential, p);
                if (haveMulti) multi.step(bp);
                if (haveFused) fused.step(bp);
            }
            const double orderingOnly = maxPositionDifference(cpuColored, cpuSequential);
            Rod fromMulti = prototype, fromFused = prototype;
            if (haveMulti) {
                multi.synchronize();
                multi.download(0, fromMulti);
            }
            if (haveFused) {
                fused.synchronize();
                fused.download(0, fromFused);
            }
            if (haveMulti && haveFused)
                strategyGap = std::max(strategyGap, maxPositionDifference(fromMulti, fromFused));

            for (int s = 0; s < 2; ++s) {
                if (!(s == 0 ? haveMulti : haveFused)) continue;
                const Rod& g = s == 0 ? fromMulti : fromFused;
                const double dx = maxPositionDifference(cpuColored, g);
                const double dq = maxOrientationDifference(cpuColored, g);
                if (cp == checkpoints.front()) worstFirst = std::max(worstFirst, dx);
                if (cp == checkpoints.back()) worstFinal = std::max(worstFinal, dx);
                csv.row(n, cp, s == 0 ? "multikernel" : "fused", dx, dq, orderingOnly);
            }
            if (cp == checkpoints.back())
                res.notes.push_back(fmt("  n=%.0f: colour ordering alone moves the CPU trajectory "
                                        "by %.3g m over %.0f steps",
                                        double(n), orderingOnly, double(cp)));
        }
    }

    res.notes.push_back(fmt("  GPU-CPU position difference: %.3g m after 1 step, %.3g m after 200 "
                            "(float-vs-double CPU alone: 5e-9 and 2.5e-4)",
                            worstFirst, worstFinal));
    res.checks.push_back(makeCheck("GPU matches CPU after one step", worstFirst, 0.0, 1e-7, 1.0));
    res.checks.push_back(
        makeCheck("GPU drift at 200 steps within float rounding", worstFinal, 0.0, 1e-3, 1.0));
    res.checks.push_back(
        makeCheck("multi-kernel and fused agree exactly", strategyGap, 0.0, 0.0, 1.0));
    return res;
}

// ---------------------------------------------------------------- determinism
//
// Run-to-run bitwise determinism. A solver whose answer depends on scheduling
// cannot be debugged, cannot be regression-tested and cannot be shipped, and
// atomics in a constraint solver are the usual way that property is lost. The
// coloring is what buys it here: every write in a colour is to state no other
// thread in that colour touches, so there is nothing for the scheduler to
// reorder.
CaseResult runGpuDeterminism(const std::string& outDir) {
    CaseResult res;
    res.name = "GPU bitwise determinism";

    if (!gpu::cudaAvailable()) {
        res.skipped = true;
        res.skipReason = gpu::diagnostic();
        return res;
    }

    Csv csv(outDir, "gpu_determinism.csv", "segments,rods,strategy,runs,identical,max_diff");

    bool allIdentical = true;
    for (int n : {32, 128}) {
        for (gpu::Strategy strat : {gpu::Strategy::kMultiKernel, gpu::Strategy::kFused}) {
            const Rod prototype = makeBenchmarkRod(n);
            const gpu::BatchParams bp = toBatchParams(benchmarkParams());
            const int rods = 64, steps = 100, runs = 3;

            std::vector<Rod> results;
            bool created = true;
            for (int r = 0; r < runs; ++r) {
                gpu::Batch batch;
                if (!batch.create(prototype, rods, strat)) {
                    created = false;
                    break;
                }
                for (int i = 0; i < steps; ++i) batch.step(bp);
                batch.synchronize();
                Rod out = prototype;
                batch.download(rods / 2, out);
                results.push_back(out);
            }
            if (!created) continue;

            double maxDiff = 0;
            for (std::size_t r = 1; r < results.size(); ++r)
                maxDiff = std::max(maxDiff, maxPositionDifference(results[0], results[r]));
            const bool identical = maxDiff == 0.0;
            allIdentical = allIdentical && identical;
            csv.row(n, rods, strat == gpu::Strategy::kFused ? "fused" : "multikernel", runs,
                    identical ? 1 : 0, maxDiff);
        }
    }

    res.notes.push_back("  same input, repeated runs, compared bit for bit");
    res.checks.push_back(makeCheck("bitwise identical across runs", allIdentical ? 1.0 : 0.0, 1.0,
                                   0.0));
    return res;
}

// ---------------------------------------------------------------- throughput
//
// Phase 4's headline measurement: rod-segment-steps per second, not frames per
// second of one pretty scene. Reported against batch size, rod length and
// substep count so the knee in each curve can be found and explained.
CaseResult runGpuThroughput(const std::string& outDir) {
    CaseResult res;
    res.name = "GPU throughput and scaling";

    if (!gpu::cudaAvailable()) {
        res.skipped = true;
        res.skipReason = gpu::diagnostic();
        return res;
    }
    res.notes.push_back("  device: " + gpu::deviceName());
    res.notes.push_back(fmt("  fused path fits rods up to %.0f segments in shared memory",
                            double(gpu::maxFusedSegments())));

    Csv csv(outDir, "gpu_throughput.csv",
            "strategy,rods,segments,substeps,steps,seconds,segment_steps_per_sec,launches_per_step");

    auto measure = [&](gpu::Strategy strat, int rods, int n, int substeps) -> double {
        const Rod prototype = makeBenchmarkRod(n);
        gpu::Batch batch;
        if (!batch.create(prototype, rods, strat)) return 0;

        gpu::BatchParams bp = toBatchParams(benchmarkParams());
        bp.substeps = substeps;

        // Warm up: first launch pays context and module load.
        for (int i = 0; i < 5; ++i) batch.step(bp);
        batch.synchronize();

        const int steps = 200;
        const auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < steps; ++i) batch.step(bp);
        batch.synchronize();
        const auto t1 = std::chrono::high_resolution_clock::now();

        const double seconds = std::chrono::duration<double>(t1 - t0).count();
        const double segmentSteps = double(rods) * n * steps * substeps;
        const double rate = segmentSteps / seconds;
        csv.row(strat == gpu::Strategy::kFused ? "fused" : "multikernel", rods, n, substeps, steps,
                seconds, rate, batch.launchesPerStep(bp));
        return rate;
    };

    double best = 0;
    // Batch-size sweep at a fixed rod length: where does the GPU fill up?
    for (int rods : {1, 4, 16, 64, 256, 1024, 4096, 16384}) {
        for (gpu::Strategy strat : {gpu::Strategy::kMultiKernel, gpu::Strategy::kFused})
            best = std::max(best, measure(strat, rods, 64, 8));
    }
    // Rod-length sweep at a fixed batch.
    for (int n : {16, 32, 64, 128, 256})
        for (gpu::Strategy strat : {gpu::Strategy::kMultiKernel, gpu::Strategy::kFused})
            best = std::max(best, measure(strat, 2048, n, 8));
    // Substep sweep.
    for (int sub : {1, 2, 4, 8, 16, 32})
        for (gpu::Strategy strat : {gpu::Strategy::kMultiKernel, gpu::Strategy::kFused})
            best = std::max(best, measure(strat, 2048, 64, sub));

    res.notes.push_back(fmt("  peak %.4g million segment-substeps/sec", best / 1e6));
    res.checks.push_back(makeCheck("throughput measured", best > 0 ? 1.0 : 0.0, 1.0, 0.0));
    return res;
}

#else  // !CRS_WITH_CUDA

CaseResult runGpuParity(const std::string&) {
    CaseResult res;
    res.name = "GPU vs CPU trajectory parity (built without CUDA)";
    return res;
}
CaseResult runGpuDeterminism(const std::string&) {
    CaseResult res;
    res.name = "GPU bitwise determinism (built without CUDA)";
    return res;
}
CaseResult runGpuThroughput(const std::string&) {
    CaseResult res;
    res.name = "GPU throughput (built without CUDA)";
    return res;
}

#endif

}  // namespace crs
