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
    //
    // Three scenarios. "gravity" is the benchmark rod swinging under its own
    // weight. "loaded" adds a force on the tip particle, a torque on the middle
    // segment, the pinned root particle moving at a prescribed velocity (how a
    // gripper drives a rod), and the root's fixed frame driven round the rod
    // axis a little every step (how a twist is imposed), uploaded to the device
    // through Batch::setLoads. "contact" lets the rod settle onto a floor 0.1 mm
    // below it and onto a sphere under its middle, both with friction; from
    // about step 5 it rests in 6-11 persistent contacts. (A first version
    // started the rod 1 mm INSIDE the floor: the first position correction
    // launched it upward at 8 m/s and it barely touched anything again.)
    Csv csv(outDir, "gpu_parity.csv",
            "scenario,segments,steps,strategy,max_dx,max_dq,ordering_only_dx");

    struct Scenario {
        const char* name;
        bool loaded;
        bool contact;
    };
    const Scenario scenarios[] = {
        {"gravity", false, false}, {"loaded", true, false}, {"contact", false, true}};
    const std::vector<int> checkpoints = {1, 10, 50, 200};
    double worstFirst = 0, worstFinal = 0, strategyGap = 0;
    std::size_t peakCpuContacts = 0;

    CollisionWorld world;
    world.primitives.push_back(Primitive::makePlane(Vec3(0, 0, Real(-0.0051)), Vec3(0, 0, 1),
                                                    Real(0.5)));
    world.primitives.push_back(Primitive::makeSphere(Vec3(Real(0.5), 0, Real(-0.0851)),
                                                     Real(0.08), Real(0.3)));
    // For sizing what contact changes: the gravity run of each mesh at step 50.
    std::vector<std::pair<int, Rod>> gravityAt50;
    double contactEffect = 1e9, contactErrorAt50 = 0;

    for (const Scenario& sc : scenarios) {
        for (int n : {16, 48, 128}) {
            Rod prototype = makeBenchmarkRod(n);
            const int rootFrame = static_cast<int>(prototype.state.numSegments()) - 1;
            const Quat rootRest = prototype.state.q[rootFrame];
            if (sc.loaded) {
                prototype.state.extForce.back() = Vec3(0, Real(0.02), Real(0.05));
                prototype.state.extTorque[n / 2] = Vec3(Real(2e-3), 0, Real(-1e-3));
                prototype.state.kinematicVelocity[0] = Vec3(0, Real(0.05), Real(0.1));  // moving root
            }
            // Twist the root by 0.5 rad per second of simulated time.
            auto drive = [&](Rod& rod, int stepIndex) {
                if (!sc.loaded) return;
                const Real angle = Real(0.5e-3) * Real(stepIndex + 1);
                rod.state.q[rootFrame] = normalize(quatAxisAngle(Vec3(1, 0, 0), angle) * rootRest);
            };
            const SolverParams p = benchmarkParams();

            const Coloring stretchColors = colorStretchConstraints(prototype);
            const Coloring bendColors = colorBendConstraints(prototype);
            SolverParams pc = p;
            pc.stretchColoring = &stretchColors;
            pc.bendColoring = &bendColors;

            // CPU in the GPU's order, and in index order to size the ordering effect.
            Rod cpuColored = prototype, cpuSequential = prototype;
            SolverContext ctxColored, ctxSequential;
            const CollisionWorld* gpuWorld = sc.contact ? &world : nullptr;
            auto cpuStep = [&](Rod& rod, const SolverParams& params, SolverContext& ctx) {
                if (sc.contact)
                    step(rod, params, world, ctx);
                else
                    step(rod, params);
            };

            gpu::Batch multi, fused;
            const bool haveMulti =
                multi.create(prototype, 1, gpu::Strategy::kMultiKernel, gpuWorld);
            const bool haveFused = fused.create(prototype, 1, gpu::Strategy::kFused, gpuWorld);
            if (!haveFused)
                res.notes.push_back(fmt("  n=%.0f does not fit the fused path", double(n)));
            const gpu::BatchParams bp = toBatchParams(p);

            int done = 0;
            double finalDx = 0, finalOrdering = 0;
            for (int cp : checkpoints) {
                for (; done < cp; ++done) {
                    drive(cpuColored, done);
                    drive(cpuSequential, done);
                    if (sc.loaded) {
                        if (haveMulti) multi.setLoads(0, cpuColored);
                        if (haveFused) fused.setLoads(0, cpuColored);
                    }
                    cpuStep(cpuColored, pc, ctxColored);
                    cpuStep(cpuSequential, p, ctxSequential);
                    peakCpuContacts = std::max(peakCpuContacts, ctxColored.contacts.size());
                    if (haveMulti) multi.step(bp);
                    if (haveFused) fused.step(bp);
                }
                const double orderingOnly = maxPositionDifference(cpuColored, cpuSequential);
                if (cp == 50 && !sc.loaded && !sc.contact) gravityAt50.emplace_back(n, cpuColored);
                if (cp == 50 && sc.contact)
                    for (const auto& g : gravityAt50)
                        if (g.first == n)
                            contactEffect = std::min(contactEffect,
                                                     maxPositionDifference(g.second, cpuColored));
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
                    if (cp == 50 && sc.contact) contactErrorAt50 = std::max(contactErrorAt50, dx);
                    if (cp == checkpoints.back()) {
                        worstFinal = std::max(worstFinal, dx);
                        finalDx = std::max(finalDx, dx);
                    }
                    csv.row(sc.name, n, cp, s == 0 ? "multikernel" : "fused", dx, dq, orderingOnly);
                }
                if (cp == checkpoints.back()) finalOrdering = orderingOnly;
            }
            res.notes.push_back(std::string("  ") + sc.name +
                                fmt(", n=%.0f: GPU-CPU %.3g m after 200 steps (colour ordering "
                                    "alone moves the CPU %.3g m)",
                                    double(n), finalDx, finalOrdering));
        }
    }

    res.notes.push_back(fmt("  GPU-CPU position difference: %.3g m after 1 step, %.3g m after 200 "
                            "(float-vs-double CPU alone: 5e-9 and 2.5e-4)",
                            worstFirst, worstFinal));
    res.checks.push_back(makeCheck("GPU matches CPU after one step", worstFirst, 0.0, 1e-7, 1.0));

    // The one-step check only means something if one step is enough for the
    // loads to matter: a device that ignored them must fail it. Size what the
    // loads change in one step on the CPU and require the GPU error to be far
    // below that.
    {
        Rod plain = makeBenchmarkRod(16), loaded = makeBenchmarkRod(16);
        loaded.state.extForce.back() = Vec3(0, Real(0.02), Real(0.05));
        loaded.state.extTorque[8] = Vec3(Real(2e-3), 0, Real(-1e-3));
        loaded.state.kinematicVelocity[0] = Vec3(0, Real(0.05), Real(0.1));
        step(plain, benchmarkParams());
        step(loaded, benchmarkParams());
        const double loadEffect = maxPositionDifference(plain, loaded);
        res.notes.push_back(fmt("  one step of the loads moves the rod %.3g m", loadEffect));
        res.checks.push_back(makeRangeCheck("one-step GPU error under 1% of the load effect",
                                            worstFirst / loadEffect, 0.0, 0.01));
    }
    res.checks.push_back(
        makeCheck("GPU drift at 200 steps within float rounding", worstFinal, 0.0, 1e-3, 1.0));
    res.checks.push_back(
        makeCheck("multi-kernel and fused agree exactly", strategyGap, 0.0, 0.0, 1.0));
    res.notes.push_back(fmt("  contact scenario: up to %.0f simultaneous contacts on the CPU",
                            double(peakCpuContacts)));
    res.checks.push_back(makeRangeCheck("contact scenario actually made contact",
                                        double(peakCpuContacts), 5.0, 1e9));
    res.notes.push_back(fmt("  by step 50 contact moves the rod %.3g m; GPU-CPU there %.3g m",
                            contactEffect, contactErrorAt50));
    res.checks.push_back(makeRangeCheck("GPU contact error under 1% of the contact effect",
                                        contactErrorAt50 / contactEffect, 0.0, 0.01));

    // ---- self-collision -------------------------------------------------------
    //
    // A rope dropped end-first onto the floor coils onto itself. Coiling is
    // chaotic, so a long trajectory cannot be compared to float precision; a
    // restart can. The rope is simulated on the CPU, in the GPU's colour order,
    // until it holds at least five self contacts. That state is uploaded to both
    // GPU strategies and all three take ONE step.
    //
    // One step here is 16 substeps x 4 sweeps of non-smooth contact, and float
    // rounding is correspondingly larger than in the benchmark: the same CPU
    // code built with CRS_REAL_FLOAT, stepped once from the identical state,
    // lands 1.8e-5 m from the double build (and even finds 6 self contacts at
    // the last substep instead of 8). The tolerance is 5e-5 m, about three times
    // that. For the check to mean anything, self-collision itself must change the
    // step by much more than the tolerance; that is asserted too, by stepping the
    // same state on the CPU without self-collision.
    {
        RodMaterial mat = referenceMaterial();
        mat.youngs = Real(2e6);
        // The configuration of the CPU self-collision case, which coils reliably.
        // Segments are shorter than the diameter, so the index gap is 3 and only
        // genuine coiling contacts count. (With segments barely longer than the
        // diameter, strands two apart start AT the contact threshold, where float
        // and double disagree about whether a contact exists at all.)
        const int n = 150;
        const Real L = Real(1.2);
        Rod rope = makeStraightRod(n, L, mat, Vec3(0, 0, Real(0.02)),
                                   normalize(Vec3(Real(0.08), Real(0.03), 1)));
        for (Vec3& x : rope.state.x) x.z = L + Real(0.05) - x.z;
        rope.state.xPrev = rope.state.x;
        setRestFromCurrent(rope);

        CollisionWorld selfWorld;
        selfWorld.primitives.push_back(
            Primitive::makePlane(Vec3(0, 0, 0), Vec3(0, 0, 1), Real(0.6)));
        selfWorld.selfCollision = true;
        selfWorld.selfFriction = Real(0.3);
        selfWorld.hashTableSize = 256;
        CollisionWorld noSelf = selfWorld;
        noSelf.selfCollision = false;

        SolverParams sp;
        sp.dt = Real(1e-3);
        sp.substeps = 16;
        sp.iterations = 4;
        sp.gravity = Vec3(0, 0, Real(-9.81));
        sp.linearDamping = sp.angularDamping = Real(1);
        const Coloring ropeStretch = colorStretchConstraints(rope);
        const Coloring ropeBend = colorBendConstraints(rope);
        sp.stretchColoring = &ropeStretch;
        sp.bendColoring = &ropeBend;

        auto selfContactsIn = [](const SolverContext& c) {
            int k = 0;
            for (const Contact& ct : c.contacts.contacts) k += ct.count == 4 ? 1 : 0;
            return k;
        };
        SolverContext ctx;
        int presteps = 0;
        while (presteps < 3000 && selfContactsIn(ctx) < 5) {
            step(rope, sp, selfWorld, ctx);
            ++presteps;
        }
        const int selfAtRestart = selfContactsIn(ctx);

        Rod cpuWith = rope, cpuWithout = rope;
        SolverContext ctxWith, ctxWithout;
        step(cpuWith, sp, selfWorld, ctxWith);
        step(cpuWithout, sp, noSelf, ctxWithout);
        const double selfEffect = maxPositionDifference(cpuWith, cpuWithout);

        const gpu::BatchParams sbp = toBatchParams(sp);
        double selfError = 0, selfStrategyGap = 0, drift50 = 0;
        long long overflow = 0;
        Rod fromMulti = rope, fromFused = rope;
        gpu::Batch multi, fused;
        const bool haveMulti = multi.create(rope, 1, gpu::Strategy::kMultiKernel, &selfWorld);
        const bool haveFused = fused.create(rope, 1, gpu::Strategy::kFused, &selfWorld);
        if (haveMulti) {
            multi.step(sbp);
            multi.synchronize();
            multi.download(0, fromMulti);
            selfError = std::max(selfError, maxPositionDifference(cpuWith, fromMulti));
        }
        if (haveFused) {
            fused.step(sbp);
            fused.synchronize();
            fused.download(0, fromFused);
            selfError = std::max(selfError, maxPositionDifference(cpuWith, fromFused));
        }
        if (haveMulti && haveFused) selfStrategyGap = maxPositionDifference(fromMulti, fromFused);

        // Then 50 more steps, reported only: coiling amplifies rounding.
        if (haveFused) {
            Rod cpu = cpuWith;
            SolverContext c2 = ctxWith;
            for (int i = 0; i < 50; ++i) {
                step(cpu, sp, selfWorld, c2);
                fused.step(sbp);
            }
            fused.synchronize();
            fused.download(0, fromFused);
            drift50 = maxPositionDifference(cpu, fromFused);
            overflow += fused.selfContactOverflow();
        }
        if (haveMulti) overflow += multi.selfContactOverflow();

        res.notes.push_back(fmt("  self-collision: restart after %.0f steps with %.0f self contacts; "
                                "one step of self-collision moves the rope %.3g m",
                                double(presteps), double(selfAtRestart), selfEffect));
        res.notes.push_back(fmt("  self-collision: GPU-CPU %.3g m after one step, %.3g m 50 steps "
                                "later", selfError, drift50));
        res.checks.push_back(makeRangeCheck("self-collision restart made contact",
                                            double(selfAtRestart), 5.0, 1e9));
        const double selfTol = 5e-5;
        res.checks.push_back(makeCheck("self-collision: GPU matches CPU after one step", selfError,
                                       0.0, selfTol, 1.0));
        res.checks.push_back(makeRangeCheck("self-collision effect is 10x the tolerance",
                                            selfEffect / selfTol, 10.0, 1e12));
        res.checks.push_back(makeCheck("self-collision: strategies agree exactly", selfStrategyGap,
                                       0.0, 0.0, 1.0));
        res.checks.push_back(
            makeCheck("self-collision: no contact overflow", double(overflow), 0.0, 0.0, 1.0));
    }
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
            "strategy,rods,segments,substeps,steps,seconds,segment_steps_per_sec,launches_per_step,"
            "primitives,self_collision");

    // The contact workload: a floor and a sphere under every rod, so every
    // particle tests two primitives each substep and projects its contacts
    // each iteration.
    CollisionWorld world;
    world.primitives.push_back(Primitive::makePlane(Vec3(0, 0, Real(-0.0051)), Vec3(0, 0, 1),
                                                    Real(0.5)));
    world.primitives.push_back(Primitive::makeSphere(Vec3(Real(0.5), 0, Real(-0.0851)),
                                                     Real(0.08), Real(0.3)));

    CollisionWorld selfWorld = world;
    selfWorld.selfCollision = true;
    selfWorld.selfFriction = Real(0.3);
    selfWorld.hashTableSize = 256;

    enum class Load { kNone, kContacts, kSelf };
    auto measure = [&](gpu::Strategy strat, int rods, int n, int substeps,
                       Load load = Load::kNone) -> double {
        const Rod prototype = makeBenchmarkRod(n);
        gpu::Batch batch;
        const CollisionWorld* w =
            load == Load::kNone ? nullptr : (load == Load::kSelf ? &selfWorld : &world);
        if (!batch.create(prototype, rods, strat, w)) return 0;

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
                seconds, rate, batch.launchesPerStep(bp), load == Load::kNone ? 0 : 2,
                load == Load::kSelf ? 1 : 0);
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
    // With contacts against two primitives, at the batch-size knee and beyond.
    double contactRate = 0, plainRate = 0, selfRate = 0;
    for (int rods : {1024, 16384})
        for (gpu::Strategy strat : {gpu::Strategy::kMultiKernel, gpu::Strategy::kFused}) {
            const double r = measure(strat, rods, 64, 8, Load::kContacts);
            if (rods == 16384 && strat == gpu::Strategy::kFused) contactRate = r;
            const double rs = measure(strat, rods, 64, 8, Load::kSelf);
            if (rods == 16384 && strat == gpu::Strategy::kFused) selfRate = rs;
        }
    plainRate = measure(gpu::Strategy::kFused, 16384, 64, 8);
    res.notes.push_back(fmt("  contacts against 2 primitives cost %.3gx throughput (fused, "
                            "16384 x 64): %.4g M vs %.4g M segment-substeps/s",
                            plainRate > 0 ? contactRate / plainRate : 0.0, contactRate / 1e6,
                            plainRate / 1e6));
    res.notes.push_back(fmt("  adding self-collision: %.4g M segment-substeps/s (%.3gx of plain)",
                            selfRate / 1e6, plainRate > 0 ? selfRate / plainRate : 0.0));
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
