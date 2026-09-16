// Application cases: a real task, swept on the GPU, with its answer checked
// against theory where theory exists and against the CPU reference everywhere
// else.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

#include "../apps/cable_hanging.h"
#include "../core/coloring.h"
#include "../gpu/gpu_solver.h"
#include "cases.h"
#include "support.h"

namespace crs {

#ifdef CRS_WITH_CUDA

namespace {

gpu::BatchParams toBatch(const SolverParams& p) {
    gpu::BatchParams b;
    b.dt = float(p.dt);
    b.substeps = p.substeps;
    b.iterations = p.iterations;
    b.gravityX = float(p.gravity.x);
    b.gravityY = float(p.gravity.y);
    b.gravityZ = float(p.gravity.z);
    return b;
}

// Run one friction value for many placements at once on the GPU, returning
// whether each held. Friction belongs to the world, which a batch shares, so a
// sweep is one batch per friction value.
std::vector<bool> heldOnGpu(const apps::HangingCable& task, Real friction, Real youngs,
                            const std::vector<Real>& ratios) {
    const int rods = static_cast<int>(ratios.size());
    const Rod prototype = task.build(ratios.front(), youngs);
    const CollisionWorld world = task.world(friction);

    gpu::Batch batch;
    if (!batch.create(prototype, rods, gpu::Strategy::kFused, &world)) return {};
    std::vector<apps::HangOutcome> outcomes;
    for (int r = 0; r < rods; ++r) {
        const Rod rod = task.build(ratios[r], youngs);
        batch.uploadRod(r, rod);
        outcomes.emplace_back(rod.state.x.front().z);
    }

    const gpu::BatchParams bp = toBatch(task.params());
    const int particles = task.segments + 1;
    std::vector<Vec3f> positions;
    const int steps = task.steps();
    // The short end is sampled every 50 ms: a slipping cable takes far longer
    // than that to drag its end up by the tolerance.
    for (int s = 1; s <= steps; ++s) {
        batch.step(bp);
        if (s % 50 == 0 || s == steps) {
            batch.downloadPositions(positions);
            for (int r = 0; r < rods; ++r)
                outcomes[r].observe(Real(positions[std::size_t(r) * particles].z));
        }
    }
    std::vector<bool> held;
    for (const auto& o : outcomes) held.push_back(o.held());
    return held;
}

bool heldOnCpu(const apps::HangingCable& task, Real friction, Real youngs, Real ratio) {
    Rod rod = task.build(ratio, youngs);
    const CollisionWorld world = task.world(friction);
    // In the GPU's constraint order, so a disagreement cannot be put down to
    // the ordering alone.
    const Coloring stretch = colorStretchConstraints(rod);
    const Coloring bend = colorBendConstraints(rod);
    SolverParams p = task.params();
    p.stretchColoring = &stretch;
    p.bendColoring = &bend;
    SolverContext ctx;
    apps::HangOutcome outcome(rod.state.x.front().z);
    for (int s = 1; s <= task.steps(); ++s) {
        step(rod, p, world, ctx);
        if (s % 50 == 0) outcome.observe(rod.state.x.front().z);
    }
    return outcome.held();
}

}  // namespace

// ------------------------------------------------------------ cable hanging
//
// Friction x placement, for a soft cable (E = 1 MPa) and a three-times stiffer
// one, every combination simulated for eight seconds after release. Checked:
//
//   * the soft cable's hold/slip boundary follows the capstan prediction
//     mu* = ln(long/short) / pi, on the SAFE side: because this solver's
//     friction creeps (see HangingCable::duration), the simulated cable needs
//     slightly more friction than the ideal rope -- about 0.03 -- and it must
//     never hold where the ideal rope would slip;
//   * the CPU reference, in double precision, reaches the same verdict as the
//     GPU, including on placements close to the boundary -- the ones where a
//     first version, at 16 substeps, got the GPU wrong: float rounding stopped
//     resting cables from ever starting to slide (see HangingCable::params);
//   * gravity's motion per substep stays above float resolution;
//   * the stiff cable holds in fewer configurations than the soft one --
//     reported as a ratio, because that difference is the finding.
CaseResult runCableHanging(const std::string& outDir) {
    CaseResult res;
    res.name = "application: will a cable stay on a hook? (GPU sweep)";
    if (!gpu::cudaAvailable()) {
        res.skipped = true;
        res.skipReason = gpu::diagnostic();
        return res;
    }

    const apps::HangingCable task;
    std::vector<Real> frictions, ratios;
    for (int i = 0; i < 30; ++i) frictions.push_back(Real(0.02) * (i + 1));  // 0.02 .. 0.60
    for (int i = 0; i < 60; ++i) ratios.push_back(Real(1.05) * std::pow(Real(4.0 / 1.05), Real(i) / 59));
    const Real soft = Real(1e6), stiff = Real(3e6);

    Csv csv(outDir, "cable_hanging.csv", "youngs,friction,leg_ratio,capstan_friction,held");

    const auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<double> boundaryErrors;
    double mostOptimistic = -1;  // largest (ideal friction at boundary) - (friction used)
    int softHeld = 0, stiffHeld = 0, total = 0;
    bool allRan = true;
    for (Real youngs : {soft, stiff}) {
        for (Real mu : frictions) {
            const std::vector<bool> held = heldOnGpu(task, mu, youngs, ratios);
            if (held.size() != ratios.size()) {
                allRan = false;
                continue;
            }
            for (std::size_t r = 0; r < ratios.size(); ++r) {
                csv.row(youngs, mu, ratios[r], apps::HangingCable::capstanFriction(ratios[r]),
                        held[r] ? 1 : 0);
                (youngs == soft ? softHeld : stiffHeld) += held[r] ? 1 : 0;
            }
            if (youngs == soft) ++total;
            // Boundary for this friction: the first placement that slides.
            if (youngs == soft) {
                const auto first = std::find(held.begin(), held.end(), false);
                if (first != held.begin() && first != held.end()) {
                    const std::size_t k = std::size_t(first - held.begin());
                    const Real between = std::sqrt(ratios[k - 1] * ratios[k]);
                    const double diff = double(apps::HangingCable::capstanFriction(between) - mu);
                    boundaryErrors.push_back(std::abs(diff));
                    mostOptimistic = std::max(mostOptimistic, diff);
                }
            }
        }
    }
    const double sweepSeconds =
        std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();

    // CPU cross-check: far from the boundary on both sides, plus three slips
    // just below the ideal boundary -- the placements single precision once got
    // wrong. Placements ON the simulated boundary are not compared: there creep
    // makes the verdict hinge on differences far below either precision.
    struct Probe {
        Real mu, ratio;
    };
    const Probe probes[] = {{Real(0.10), Real(2.0)}, {Real(0.45), Real(2.0)},
                            {Real(0.04), Real(1.6)}, {Real(0.30), Real(1.2)},
                            {Real(0.20), Real(2.2)}, {Real(0.40), Real(2.2)},
                            {Real(0.30), Real(2.9)}, {Real(0.10), Real(1.5)}};
    int agree = 0;
    for (const Probe& pr : probes) {
        const bool cpu = heldOnCpu(task, pr.mu, soft, pr.ratio);
        const std::vector<bool> gpu = heldOnGpu(task, pr.mu, soft, {pr.ratio});
        agree += (!gpu.empty() && gpu.front() == cpu) ? 1 : 0;
    }

    std::sort(boundaryErrors.begin(), boundaryErrors.end());
    const double median =
        boundaryErrors.empty() ? 1.0 : boundaryErrors[boundaryErrors.size() / 2];
    const int placements = int(frictions.size() * ratios.size());
    res.notes.push_back(fmt("  %.0f placements x 2 cables simulated for %.0f s each on the GPU in %.1f s",
                            double(placements), double(task.duration), sweepSeconds));
    res.notes.push_back(fmt("  soft cable: boundary within %.3g of the capstan friction (median over "
                            "%.0f friction values)",
                            median, double(boundaryErrors.size())));
    res.notes.push_back(fmt("  holds: soft %.0f, stiff %.0f of %.0f placements", double(softHeld),
                            double(stiffHeld), double(placements)));
    res.notes.push_back(fmt("  CPU and GPU agree on %.0f of %.0f cross-check placements",
                            double(agree), double(sizeof(probes) / sizeof(probes[0]))));

    res.notes.push_back(fmt("  gravity moves a particle %.3gx the float spacing per substep",
                            task.floatMotionMargin()));
    res.checks.push_back(makeCheck("every friction batch ran", allRan ? 1.0 : 0.0, 1.0, 0.0));
    res.checks.push_back(makeRangeCheck("substep motion above float resolution",
                                        task.floatMotionMargin(), 2.0, 1e9));
    res.notes.push_back(fmt("  soft cable: most optimistic boundary point %+.3g against theory "
                            "(positive would mean holding where an ideal rope slips)",
                            mostOptimistic));
    res.checks.push_back(makeRangeCheck("soft cable follows the capstan boundary (median error)",
                                        median, 0.0, 0.06));
    res.checks.push_back(makeRangeCheck("soft cable never holds where an ideal rope slips",
                                        mostOptimistic, -1.0, 0.01));
    res.checks.push_back(makeCheck("CPU and GPU reach the same verdict", double(agree),
                                   double(sizeof(probes) / sizeof(probes[0])), 0.0));
    res.checks.push_back(makeRangeCheck("stiff cable holds in fewer placements",
                                        double(stiffHeld) / std::max(1, softHeld), 0.0, 0.999));
    (void)total;
    return res;
}

#else

CaseResult runCableHanging(const std::string&) {
    CaseResult res;
    res.name = "application: will a cable stay on a hook? (built without CUDA)";
    res.skipped = true;
    res.skipReason = "the sweep runs on the GPU; this build has no CUDA support";
    return res;
}

#endif

}  // namespace crs
