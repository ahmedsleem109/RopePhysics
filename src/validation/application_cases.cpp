// Application cases: a real task, swept on the GPU, with its answer checked
// against theory where theory exists and against the CPU reference everywhere
// else.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "../apps/cable_hanging.h"
#include "../apps/harness_routing.h"
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

// ----------------------------------------------------------- harness routing

namespace {

using Harness = apps::HarnessTask;

// One try at the harness: the robot's motion, and the cable it happens to get.
struct Attempt {
    Harness::Policy policy;
    float youngsScale = 1;
    float frictionScale = 1;
};

// Every attempt in one batch on the GPU. Each rod's gripper follows its own
// policy; velocities are uploaded once per control tick. Returns one outcome
// per attempt (empty if the batch cannot run) and, if asked, every rod's final
// positions, rod-major.
std::vector<Harness::Outcome> attemptsOnGpu(const Harness& task,
                                            const std::vector<Attempt>& attempts,
                                            std::vector<Vec3f>* finalPositions = nullptr) {
    const int rods = static_cast<int>(attempts.size());
    const CollisionWorld world = task.world();
    gpu::Batch batch;
    if (!batch.create(task.build(), rods, gpu::Strategy::kFused, &world)) return {};
    std::vector<float> youngs, friction;
    for (const Attempt& a : attempts) {
        youngs.push_back(a.youngsScale);
        friction.push_back(a.frictionScale);
    }
    batch.setMaterialScales(youngs, friction);

    const gpu::BatchParams bp = toBatch(task.params());
    const std::size_t particles = std::size_t(task.segments) + 1;
    std::vector<Vec3f> velocities(particles * rods, Vec3f(0, 0, 0));
    const int steps = task.steps();
    for (int s = 0; s < steps; ++s) {
        if (s % task.stepsPerControl == 0) {
            for (int r = 0; r < rods; ++r) {
                const Vec3 v = task.gripperVelocity(attempts[r].policy, s);
                velocities[std::size_t(r) * particles + particles - 1] =
                    Vec3f(float(v.x), float(v.y), float(v.z));
            }
            batch.setKinematicVelocities(velocities);
        }
        batch.step(bp);
    }

    std::vector<Vec3f> positions;
    batch.downloadPositions(positions);
    std::vector<Harness::Outcome> outcomes;
    std::vector<Vec3> cable(particles);
    for (int r = 0; r < rods; ++r) {
        for (std::size_t i = 0; i < particles; ++i) {
            const Vec3f& p = positions[std::size_t(r) * particles + i];
            cable[i] = Vec3(Real(p.x), Real(p.y), Real(p.z));
        }
        outcomes.push_back(task.evaluate(cable));
    }
    if (finalPositions) *finalPositions = std::move(positions);
    return outcomes;
}

Harness::Outcome attemptOnCpu(const Harness& task, const Attempt& attempt) {
    Rod rod = task.build(Real(attempt.youngsScale));
    const CollisionWorld world = task.world(Real(attempt.frictionScale));
    const Coloring stretch = colorStretchConstraints(rod);
    const Coloring bend = colorBendConstraints(rod);
    SolverParams p = task.params();
    p.stretchColoring = &stretch;
    p.bendColoring = &bend;
    SolverContext ctx;
    for (int s = 0; s < task.steps(); ++s) {
        rod.state.kinematicVelocity.back() = task.gripperVelocity(attempt.policy, s);
        step(rod, p, world, ctx);
    }
    return task.evaluate(rod.state.x);
}

// A policy as the 10 numbers the learner adjusts, and back. The gripper cannot
// take the cable end further from the connector than the cable reaches.
std::vector<double> toParams(const Harness::Policy& policy) {
    std::vector<double> v;
    for (const Vec3& w : policy) {
        v.push_back(double(w.x));
        v.push_back(double(w.y));
    }
    return v;
}

Harness::Policy toPolicy(const Harness& task, const std::vector<double>& v) {
    Harness::Policy policy;
    for (int k = 0; k < Harness::kWaypoints; ++k) {
        Vec3 w(Real(v[2 * k]), Real(v[2 * k + 1]), 0);
        const Real reach = norm(w - task.connector);
        if (reach > task.cableLength) w = task.connector + (w - task.connector) * (task.cableLength / reach);
        policy[k] = w;
    }
    return policy;
}

// A random cable: stiffness log-uniform over 0.4..4x the reference (a thin
// signal wire to a stiff power lead), friction 0.6..1.4x (board and jacket).
Attempt randomCable(const Harness::Policy& policy, std::mt19937& rng) {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    Attempt a;
    a.policy = policy;
    a.youngsScale = float(0.4 * std::pow(10.0, u(rng)));
    a.frictionScale = float(0.6 + 0.8 * u(rng));
    return a;
}

double successRate(const std::vector<Harness::Outcome>& outcomes) {
    int ok = 0;
    for (const auto& o : outcomes) ok += o.success() ? 1 : 0;
    return outcomes.empty() ? 0.0 : double(ok) / double(outcomes.size());
}

void writePolicy(const std::string& path, const Harness::Policy& policy) {
    if (std::FILE* f = std::fopen(path.c_str(), "w")) {
        for (const Vec3& w : policy) std::fprintf(f, "%.4f %.4f\n", double(w.x), double(w.y));
        std::fclose(f);
    }
}

}  // namespace

// Learning to route the harness on the GPU. A cross-entropy method over the 10
// waypoint coordinates, starting from the hand-written motion: each iteration
// tries 1024 motions, each on its own random cable, all in one GPU batch, and
// refits the motion distribution to the best 10%. The robot never sees the
// same cable twice, so what it learns is a motion that works across cables.
//
// Checked: the GPU batch reaches the same verdict as the double-precision CPU
// reference on sampled attempts; the learned motion routes far more fresh
// random cables than the hand-written one.
//
// Writes harness_learning.csv (per iteration), harness_grid.csv (the first 256
// attempts of each iteration: cable, score, verdict), harness_learned.policy,
// and, for the video, the final cable shapes of those 256 attempts per
// iteration to out/scenes/harness_grid.f32 (float32 x,y per particle).
CaseResult runHarnessLearning(const std::string& outDir) {
    CaseResult res;
    res.name = "application: a robot learns to route a wire harness (GPU)";
    if (!gpu::cudaAvailable()) {
        res.skipped = true;
        res.skipReason = gpu::diagnostic();
        return res;
    }

    const Harness task;
    const int population = 1024, iterations = 10, elites = population / 10, gridRods = 256;
    std::mt19937 rng(20260917);
    std::normal_distribution<double> gauss(0.0, 1.0);

    std::vector<double> mean = toParams(task.handWrittenPolicy());
    std::vector<double> sigma(mean.size(), 0.08);

    Csv csv(outDir, "harness_learning.csv", "iteration,success_rate,mean_score,seconds");
    Csv grid(outDir, "harness_grid.csv", "iteration,rod,youngs_scale,friction_scale,score,routed");
    std::FILE* shapes = outDir.empty() ? nullptr : std::fopen("out/scenes/harness_grid.f32", "wb");

    bool allRan = true;
    double learnSeconds = 0;
    const std::size_t particles = std::size_t(task.segments) + 1;
    for (int it = 0; it < iterations; ++it) {
        std::vector<std::vector<double>> samples;
        std::vector<Attempt> attempts;
        for (int r = 0; r < population; ++r) {
            std::vector<double> v = mean;
            for (std::size_t i = 0; i < v.size(); ++i) v[i] += sigma[i] * gauss(rng);
            const Harness::Policy policy = toPolicy(task, v);
            samples.push_back(toParams(policy));
            attempts.push_back(randomCable(policy, rng));
        }
        const auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<Vec3f> positions;
        const auto outcomes = attemptsOnGpu(task, attempts, &positions);
        const double seconds =
            std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
        learnSeconds += seconds;
        if (outcomes.size() != attempts.size()) {
            allRan = false;
            break;
        }

        double meanScore = 0;
        for (const auto& o : outcomes) meanScore += o.score / population;
        const double rate = successRate(outcomes);
        csv.row(it, rate, meanScore, seconds);
        res.notes.push_back(fmt("  iteration %.0f: %.1f%% of attempts routed, mean score %.2f",
                                double(it), 100 * rate, meanScore) +
                            fmt(" (%.1f s)", seconds));
        for (int r = 0; r < gridRods; ++r) {
            grid.row(it, r, attempts[r].youngsScale, attempts[r].frictionScale, outcomes[r].score,
                     outcomes[r].success() ? 1 : 0);
            if (shapes)
                for (std::size_t i = 0; i < particles; ++i) {
                    const Vec3f& p = positions[std::size_t(r) * particles + i];
                    std::fwrite(&p.x, sizeof(float), 1, shapes);
                    std::fwrite(&p.y, sizeof(float), 1, shapes);
                }
        }

        // Refit to the elites. A small floor keeps the search from collapsing
        // before it has seen enough cables.
        std::vector<int> order(population);
        for (int r = 0; r < population; ++r) order[r] = r;
        std::stable_sort(order.begin(), order.end(),
                         [&](int a, int b) { return outcomes[a].score > outcomes[b].score; });
        for (std::size_t i = 0; i < mean.size(); ++i) {
            double m = 0, var = 0;
            for (int e = 0; e < elites; ++e) m += samples[order[e]][i] / elites;
            for (int e = 0; e < elites; ++e) {
                const double d = samples[order[e]][i] - m;
                var += d * d / elites;
            }
            mean[i] = m;
            sigma[i] = std::max(std::sqrt(var), 0.003);
        }
    }
    if (shapes) std::fclose(shapes);

    // The learned motion against the hand-written one, on the same fresh cables.
    const Harness::Policy learned = toPolicy(task, mean);
    if (!outDir.empty()) writePolicy(outDir + "/harness_learned.policy", learned);
    std::vector<Attempt> fresh, freshHand;
    for (int r = 0; r < population; ++r) {
        fresh.push_back(randomCable(learned, rng));
        freshHand.push_back(fresh.back());
        freshHand.back().policy = task.handWrittenPolicy();
    }
    const auto learnedOutcomes = attemptsOnGpu(task, fresh);
    const auto handOutcomes = attemptsOnGpu(task, freshHand);
    const double learnedRate = successRate(learnedOutcomes);
    const double handRate = successRate(handOutcomes);
    Csv evaluation(outDir, "harness_evaluation.csv", "policy,success_rate,attempts,learning_seconds");
    evaluation.row("learned", learnedRate, population, learnSeconds);
    evaluation.row("hand_written", handRate, population, learnSeconds);
    allRan = allRan && learnedOutcomes.size() == fresh.size() && handOutcomes.size() == fresh.size();

    // CPU cross-check on the softest and stiffest fresh cables under each motion.
    int agree = 0, compared = 0;
    if (allRan) {
        auto extreme = [&](bool stiffest) {
            int best = 0;
            for (int r = 1; r < population; ++r)
                if ((fresh[r].youngsScale > fresh[best].youngsScale) == stiffest) best = r;
            return best;
        };
        for (int r : {extreme(false), extreme(true)}) {
            const bool cpuLearned = attemptOnCpu(task, fresh[r]).success();
            const bool cpuHand = attemptOnCpu(task, freshHand[r]).success();
            agree += cpuLearned == learnedOutcomes[r].success() ? 1 : 0;
            agree += cpuHand == handOutcomes[r].success() ? 1 : 0;
            compared += 2;
        }
    }

    res.notes.push_back(fmt("  %.0f iterations x %.0f attempts learned in %.1f s on the GPU",
                            double(iterations), double(population), learnSeconds));
    res.notes.push_back(fmt("  on %.0f fresh random cables: learned motion routes %.1f%%, "
                            "hand-written %.1f%%",
                            double(population), 100 * learnedRate, 100 * handRate));
    res.notes.push_back(fmt("  CPU and GPU agree on %.0f of %.0f cross-check attempts", double(agree),
                            double(compared)));
    res.checks.push_back(makeCheck("every batch ran", allRan ? 1.0 : 0.0, 1.0, 0.0));
    res.checks.push_back(makeCheck("CPU and GPU reach the same verdict", double(agree),
                                   double(compared), 0.0));
    res.checks.push_back(
        makeRangeCheck("learned motion routes most fresh cables", learnedRate, 0.8, 1.0));
    res.checks.push_back(makeRangeCheck("learned motion beats the hand-written one",
                                        learnedRate - handRate, 0.3, 1.0));
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

CaseResult runHarnessLearning(const std::string&) {
    CaseResult res;
    res.name = "application: a robot learns to route a wire harness (built without CUDA)";
    res.skipped = true;
    res.skipReason = "learning runs on the GPU; this build has no CUDA support";
    return res;
}

#endif

}  // namespace crs
