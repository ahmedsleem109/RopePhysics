// Demo scenes for the Phase 5 video.
//
// These are not validation cases: they produce trajectories for the offline
// renderer (tools/render_scene.py), not pass/fail checks. What they share with
// the validation suite is honesty about cost -- every frame records the wall
// clock the simulation took to produce it, so the video can state real timing
// instead of implying it.
//
// Output, per scene, in the chosen directory:
//   <name>.rodtraj   binary trajectory (format below)
//   <name>.json      scene description: primitives, radius, frame rate, totals
//
// .rodtraj layout, little-endian:
//   char[8]  "RODTRAJ1"
//   int32    numFrames, numRods, numParticles
//   float32  radius, fps
//   per frame:
//     float32  simulation wall-clock seconds spent producing this frame
//     float32  numRods * numParticles * 3 positions
//     int32    numContacts
//     float32  numContacts * 3 contact points
#define _CRT_SECURE_NO_WARNINGS  // plain fopen for a small JSON sidecar
#include "scenes.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <thread>
#include <vector>

#include "../core/collision.h"
#include "../apps/cable_hanging.h"
#include "../apps/harness_routing.h"
#include "../core/coloring.h"
#include "../core/solver.h"
#include "support.h"

namespace crs {

namespace {

class TrajectoryWriter {
   public:
    TrajectoryWriter(const std::string& path, int numFrames, int numRods, int numParticles,
                     float radius, float fps)
        : out_(path, std::ios::binary) {
        out_.write("RODTRAJ1", 8);
        writeI(numFrames);
        writeI(numRods);
        writeI(numParticles);
        writeF(radius);
        writeF(fps);
    }

    bool ok() const { return bool(out_); }

    // Positions already flattened rod-major, as a GPU batch downloads them.
    void frame(float wallSeconds, const std::vector<Vec3f>& positions) {
        writeF(wallSeconds);
        for (const Vec3f& x : positions) {
            writeF(x.x);
            writeF(x.y);
            writeF(x.z);
        }
        writeI(0);
    }

    void frame(float wallSeconds, const std::vector<const Rod*>& rods,
               const std::vector<Vec3>& contacts) {
        writeF(wallSeconds);
        for (const Rod* rod : rods)
            for (const Vec3& x : rod->state.x) writeV(x);
        writeI(static_cast<int32_t>(contacts.size()));
        for (const Vec3& c : contacts) writeV(c);
    }

   private:
    void writeI(int32_t v) { out_.write(reinterpret_cast<const char*>(&v), sizeof(v)); }
    void writeF(float v) { out_.write(reinterpret_cast<const char*>(&v), sizeof(v)); }
    void writeV(Vec3 v) {
        writeF(float(v.x));
        writeF(float(v.y));
        writeF(float(v.z));
    }

    std::ofstream out_;
};

// The point on the rod where a contact acts: the weighted combination of its
// first two particles, which for a world contact is the particle itself and for
// a self contact is the closest point on the first segment.
Vec3 contactPoint(const Rod& rod, const Contact& c) {
    Vec3 p;
    Real w = 0;
    for (int k = 0; k < std::min(c.count, 2); ++k) {
        p += rod.state.x[c.idx[k]] * c.weight[k];
        w += c.weight[k];
    }
    return w != Real(0) ? p / w : p;
}

void writePrimitiveJson(std::FILE* f, const Primitive& prim, bool last) {
    const char* names[] = {"plane", "sphere", "capsule", "box"};
    std::fprintf(f,
                 "    {\"type\": \"%s\", \"a\": [%g, %g, %g], \"b\": [%g, %g, %g], "
                 "\"normal\": [%g, %g, %g], \"radius\": %g, \"half_extents\": [%g, %g, %g]}%s\n",
                 names[prim.type], prim.a.x, prim.a.y, prim.a.z, prim.b.x, prim.b.y, prim.b.z,
                 prim.normal.x, prim.normal.y, prim.normal.z, prim.radius, prim.halfExtents.x,
                 prim.halfExtents.y, prim.halfExtents.z, last ? "" : ",");
}

struct SceneTotals {
    double wallSeconds = 0;
    double simSeconds = 0;
    double segmentSubsteps = 0;
    int threads = 1;
    const char* device = "CPU";
};

struct SceneLabel {
    std::string text;
    Vec3 at;
};

void writeJson(const std::string& path, const char* name, const char* description,
               const CollisionWorld& world, float radius, float fps, int frames, int rods,
               int particles, const SceneTotals& t, const SolverParams& p, int stepsPerFrame,
               const std::vector<SceneLabel>& labels = {}) {
    std::FILE* f = std::fopen(path.c_str(), "w");
    if (!f) return;
    std::fprintf(f, "{\n  \"name\": \"%s\",\n  \"description\": \"%s\",\n", name, description);
    std::fprintf(f, "  \"fps\": %g,\n  \"frames\": %d,\n  \"rods\": %d,\n  \"particles\": %d,\n",
                 fps, frames, rods, particles);
    std::fprintf(f, "  \"radius\": %g,\n", radius);
    std::fprintf(f,
                 "  \"solver\": {\"dt\": %g, \"substeps\": %d, \"iterations\": %d, "
                 "\"steps_per_frame\": %d},\n",
                 double(p.dt), p.substeps, p.iterations, stepsPerFrame);
    std::fprintf(f,
                 "  \"timing\": {\"simulated_seconds\": %g, \"wall_seconds\": %g, "
                 "\"realtime_factor\": %g, \"segment_substeps\": %g, "
                 "\"segment_substeps_per_sec\": %g, \"threads\": %d, \"device\": \"%s\"},\n",
                 t.simSeconds, t.wallSeconds, t.simSeconds / t.wallSeconds, t.segmentSubsteps,
                 t.segmentSubsteps / t.wallSeconds, t.threads, t.device);
    std::fprintf(f, "  \"labels\": [\n");
    for (std::size_t i = 0; i < labels.size(); ++i)
        std::fprintf(f, "    {\"text\": \"%s\", \"at\": [%g, %g, %g]}%s\n", labels[i].text.c_str(),
                     double(labels[i].at.x), double(labels[i].at.y), double(labels[i].at.z),
                     i + 1 == labels.size() ? "" : ",");
    std::fprintf(f, "  ],\n");
    std::fprintf(f, "  \"primitives\": [\n");
    for (std::size_t i = 0; i < world.primitives.size(); ++i)
        writePrimitiveJson(f, world.primitives[i], i + 1 == world.primitives.size());
    std::fprintf(f, "  ]\n}\n");
    std::fclose(f);
}

using Clock = std::chrono::high_resolution_clock;

// ---------------------------------------------------------------- drape
//
// A cable dropped across a horizontal post drapes over it and piles onto the
// floor on both sides, resting on itself. Then both ends are gripped and lifted,
// and one end is wound ten turns about its own axis.
//
// Both ends have to be held for the second half to show anything. The first
// version of this scene twisted a cable with a free end, and nothing happened --
// correctly: a free end simply spins the twist straight back out. With both ends
// clamped the twist has nowhere to go, and past the Michell threshold it trades
// itself for writhe: the hanging cable throws loops and coils around itself.
// That is the behaviour a rod model without a material frame cannot produce.
int sceneDrape(const std::string& outDir) {
    const float fps = 60;
    const int frames = 600;  // 10 s
    const int stepsPerFrame = 4;

    RodMaterial mat;
    mat.youngs = Real(5e6);
    mat.poisson = Real(0.4);
    mat.density = Real(1100);
    mat.radius = Real(8e-3);

    // Element length 16.7 mm against a 16 mm diameter: longer than the rod is
    // thick, which is the regime the failure study found the stability rule to
    // hold in.
    const int n = 120;
    const Real L = Real(2.0);
    const Real postX = Real(0.05), postZ = Real(0.5), postR = Real(0.04);
    const Vec3 start(postX - L / 2, 0, postZ + postR + mat.radius + Real(0.01));
    Rod rod = makeStraightRod(n, L, mat, start, Vec3(1, 0, 0));

    CollisionWorld world;
    world.primitives.push_back(Primitive::makePlane(Vec3(0, 0, 0), Vec3(0, 0, 1), Real(0.6)));
    world.primitives.push_back(Primitive::makeCapsule(Vec3(postX, -0.35, postZ),
                                                      Vec3(postX, 0.35, postZ), postR, Real(0.5)));
    world.selfCollision = true;
    world.selfFriction = Real(0.4);

    SolverParams p;
    p.dt = Real(1.0 / (fps * stepsPerFrame));
    p.substeps = 8;
    p.iterations = 2;
    p.gravity = Vec3(0, 0, Real(-9.81));
    p.linearDamping = Real(0.5);
    p.angularDamping = Real(0.5);

    SolverContext ctx;
    TrajectoryWriter traj(outDir + "/drape.rodtraj", frames, 1, int(rod.state.numParticles()),
                          float(mat.radius), fps);
    if (!traj.ok()) return 1;

    // Choreography, in simulated seconds.
    const double gripAt = 3.5, liftEnd = 5.0, windTurnsPerSec = 2.0;
    const int endA = 0, endB = int(rod.state.numParticles()) - 1;
    const Vec3 targetA(postX - Real(0.45), 0, Real(0.30));
    const Vec3 targetB(postX + Real(0.45), 0, Real(0.30));
    bool gripped = false;
    Vec3 fromA, fromB, windAxis;
    int ghostB = -1;
    Quat ghostBBase;

    SceneTotals totals;
    for (int f = 0; f < frames; ++f) {
        const auto t0 = Clock::now();
        for (int s = 0; s < stepsPerFrame; ++s) {
            const double t = totals.simSeconds;
            if (!gripped && t >= gripAt) {
                // Grip: clamp both ends where they lie, position and frame.
                fromA = rod.state.x[endA];
                fromB = rod.state.x[endB];
                clampEndExact(rod, false);
                ghostB = clampEndExact(rod, true);
                ghostBBase = rod.state.q[ghostB];
                // Wind about the end's own tangent, so the rotation is twist
                // and not a bend imposed at the grip.
                windAxis = rotate(ghostBBase, Vec3(0, 0, 1));
                gripped = true;
            }
            if (gripped) {
                const double u = std::min(1.0, (t - gripAt) / (liftEnd - gripAt));
                const Real w = Real(u * u * (3 - 2 * u));  // smoothstep: no jerk at either end
                rod.state.x[endA] = fromA + (targetA - fromA) * w;
                rod.state.x[endB] = fromB + (targetB - fromB) * w;
                if (t > liftEnd) {
                    const Real angle = Real(2 * kPi * windTurnsPerSec * (t - liftEnd));
                    rod.state.q[ghostB] = normalize(quatAxisAngle(windAxis, angle) * ghostBBase);
                }
            }
            step(rod, p, world, ctx);
            totals.simSeconds += double(p.dt);
        }
        const double wall = std::chrono::duration<double>(Clock::now() - t0).count();
        totals.wallSeconds += wall;
        totals.segmentSubsteps += double(n) * stepsPerFrame * p.substeps;

        std::vector<Vec3> contacts;
        for (const Contact& c : ctx.contacts.contacts) contacts.push_back(contactPoint(rod, c));
        traj.frame(float(wall), {&rod}, contacts);
    }

    writeJson(outDir + "/drape.json", "drape",
              "a cable drapes over a post; its ends are lifted and one is wound ten turns", world,
              float(mat.radius), fps, frames, 1, int(rod.state.numParticles()), totals, p,
              stepsPerFrame);
    std::printf("drape: %d frames, %.2f s simulated in %.2f s wall (%.2fx realtime)\n", frames,
                totals.simSeconds, totals.wallSeconds, totals.simSeconds / totals.wallSeconds);
    return 0;
}

// ---------------------------------------------------------------- grid
//
// A field of hanging rods, each released from a different tilt, whipping and
// dragging their tips across the floor before settling. Every rod is independent -- this is the batched-environment shape
// Phase 4 is about, stepped here across CPU threads.
int sceneGrid(const std::string& outDir) {
    const float fps = 60;
    const int frames = 900;  // 15 s
    const int stepsPerFrame = 4;
    const int side = 16;
    const int rods = side * side;
    const int n = 20;
    const Real L = Real(0.8);
    const Real spacing = Real(0.22);

    const RodMaterial mat = referenceMaterial();
    std::vector<Rod> batch;
    batch.reserve(rods);
    for (int iy = 0; iy < side; ++iy) {
        for (int ix = 0; ix < side; ++ix) {
            const Vec3 anchor(spacing * (ix - (side - 1) * Real(0.5)),
                              spacing * (iy - (side - 1) * Real(0.5)), Real(0.7));
            // Tilt varies smoothly across the grid, so the field moves as a wave
            // rather than as noise.
            const Real azimuth = Real(0.35) * ix + Real(0.2) * iy;
            const Real tilt = Real(0.3) + Real(1.1) * (Real(iy) / (side - 1));
            const Vec3 dir(std::sin(tilt) * std::cos(azimuth), std::sin(tilt) * std::sin(azimuth),
                           -std::cos(tilt));
            Rod rod = makeStraightRod(n, L, mat, anchor, dir);
            clampRootExact(rod);
            batch.push_back(std::move(rod));
        }
    }

    CollisionWorld world;
    world.primitives.push_back(Primitive::makePlane(Vec3(0, 0, 0), Vec3(0, 0, 1), Real(0.5)));

    SolverParams p;
    p.dt = Real(1.0 / (fps * stepsPerFrame));
    p.substeps = 8;
    p.iterations = 1;
    p.gravity = Vec3(0, 0, Real(-9.81));
    p.linearDamping = Real(0.3);
    p.angularDamping = Real(0.3);

    const int threads = int(std::max(1u, std::thread::hardware_concurrency()));
    std::vector<SolverContext> contexts(rods);

    std::vector<const Rod*> views;
    for (const Rod& r : batch) views.push_back(&r);

    TrajectoryWriter traj(outDir + "/grid.rodtraj", frames, rods, n + 1, float(mat.radius), fps);
    if (!traj.ok()) return 1;

    SceneTotals totals;
    totals.threads = threads;
    for (int f = 0; f < frames; ++f) {
        const auto t0 = Clock::now();
        std::vector<std::thread> pool;
        for (int t = 0; t < threads; ++t) {
            pool.emplace_back([&, t] {
                for (int r = t; r < rods; r += threads)
                    for (int s = 0; s < stepsPerFrame; ++s) step(batch[r], p, world, contexts[r]);
            });
        }
        for (std::thread& th : pool) th.join();
        const double wall = std::chrono::duration<double>(Clock::now() - t0).count();

        totals.wallSeconds += wall;
        totals.simSeconds += double(p.dt) * stepsPerFrame;
        totals.segmentSubsteps += double(rods) * n * stepsPerFrame * p.substeps;
        traj.frame(float(wall), views, {});
    }

    writeJson(outDir + "/grid.json", "grid",
              "256 independent hanging rods released from graded tilts", world, float(mat.radius),
              fps, frames, rods, n + 1, totals, p, stepsPerFrame);
    std::printf("grid: %d rods x %d frames, %.2f s simulated in %.2f s wall on %d threads "
                "(%.3g segment-substeps/s)\n",
                rods, frames, totals.simSeconds, totals.wallSeconds, threads,
                totals.segmentSubsteps / totals.wallSeconds);
    return 0;
}


// ---------------------------------------------------------------- cable-hanging
//
// The application in one picture: six identical cables, each draped over a bar
// with the same 2:1 placement and released, on six bars that differ only in
// friction. The capstan equation says a 2:1 drape needs mu >= ln 2 / pi = 0.22,
// so the three on the left should slide off and the three on the right hold.
// The frictions stay clear of the boundary on both sides: close above it this
// solver's friction creeps, and a cable theory says holds can slide off after
// a few seconds (see apps::HangingCable::duration).
// Simulated with the task definition the GPU sweep uses (apps::HangingCable),
// here on the CPU because each bar has its own friction.
int sceneCableHanging(const std::string& outDir) {
    const apps::HangingCable task;
    const Real ratio = Real(2.0);
    const Real youngs = Real(1e6);
    const std::vector<Real> frictions = {Real(0.08), Real(0.15), Real(0.20),
                                         Real(0.32), Real(0.40), Real(0.50)};
    const Real spacing = Real(0.55);

    const float fps = 60;
    const int frames = 360;  // 6 s: release, then either slide off or settle
    SolverParams p = task.params();
    const int stepsPerFrame = int(std::lround(1.0 / (fps * double(p.dt))));
    p.dt = Real(1.0 / (fps * stepsPerFrame));

    std::vector<Rod> rods;
    std::vector<CollisionWorld> worlds;
    std::vector<SolverContext> contexts(frictions.size());
    CollisionWorld display;  // every bar, for the renderer
    std::vector<SceneLabel> labels;
    for (std::size_t k = 0; k < frictions.size(); ++k) {
        const Vec3 at(spacing * (Real(k) - Real(frictions.size() - 1) / 2), 0, 0);
        rods.push_back(task.build(ratio, youngs, at));
        CollisionWorld w = task.world(frictions[k], at);
        worlds.push_back(w);
        display.primitives.push_back(w.primitives.front());
        char text[64];
        std::snprintf(text, sizeof(text), "mu = %.2f", double(frictions[k]));
        labels.push_back({text, at + Vec3(0, 0, task.barHeight + Real(0.16))});
    }
    display.primitives.push_back(Primitive::makePlane(Vec3(0, 0, 0), Vec3(0, 0, 1), Real(0.5)));

    std::vector<const Rod*> views;
    for (const Rod& r : rods) views.push_back(&r);
    TrajectoryWriter traj(outDir + "/cable-hanging.rodtraj", frames, int(rods.size()),
                          task.segments + 1, float(task.material().radius), fps);
    if (!traj.ok()) return 1;

    SceneTotals totals;
    for (int f = 0; f < frames; ++f) {
        const auto t0 = Clock::now();
        for (std::size_t k = 0; k < rods.size(); ++k)
            for (int s = 0; s < stepsPerFrame; ++s) step(rods[k], p, worlds[k], contexts[k]);
        const double wall = std::chrono::duration<double>(Clock::now() - t0).count();
        totals.wallSeconds += wall;
        totals.simSeconds += double(p.dt) * stepsPerFrame;
        totals.segmentSubsteps += double(rods.size()) * task.segments * stepsPerFrame * p.substeps;
        traj.frame(float(wall), views, {});
    }

    writeJson(outDir + "/cable-hanging.json", "cable-hanging",
              "Will the cable stay on the hook? Same cable, same 2:1 drape. Theory: needs mu >= 0.22",
              display, float(task.material().radius), fps, frames, int(rods.size()),
              task.segments + 1, totals, p, stepsPerFrame, labels);
    std::printf("cable-hanging: %zu cables x %d frames, %.2f s simulated in %.2f s wall\n",
                rods.size(), frames, totals.simSeconds, totals.wallSeconds);
    return 0;
}

// ---------------------------------------------------------------- harness
//
// One attempt at routing the harness with the hand-written policy, on the CPU,
// for looking at: the cable trajectory, plus a sidecar CSV of the gripper's
// path so the renderer can draw the robot holding it.
int sceneHarness(const std::string& outDir) {
    const apps::HarnessTask task;
    apps::HarnessTask::Policy policy = task.handWrittenPolicy();
    // A policy to replay instead, if <outDir>/harness.policy exists: one "x y"
    // pair per waypoint (the learning case writes one).
    if (std::FILE* pf = std::fopen((outDir + "/harness.policy").c_str(), "r")) {
        for (Vec3& w : policy) {
            double x = 0, y = 0;
            if (std::fscanf(pf, "%lf %lf", &x, &y) == 2) w = Vec3(Real(x), Real(y), 0);
        }
        std::fclose(pf);
        std::printf("harness: replaying %s/harness.policy\n", outDir.c_str());
    }
    // And the cable, if <outDir>/harness.cable exists: "youngsScale frictionScale".
    double youngsScale = 1, frictionScale = 1;
    if (std::FILE* cf = std::fopen((outDir + "/harness.cable").c_str(), "r")) {
        if (std::fscanf(cf, "%lf %lf", &youngsScale, &frictionScale) != 2)
            youngsScale = frictionScale = 1;
        std::fclose(cf);
        std::printf("harness: cable stiffness x%.2f, friction x%.2f\n", youngsScale, frictionScale);
    }
    Rod rod = task.build(Real(youngsScale));
    const CollisionWorld world = task.world(Real(frictionScale));
    SolverParams p = task.params();
    SolverContext ctx;

    const float fps = 60;
    const int stepsPerFrame = int(std::lround(1.0 / (fps * double(p.dt))));
    const int frames = task.steps() / stepsPerFrame;

    TrajectoryWriter traj(outDir + "/harness.rodtraj", frames, 1, task.segments + 1,
                          float(task.material().radius), fps);
    std::FILE* grip = std::fopen((outDir + "/harness.gripper.csv").c_str(), "w");
    if (!traj.ok() || !grip) return 1;
    std::fprintf(grip, "frame,x,y,z\n");

    SceneTotals totals;
    int stepIndex = 0;
    const std::vector<const Rod*> views = {&rod};
    for (int f = 0; f < frames; ++f) {
        const auto t0 = Clock::now();
        for (int s = 0; s < stepsPerFrame; ++s, ++stepIndex) {
            rod.state.kinematicVelocity.back() =
                task.gripperVelocity(policy, stepIndex);
            step(rod, p, world, ctx);
        }
        const double wall = std::chrono::duration<double>(Clock::now() - t0).count();
        totals.wallSeconds += wall;
        totals.simSeconds += double(p.dt) * stepsPerFrame;
        totals.segmentSubsteps += double(task.segments) * stepsPerFrame * p.substeps;
        traj.frame(float(wall), views, {});
        const Vec3 g = rod.state.x.back();
        std::fprintf(grip, "%d,%g,%g,%g\n", f, double(g.x), double(g.y), double(g.z));
    }
    std::fclose(grip);

    const auto outcome = task.evaluate(rod.state.x);
    writeJson(outDir + "/harness.json", "harness",
              "A robot routes a cable: under peg A, over peg B, through the clip", world,
              float(task.material().radius), fps, frames, 1, task.segments + 1, totals, p,
              stepsPerFrame);
    std::printf("harness: under A %s (%+.3f), over B %s (%+.3f), through clip %s (miss %.3f) -> "
                "%s (score %.2f)\n",
                outcome.belowA ? "yes" : "no", outcome.sideA, outcome.aboveB ? "yes" : "no",
                outcome.sideB, outcome.throughClip ? "yes" : "no", outcome.clipMiss,
                outcome.success() ? "ROUTED" : "FAILED", outcome.score);
    return 0;
}

}  // namespace

int runScene(const std::string& name, const std::string& outDir) {
    if (name == "harness") return sceneHarness(outDir);
    if (name == "drape") return sceneDrape(outDir);
    if (name == "grid") return sceneGrid(outDir);
    if (name == "cable-hanging") return sceneCableHanging(outDir);
    std::printf("unknown scene '%s' (drape|grid|cable-hanging)\n", name.c_str());
    return 2;
}

}  // namespace crs
