// The Phase 1-2 validation suite, run on the GPU path.
//
// `gpu-parity` shows the device reproduces the CPU trajectory for a handful of
// steps. That is necessary and not sufficient: a batch of float rods could
// agree with the CPU for 200 steps and still get a slip angle, a capstan ratio
// or a buckling threshold wrong over the thousands of steps those measurements
// take -- because rounding accumulates, because a contact regime the short runs
// never reached behaves differently, or because float cannot resolve the
// motion at all. So every CPU validation case with a scenario the batch can
// express is run here, end to end, through gpu::Batch, and held to the
// analytic reference at the CPU case's own tolerance.
//
// Each scenario is written once against a small `Sim` interface and run twice:
// on the GPU, and on the double-precision CPU solver in the GPU's colour order.
// The GPU-minus-CPU difference of every measured quantity is then checked
// against a bound taken from the float CPU build (CRS_REAL_FLOAT): what the
// same CPU code moves by when only the precision changes. Bitwise agreement
// with double is not the claim and cannot be.
//
// Where the CPU case solves statics directly (cantilever, tip-load and
// end-moment elastica), this runs the XPBD relaxation instead -- there is no
// direct solver on the device -- and also checks it against the direct solve
// of the same discrete rod.
//
// Scenarios that vary per environment are batched the way an RL batch would
// be: the capstan runs every tension ratio and friction coefficient of a wrap
// angle as one batch, the buckling case every twist of a mesh.
//
// Without CUDA the case skips. Setting CRS_GPU_SUITE_CPU=1 runs the CPU half
// alone and prints what it measured, which is how the float bounds were taken.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <thread>
#include <vector>

#include "../core/coloring.h"
#include "../core/solver.h"
#include "../core/statics.h"
#include "../validation/cases.h"
#include "../validation/support.h"
#include "gpu_solver.h"

namespace crs {

namespace {

enum class Backend { kCpu, kGpu };

// A batch of environments sharing one rod topology and one collision world.
class Sim {
   public:
    virtual ~Sim() = default;
    // Full state and applied loads of one environment.
    virtual void setRod(int i, const Rod& rod) = 0;
    // Per-environment multiplier on every friction coefficient.
    virtual void setFrictionScales(const std::vector<float>& scales) = 0;
    virtual void advance(const SolverParams& p, int steps) = 0;
    // Positions, velocities, orientations and angular velocities.
    virtual void get(int i, Rod& out) = 0;
};

// The reference: double precision, swept in the GPU's colour order, one thread
// per group of environments.
class CpuSim : public Sim {
   public:
    CpuSim(const Rod& prototype, int rods, const CollisionWorld* world)
        : stretch_(colorStretchConstraints(prototype)),
          bend_(colorBendConstraints(prototype)),
          rods_(rods, prototype),
          ctx_(rods) {
        if (world) worlds_.assign(rods, *world);
    }

    void setRod(int i, const Rod& rod) override { rods_[i] = rod; }

    void setFrictionScales(const std::vector<float>& scales) override {
        for (std::size_t r = 0; r < scales.size() && r < worlds_.size(); ++r) {
            for (Primitive& prim : worlds_[r].primitives) prim.friction *= Real(scales[r]);
            worlds_[r].selfFriction *= Real(scales[r]);
        }
    }

    void advance(const SolverParams& params, int steps) override {
        SolverParams p = params;
        p.stretchColoring = &stretch_;
        p.bendColoring = &bend_;
        auto run = [&](int first, int last) {
            for (int r = first; r < last; ++r)
                for (int s = 0; s < steps; ++s) {
                    if (worlds_.empty())
                        step(rods_[r], p);
                    else
                        step(rods_[r], p, worlds_[r], ctx_[r]);
                }
        };
        const int n = static_cast<int>(rods_.size());
        const int threads = std::min(n, std::max(1, int(std::thread::hardware_concurrency())));
        if (threads <= 1) {
            run(0, n);
            return;
        }
        std::vector<std::thread> pool;
        for (int t = 0; t < threads; ++t)
            pool.emplace_back(run, n * t / threads, n * (t + 1) / threads);
        for (std::thread& th : pool) th.join();
    }

    void get(int i, Rod& out) override { out = rods_[i]; }

   private:
    Coloring stretch_, bend_;
    std::vector<Rod> rods_;
    std::vector<SolverContext> ctx_;
    std::vector<CollisionWorld> worlds_;
};

#ifdef CRS_WITH_CUDA
class GpuSim : public Sim {
   public:
    bool create(const Rod& prototype, int rods, const CollisionWorld* world) {
        // Fused wherever the rod fits a block's shared memory; the capstan's
        // longest wrap may not.
        return batch_.create(prototype, rods, gpu::Strategy::kFused, world) ||
               batch_.create(prototype, rods, gpu::Strategy::kMultiKernel, world);
    }

    void setRod(int i, const Rod& rod) override {
        batch_.synchronize();
        batch_.uploadRod(i, rod);
    }

    void setFrictionScales(const std::vector<float>& scales) override {
        batch_.setMaterialScales({}, scales);
    }

    void advance(const SolverParams& p, int steps) override {
        gpu::BatchParams b;
        b.dt = float(p.dt);
        b.substeps = p.substeps;
        b.iterations = p.iterations;
        b.gravityX = float(p.gravity.x);
        b.gravityY = float(p.gravity.y);
        b.gravityZ = float(p.gravity.z);
        b.linearDamping = float(p.linearDamping);
        b.angularDamping = float(p.angularDamping);
        b.contactInterval = p.contactInterval;
        for (int s = 0; s < steps; ++s) batch_.step(b);
    }

    void get(int i, Rod& out) override {
        batch_.synchronize();
        batch_.download(i, out);
    }

    long long selfContactOverflow() const { return batch_.selfContactOverflow(); }

   private:
    gpu::Batch batch_;
};
#endif

// Null only if the GPU batch could not be created.
std::unique_ptr<Sim> makeSim(Backend b, const Rod& prototype, int rods,
                             const CollisionWorld* world = nullptr) {
#ifdef CRS_WITH_CUDA
    if (b == Backend::kGpu) {
        auto sim = std::make_unique<GpuSim>();
        if (!sim->create(prototype, rods, world)) {
            std::fprintf(stderr, "gpu-suite: batch of %d x %zu segments not created: %s\n", rods,
                         prototype.stretch.size(), gpu::diagnostic().c_str());
            return nullptr;
        }
        return sim;
    }
#else
    if (b == Backend::kGpu) return nullptr;
#endif
    return std::make_unique<CpuSim>(prototype, rods, world);
}

// One measured quantity: accepted if lo <= value <= hi (the CPU case's
// tolerance), and compared across backends if agree >= 0.
struct Measure {
    std::string name;
    double value = 0;
    double lo = 0, hi = 0;
    double agree = -1;  // largest accepted |GPU - CPU|; negative: not compared
};

using Measures = std::vector<Measure>;

Measure measure(const std::string& name, double value, double lo, double hi, double agree) {
    Measure m;
    m.name = name;
    m.value = value;
    m.lo = lo;
    m.hi = hi;
    m.agree = agree;
    return m;
}

// ---------------------------------------------------------------- statics
//
// Cantilever and tip-load elastica in one batch: one rod per load, alpha =
// P L^2 / EI. The smallest load (tip deflection 0.5% of L) is the cantilever,
// checked against Euler-Bernoulli; the rest against the exact elastica. Every
// rod is also checked against the direct static solve of the same discrete rod,
// which is what the XPBD relaxation converges to.
Measures scenarioTipLoad(Backend b) {
    const RodMaterial mat = referenceMaterial();
    const Real L = Real(1);
    const Real EI = mat.bendStiffness();
    const int n = 32;
    const std::vector<double> alphas = {0.015, 0.5, 1.0, 2.0, 3.0, 5.0};

    Rod prototype = makeStraightRod(n, L, mat, Vec3(0, 0, 0), Vec3(1, 0, 0));
    clampRootExact(prototype);
    auto loaded = [&](double alpha) {
        Rod rod = prototype;
        rod.state.extForce.back() = Vec3(0, 0, -Real(alpha) * EI / (L * L));
        return rod;
    };
    // Substeps sized for the heaviest load, shared by the batch.
    const SolverParams p = staticParams(loaded(alphas.back()), 2 * cantileverOmega1(mat, L));

    auto sim = makeSim(b, prototype, int(alphas.size()));
    if (!sim) return {};
    for (std::size_t k = 0; k < alphas.size(); ++k) sim->setRod(int(k), loaded(alphas[k]));
    sim->advance(p, 3000);

    double worstDirect = 0, worstElastica = 0, cantilever = 0;
    for (std::size_t k = 0; k < alphas.size(); ++k) {
        Rod rod = prototype;
        sim->get(int(k), rod);
        const Vec3 tip = rod.state.x.back();
        Rod direct = loaded(alphas[k]);
        solveStatic(direct);
        worstDirect = std::max(worstDirect, double(norm(tip - direct.state.x.back())) / L);
        const double P = alphas[k] * EI / (L * L);
        if (k == 0) {
            const double deltaEB = P * L * L * L / (3 * EI);
            cantilever = std::abs(-tip.z - deltaEB) / deltaEB;
        } else {
            const ElasticaRef ref = solveElasticaRef(P, L, EI);
            worstElastica = std::max(
                worstElastica, std::hypot(tip.x - ref.tipX, tip.z - ref.tipZ) / L);
        }
    }
    return {
        measure("cantilever tip vs Euler-Bernoulli", cantilever, 0, 0.01, 1e-4),
        measure("elastica tip vs exact, alpha 0.5-5 (/L)", worstElastica, 0, 0.005, 1e-4),
        measure("tip vs direct static solve (/L)", worstDirect, 0, 1e-4, 1e-4),
    };
}

// Pure end moment: a half and a quarter circle, radius EI / M.
Measures scenarioEndMoment(Backend b) {
    const RodMaterial mat = referenceMaterial();
    const Real L = Real(1);
    const Real EI = mat.bendStiffness();
    const int n = 32;
    const std::vector<double> kappas = {kPi / L, 0.5 * kPi / L};

    Rod prototype = makeStraightRod(n, L, mat, Vec3(0, 0, 0), Vec3(1, 0, 0));
    const int lastSeg = static_cast<int>(prototype.state.numSegments()) - 1;
    clampRootExact(prototype);
    auto loaded = [&](double kappa) {
        Rod rod = prototype;
        rod.state.extTorque[lastSeg] = Vec3(0, -EI * Real(kappa), 0);
        return rod;
    };
    const SolverParams p = staticParams(loaded(kappas.front()), 2 * cantileverOmega1(mat, L));

    auto radiusOf = [&](const Rod& rod) {
        const Vec3 t0 = rotate(rod.state.q.front(), Vec3(0, 0, 1));
        const Vec3 t1 = rotate(rod.state.q[lastSeg], Vec3(0, 0, 1));
        const double turn = std::atan2(double(norm(cross(t0, t1))), double(dot(t0, t1)));
        return L * (n - 1.0) / n / turn;
    };

    auto sim = makeSim(b, prototype, int(kappas.size()));
    if (!sim) return {};
    for (std::size_t k = 0; k < kappas.size(); ++k) sim->setRod(int(k), loaded(kappas[k]));
    sim->advance(p, 3000);

    double worstRef = 0, worstDirect = 0;
    for (std::size_t k = 0; k < kappas.size(); ++k) {
        Rod rod = prototype;
        sim->get(int(k), rod);
        Rod direct = loaded(kappas[k]);
        solveStatic(direct);
        const double r = radiusOf(rod), rRef = 1 / kappas[k];
        worstRef = std::max(worstRef, std::abs(r - rRef) / rRef);
        worstDirect = std::max(worstDirect, std::abs(r - radiusOf(direct)) / rRef);
    }
    return {
        measure("end-moment arc radius vs EI/M", worstRef, 0, 0.005, 1e-4),
        measure("arc radius vs direct static solve", worstDirect, 0, 1e-4, 1e-4),
    };
}

// Intrinsic curvature and twist relax to a helix.
Measures scenarioHelix(Backend b) {
    const RodMaterial mat = referenceMaterial();
    const Real L = Real(1);
    const int n = 100;
    const Real kappa = Real(10), tau = Real(5);
    const double Rref = kappa / (kappa * kappa + tau * tau);
    const double pitchRef = 2 * kPi * tau / (kappa * kappa + tau * tau);

    Rod rod = makeStraightRod(n, L, mat, Vec3(0, 0, 0), Vec3(0, 0, 1));
    for (std::size_t k = 0; k < rod.bend.size(); ++k) rod.bend.restDarboux[k] = Vec3(0, kappa, tau);
    const int lastSeg = static_cast<int>(rod.state.numSegments()) - 1;
    Rod straight = rod;

    SolverParams p;
    p.dt = Real(0.005);
    p.substeps = 16;
    p.iterations = 32;
    p.gravity = Vec3();
    p.linearDamping = p.angularDamping = Real(20);

    auto sim = makeSim(b, rod, 1);
    if (!sim) return {};
    sim->advance(p, 2500);  // the CPU case converges in 1836
    sim->get(0, rod);

    const HelixShape shape = measureHelix(rod, lastSeg);
    const double energy = rod.elasticEnergy() / straight.elasticEnergy();
    return {
        measure("helix radius (relative)", std::abs(shape.radius - Rref) / Rref, 0, 0.02, 3e-3),
        measure("helix pitch (relative)", std::abs(shape.pitch - pitchRef) / pitchRef, 0, 0.02,
                1e-3),
        measure("helix residual elastic energy", energy, 0, 0.01, 1e-3),
    };
}

// ---------------------------------------------------------------- buckling
//
// Michell/Greenhill threshold from growth rates, as the CPU case measures it,
// with every twist of a mesh in one batch. Amplitudes are sampled every few
// steps and the crossing times interpolated in log amplitude.
Measures scenarioTwistBuckling(Backend b) {
    const RodMaterial mat = referenceMaterial();
    const Real L = Real(1);
    const double phiCritRef = 8.986819 * mat.bendStiffness() / mat.twistStiffness();
    const std::vector<int> counts = {16, 24, 32};
    const std::vector<double> overshoots = {0.02, 0.04, 0.06, 0.08, 0.10, 0.12};
    const int maxSteps = 30000, chunk = 4;

    std::vector<double> hs, errs;
    Measures out;
    for (int n : counts) {
        Rod prototype = makeStraightRod(n, L, mat, Vec3(0, 0, 0), Vec3(0, 0, 1));
        clampEndExact(prototype, false);
        const int tipGhost = clampEndExact(prototype, true);
        const Real l = prototype.stretch.restLength.front();
        const int nSeg = static_cast<int>(prototype.stretch.size());

        auto sim = makeSim(b, prototype, int(overshoots.size()));
        if (!sim) return {};
        for (std::size_t k = 0; k < overshoots.size(); ++k) {
            const double phi = phiCritRef * (1 + overshoots[k]);
            Rod rod = prototype;
            for (int j = 0; j < nSeg; ++j) {
                const Real sMid = (Real(j) + Real(0.5)) * l;
                rod.state.q[j] =
                    normalize(quatAxisAngle(Vec3(0, 0, 1), Real(phi) * sMid / L) * rod.state.q[j]);
            }
            rod.state.q[tipGhost] =
                normalize(quatAxisAngle(Vec3(0, 0, 1), Real(phi)) * rod.state.q[tipGhost]);
            rod.state.qPrev = rod.state.q;
            for (std::size_t i = 1; i + 1 < rod.state.numParticles(); ++i)
                rod.state.x[i].x += Real(1e-7) * std::sin(Real(kPi) * rod.state.x[i].z / L);
            rod.state.xPrev = rod.state.x;
            sim->setRod(int(k), rod);
        }

        SolverParams p;
        p.dt = Real(1e-3);
        p.substeps = 16 * n;
        p.iterations = 1;
        p.gravity = Vec3();

        const std::size_t K = overshoots.size();
        std::vector<double> t1(K, -1), t2(K, -1), prevAmp(K, 0);
        Rod rod = prototype;
        for (int i = chunk; i <= maxSteps; i += chunk) {
            sim->advance(p, chunk);
            bool all = true;
            for (std::size_t k = 0; k < K; ++k) {
                if (t2[k] >= 0) continue;
                sim->get(int(k), rod);
                double amp = 0;
                for (const Vec3& x : rod.state.x)
                    amp = std::max(amp, std::sqrt(double(x.x * x.x + x.y * x.y)));
                const double t = i * double(p.dt);
                auto crossing = [&](double level) {
                    if (prevAmp[k] <= 0) return t;
                    const double f = std::log(level / prevAmp[k]) / std::log(amp / prevAmp[k]);
                    return t - chunk * double(p.dt) * (1 - std::clamp(f, 0.0, 1.0));
                };
                if (t1[k] < 0 && amp > 1e-5) t1[k] = crossing(1e-5);
                if (t1[k] >= 0 && amp > 1e-3) t2[k] = crossing(1e-3);
                prevAmp[k] = amp;
                all = all && t2[k] >= 0;
            }
            if (all) break;
        }

        // sigma^2 is quadratic in the overshoot u; its rising root is the threshold.
        double S[5] = {0, 0, 0, 0, 0}, T[3] = {0, 0, 0};
        int used = 0;
        for (std::size_t k = 0; k < K; ++k) {
            if (t2[k] < 0) continue;
            const double sigma = std::log(100.0) / (t2[k] - t1[k]);
            const double u = overshoots[k], y = sigma * sigma;
            double uk = 1;
            for (int e = 0; e < 5; ++e, uk *= u) S[e] += uk;
            T[0] += y, T[1] += y * u, T[2] += y * u * u;
            ++used;
        }
        double rel = 1;
        if (used >= 4) {
            auto det3 = [](double m[3][3]) {
                return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
                       m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                       m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
            };
            double A[3][3] = {{S[0], S[1], S[2]}, {S[1], S[2], S[3]}, {S[2], S[3], S[4]}};
            double c[3];
            for (int e = 0; e < 3; ++e) {
                double M[3][3];
                for (int r = 0; r < 3; ++r)
                    for (int q = 0; q < 3; ++q) M[r][q] = q == e ? T[r] : A[r][q];
                c[e] = det3(M) / det3(A);
            }
            const double disc = std::sqrt(std::max(0.0, c[1] * c[1] - 4 * c[2] * c[0]));
            const double r1 = (-c[1] + disc) / (2 * c[2]), r2 = (-c[1] - disc) / (2 * c[2]);
            const double root = (c[1] + 2 * c[2] * r1 > 0) ? r1 : r2;
            rel = std::abs(root) / 1.0;  // phiCrit / phiCritRef - 1
        }
        hs.push_back(double(L) / n);
        errs.push_back(rel);
        out.push_back(measure("buckling threshold, n = " + std::to_string(n), rel, 0,
                              n == counts.back() ? 0.01 : 0.02, 1e-3));
    }
    out.push_back(measure("buckling threshold convergence order", logLogSlope(hs, errs), 1.7, 2.3,
                          0.2));
    return out;
}

// ---------------------------------------------------------------- energy
//
// Free flight for 1e5 steps at 2, 4, 8 and 16 substeps: dissipation, energy
// gain and momentum, as the CPU case measures them.
Measures scenarioEnergy(Backend b) {
    const RodMaterial mat = referenceMaterial();
    const Real L = Real(1);
    const int n = 32, steps = 100000, transient = 20000;
    const std::vector<int> substepList = {2, 4, 8, 16};

    struct Result {
        double dissipation = 0, gain = 0, dp = 0, dl = 0;
        bool ok = true;
    };
    std::vector<Result> results(substepList.size());
    auto runOne = [&](std::size_t si) {
        Rod rod = makeStraightRod(n, L, mat, Vec3(0, 0, 0), Vec3(1, 0, 0));
        for (std::size_t i = 0; i < rod.state.numParticles(); ++i)
            rod.state.v[i] = Vec3(0, 0, Real(0.1) * std::sin(2 * Real(kPi) * rod.state.x[i].x / L));
        auto sim = makeSim(b, rod, 1);
        if (!sim) {
            results[si].ok = false;
            return;
        }
        sim->setRod(0, rod);
        SolverParams p;
        p.dt = Real(1e-3);
        p.substeps = substepList[si];
        p.gravity = Vec3();

        const double e0 = rod.kineticEnergy() + rod.elasticEnergy();
        const Vec3 p0 = rod.linearMomentum(), l0 = rod.angularMomentum();
        double pScale = 0;
        for (std::size_t i = 0; i < rod.state.numParticles(); ++i)
            pScale += rod.state.mass[i] * norm(rod.state.v[i]);
        const double lScale = std::max(double(norm(l0)), pScale * L);

        Result& r = results[si];
        double eTransient = e0;
        for (int i = 0; i <= steps; i += 100) {
            if (i > 0) {
                sim->advance(p, 100);
                sim->get(0, rod);
            }
            const double total = rod.kineticEnergy() + rod.elasticEnergy();
            r.dp = std::max(r.dp, double(norm(rod.linearMomentum() - p0)) / pScale);
            r.dl = std::max(r.dl, double(norm(rod.angularMomentum() - l0)) / lScale);
            r.gain = std::max(r.gain, (total - e0) / e0);
            if (i == transient) eTransient = total;
        }
        r.dissipation = (e0 - eTransient) / e0;
    };
    // The CPU backend runs one rod per Sim, so run the substep counts side by
    // side; device batches go one at a time.
    if (b == Backend::kCpu) {
        std::vector<std::thread> pool;
        for (std::size_t si = 0; si < substepList.size(); ++si) pool.emplace_back(runOne, si);
        for (std::thread& t : pool) t.join();
    } else {
        for (std::size_t si = 0; si < substepList.size(); ++si) runOne(si);
    }

    for (const Result& r : results)
        if (!r.ok) return {};
    Measures out;
    for (std::size_t i = 0; i < results.size(); ++i) {
        const std::string at = " at " + std::to_string(substepList[i]) + " substeps";
        out.push_back(measure("energy lost over 20 s" + at, results[i].dissipation, 0, 1, -1));
        out.push_back(measure("largest energy gain" + at, results[i].gain, -1, 1e-3, -1));
        out.push_back(measure("linear momentum error" + at, results[i].dp, 0, 1e-8, -1));
        out.push_back(measure("angular momentum error" + at, results[i].dl, 0, 1e-3, -1));
    }
    return out;
}

// ---------------------------------------------------------------- contact
Measures scenarioContactPrimitives(Backend b) {
    const RodMaterial mat = referenceMaterial();
    const Real L = Real(0.4), r = mat.radius, mu = Real(0.8);
    const int n = 32;
    const Real sphereR = Real(0.25), capsuleR = Real(0.15), boxH = Real(0.2);
    struct Scenario {
        Primitive prim;
        Real supportZ;
    };
    const std::vector<Scenario> scenarios = {
        {Primitive::makePlane(Vec3(0, 0, 0), Vec3(0, 0, 1), mu), r},
        {Primitive::makeSphere(Vec3(0, 0, 0), sphereR, mu), sphereR + r},
        {Primitive::makeCapsule(Vec3(-1, 0, 0), Vec3(1, 0, 0), capsuleR, mu), capsuleR + r},
        {Primitive::makeBox(Vec3(0, 0, 0), Vec3(1, 1, boxH), Quat(), mu), boxH + r},
    };
    SolverParams p;
    p.dt = Real(2e-3);
    p.substeps = 16;
    p.iterations = 8;
    p.gravity = Vec3(0, 0, Real(-9.81));
    p.linearDamping = p.angularDamping = Real(12);

    double worstHeight = 0, worstPenetration = 0;
    for (const Scenario& sc : scenarios) {
        CollisionWorld world;
        world.primitives.push_back(sc.prim);
        Rod rod = makeStraightRod(n, L, mat, Vec3(-L / 2, 0, sc.supportZ + Real(0.004)),
                                  Vec3(1, 0, 0));
        auto sim = makeSim(b, rod, 1, &world);
        if (!sim) return {};
        sim->advance(p, 2000);  // the slowest, the sphere, settles on the CPU in 1102
        sim->get(0, rod);
        Real lowest = rod.state.x[0].z, penetration = 0;
        for (const Vec3& x : rod.state.x) {
            lowest = std::min(lowest, x.z);
            Vec3 nrm;
            penetration = std::max(penetration, r - signedDistance(sc.prim, x, nrm));
        }
        const double h = sc.prim.type == Primitive::kSphere ? rod.state.x[n / 2].z : lowest;
        worstHeight = std::max(worstHeight, std::abs(h - sc.supportZ) / sc.supportZ);
        worstPenetration = std::max(worstPenetration, double(penetration));
    }
    return {
        measure("rest height on four primitives (relative)", worstHeight, 0, 0.02, 1e-4),
        measure("residual penetration (m)", worstPenetration, -1, 1e-5, 1e-5),
    };
}

// Slip angle and sliding friction on an incline, as the CPU case measures them,
// at a given substep count.
Measures inclineAt(Backend b, int substeps) {
    const RodMaterial mat = referenceMaterial();
    const Real L = Real(0.3);
    const int n = 16;
    const double g = 9.81;
    double worstAngle = 0, worstMu = 0;
    for (double mu : {0.1, 0.3, 0.5, 0.8}) {
        const double alphaC = std::atan(mu);
        std::vector<double> alphas, accels;
        for (int k = 0; k <= 10; ++k) {
            const double alpha = alphaC * (0.7 + 0.09 * k);
            const Vec3 normal(-std::sin(alpha), 0, std::cos(alpha));
            const Vec3 downSlope(-std::cos(alpha), 0, -std::sin(alpha));
            CollisionWorld world;
            world.primitives.push_back(Primitive::makePlane(Vec3(0, 0, 0), normal, Real(mu)));
            Rod rod = makeStraightRod(n, L, mat, normal * mat.radius, downSlope);
            auto sim = makeSim(b, rod, 1, &world);
            if (!sim) return {};

            SolverParams p;
            p.dt = Real(1e-3);
            p.substeps = substeps;
            p.iterations = 4;
            p.gravity = Vec3(0, 0, Real(-g));
            SolverParams settle = p;
            settle.linearDamping = settle.angularDamping = Real(30);
            sim->advance(settle, 300);

            const int half = 200;
            const double T = 2 * half * double(p.dt);
            auto along = [&]() {
                sim->get(0, rod);
                return double(dot(rod.state.x[n / 2], downSlope));
            };
            const double s0 = along();
            sim->advance(p, half);
            const double s1 = along();
            sim->advance(p, half);
            const double s2 = along();
            const double accel = 4 * (s2 - 2 * s1 + s0) / (T * T);
            if (accel > 0.2) {
                alphas.push_back(alpha);
                accels.push_back(accel);
                const double muEff = (g * std::sin(alpha) - accel) / (g * std::cos(alpha));
                worstMu = std::max(worstMu, std::abs(muEff - mu) / mu);
            }
        }
        // Zero crossing of the least-squares line accel(alpha).
        double sx = 0, sy = 0, sxx = 0, sxy = 0;
        const double m = double(alphas.size());
        for (std::size_t i = 0; i < alphas.size(); ++i) {
            sx += alphas[i], sy += accels[i], sxx += alphas[i] * alphas[i];
            sxy += alphas[i] * accels[i];
        }
        const double slope = (m * sxy - sx * sy) / (m * sxx - sx * sx);
        const double slip = -((sy - slope * sx) / m) / slope;
        worstAngle = std::max(worstAngle, std::abs(slip - alphaC) / alphaC);
    }
    return {
        measure("slip angle vs atan(mu) (relative)", worstAngle, 0, 0.05, 0.04),
        measure("effective mu while sliding (relative)", worstMu, 0, 0.08, 0.04),
    };
}

// The CPU case's setting, and one whose substep float can resolve (see kScenarios).
Measures scenarioIncline(Backend b) { return inclineAt(b, 8); }
Measures scenarioIncline2(Backend b) { return inclineAt(b, 2); }

// Capstan: every tension ratio of a wrap angle, and at half a turn every
// friction coefficient too, in one batch.
Measures scenarioCapstan(Backend b) {
    RodMaterial mat;
    mat.youngs = Real(5e9);
    mat.poisson = Real(0.35);
    mat.density = Real(1e7);  // see ropeMaterial() in contact_cases.cpp
    mat.radius = Real(2.5e-4);
    const Real R = Real(0.05), T1 = Real(1), lead = Real(0.03), segLen = Real(1e-3);
    const Real baseMu = Real(0.25);
    const int ratiosPerMu = 10;

    // Relative error of the critical ratio for each of `mus` at `turns`; empty
    // if the batch could not be created.
    auto criticalRatioErrors = [&](double turns,
                                   const std::vector<double>& mus) -> std::vector<double> {
        const double theta = 2 * kPi * turns;
        const Real Rc = R + mat.radius;
        auto arcPoint = [&](double phi) { return Vec3(Rc * std::cos(phi), Rc * std::sin(phi), 0); };
        auto tangent = [&](double phi) { return Vec3(-std::sin(phi), std::cos(phi), 0); };
        std::vector<Vec3> pts;
        const int leadN = std::max(2, int(lead / segLen));
        for (int i = leadN; i >= 1; --i) pts.push_back(arcPoint(0) - tangent(0) * (segLen * i));
        const int arcN = std::max(4, int(Rc * theta / segLen));
        for (int i = 0; i <= arcN; ++i) pts.push_back(arcPoint(theta * i / arcN));
        for (int i = 1; i <= leadN; ++i)
            pts.push_back(arcPoint(theta) + tangent(theta) * (segLen * i));

        Rod prototype = makeRodFromCenterline(pts, mat, Vec3(0, 0, 1));
        for (std::size_t i = 0; i < prototype.bend.size(); ++i)
            prototype.bend.restDarboux[i] = Vec3();
        CollisionWorld world;
        world.primitives.push_back(Primitive::makeCapsule(Vec3(0, 0, -1), Vec3(0, 0, 1), R, baseMu));

        const int rods = int(mus.size()) * ratiosPerMu;
        std::vector<Rod> loaded;
        std::vector<float> scales;
        std::vector<double> ratios;
        int substeps = 8;
        for (double mu : mus)
            for (int k = 0; k < ratiosPerMu; ++k) {
                const double ratio = std::exp(mu * theta) * (0.70 + 0.12 * k);
                Rod rod = prototype;
                rod.state.extForce.front() = -tangent(0) * T1;
                rod.state.extForce.back() = tangent(theta) * Real(T1 * ratio);
                substeps = std::max(substeps, std::min(64, stableSubsteps(rod, Vec3(), Real(5e-4))));
                loaded.push_back(rod);
                scales.push_back(float(mu / baseMu));
                ratios.push_back(ratio);
            }

        auto sim = makeSim(b, prototype, rods, &world);
        if (!sim) return {};
        for (int r = 0; r < rods; ++r) sim->setRod(r, loaded[r]);
        sim->setFrictionScales(scales);

        SolverParams p;
        p.dt = Real(5e-4);
        p.substeps = substeps;
        p.iterations = 6;
        p.gravity = Vec3();
        p.linearDamping = p.angularDamping = Real(40);
        sim->advance(p, 400);

        const Vec3 midTangent(-std::sin(theta / 2), std::cos(theta / 2), 0);
        const int mid = int(pts.size()) / 2;
        std::vector<Vec3> before(rods);
        Rod rod = prototype;
        for (int r = 0; r < rods; ++r) {
            sim->get(r, rod);
            before[r] = rod.state.x[mid];
        }
        sim->advance(p, 400);

        std::vector<double> errors;
        for (std::size_t m = 0; m < mus.size(); ++m) {
            std::vector<double> speeds;
            for (int k = 0; k < ratiosPerMu; ++k) {
                const int r = int(m) * ratiosPerMu + k;
                sim->get(r, rod);
                speeds.push_back(double(dot(rod.state.x[mid] - before[r], midTangent)) /
                                 (400 * double(p.dt)));
            }
            // Fit only the points clearly sliding, as the CPU case does.
            const double fastest = *std::max_element(speeds.begin(), speeds.end());
            double sx = 0, sy = 0, sxx = 0, sxy = 0, cnt = 0;
            for (int k = 0; k < ratiosPerMu; ++k) {
                if (speeds[k] <= 0.05 * fastest) continue;
                const double x = ratios[m * ratiosPerMu + k];
                sx += x, sy += speeds[k], sxx += x * x, sxy += x * speeds[k], ++cnt;
            }
            const double slope = (cnt * sxy - sx * sy) / (cnt * sxx - sx * sx);
            const double critical = -((sy - slope * sx) / cnt) / slope;
            const double capstan = std::exp(mus[m] * theta);
            errors.push_back(std::abs(critical - capstan) / capstan);
        }
        return errors;
    };

    const std::vector<double> half = criticalRatioErrors(0.5, {0.1, 0.25, 0.5, 0.75});
    if (half.empty()) return {};
    double wraps = half[1], mus = 0;
    for (double e : half) mus = std::max(mus, e);
    for (double turns : {0.25, 0.75, 1.0}) {
        const std::vector<double> e = criticalRatioErrors(turns, {0.25});
        if (e.empty()) return {};
        wraps = std::max(wraps, e[0]);
    }
    return {
        measure("capstan ratio, 0.25-1 turn at mu 0.25", wraps, 0, 0.06, 0.01),
        measure("capstan ratio, half turn at mu 0.1-0.75", mus, 0, 0.06, 0.01),
    };
}

// A rope dropped end-first coils into a pile: nothing may pass through
// anything, and the strands must actually meet.
Measures scenarioSelfCollision(Backend b) {
    RodMaterial mat = referenceMaterial();
    mat.youngs = Real(2e6);
    const int n = 150;
    const Real L = Real(1.2);
    const double diameter = 2 * double(mat.radius);
    Rod rod = makeStraightRod(n, L, mat, Vec3(0, 0, Real(0.02)),
                              normalize(Vec3(Real(0.08), Real(0.03), 1)));
    for (Vec3& x : rod.state.x) x.z = L + Real(0.05) - x.z;
    rod.state.xPrev = rod.state.x;
    setRestFromCurrent(rod);

    CollisionWorld world;
    world.primitives.push_back(Primitive::makePlane(Vec3(0, 0, 0), Vec3(0, 0, 1), Real(0.6)));
    world.selfCollision = true;
    world.selfFriction = Real(0.3);
    world.hashTableSize = 256;  // fits a fused block; the CPU uses the same table

    SolverParams p;
    p.dt = Real(1e-3);
    p.substeps = 16;
    p.iterations = 4;
    p.gravity = Vec3(0, 0, Real(-9.81));
    p.linearDamping = p.angularDamping = Real(1);

    auto sim = makeSim(b, rod, 1, &world);
    if (!sim) return {};
    const int gap = selfCollisionIndexGap(rod, world);
    // "In contact": closer than the diameter plus the contact margin.
    const double touching = diameter + double(world.contactMarginRadii * mat.radius);
    double worstOverlap = 0;
    int peakPairs = 0;
    for (int i = 0; i <= 2500; i += 25) {
        if (i > 0) sim->advance(p, 25);
        sim->get(0, rod);
        double minDist = 1e30;
        int pairs = 0;
        for (int a = 0; a < n; ++a)
            for (int c = a + gap; c < n; ++c) {
                Real u, v;
                const Vec3 a0 = rod.state.x[a], a1 = rod.state.x[a + 1];
                const Vec3 b0 = rod.state.x[c], b1 = rod.state.x[c + 1];
                closestPointsBetweenSegments(a0, a1, b0, b1, u, v);
                const double d = norm((a0 + (a1 - a0) * u) - (b0 + (b1 - b0) * v));
                minDist = std::min(minDist, d);
                pairs += d < touching ? 1 : 0;
            }
        peakPairs = std::max(peakPairs, pairs);
        worstOverlap = std::max(worstOverlap, (diameter - minDist) / diameter);
    }
    Measures out = {
        measure("self-interpenetration (/diameter)", std::max(0.0, worstOverlap), 0, 0.05, -1),
        measure("peak touching strand pairs", peakPairs, 10, 1e9, -1),
    };
    // The CPU's contact list is unbounded; the device pool is not.
    double overflow = 0;
#ifdef CRS_WITH_CUDA
    if (b == Backend::kGpu)
        overflow = double(static_cast<GpuSim*>(sim.get())->selfContactOverflow());
#endif
    out.push_back(measure("self-contact overflow", overflow, 0, 0, -1));
    return out;
}

struct Scenario {
    const char* name;
    Measures (*run)(Backend);
    // Null: the GPU is held to the CPU case's tolerances and to the CPU.
    // Otherwise the scenario is outside what single precision (or the GPU's
    // colour order) can reproduce: its numbers are reported, not checked, and
    // this says why and what showed it.
    const char* limit;
};

const Scenario kScenarios[] = {
    {"helix", scenarioHelix, nullptr},
    {"contact", scenarioContactPrimitives, nullptr},
    {"self-collision", scenarioSelfCollision, nullptr},
    {"incline-2", scenarioIncline2, nullptr},
    {"incline", scenarioIncline, "float: at 8 substeps gravity pushes a resting rod into the "
                                 "plane by g h^2 = 1.5e-7 m per substep, a few float steps at "
                                 "1 m; the float CPU build misses too (slip angle 9.9%, mu 26%) "
                                 "and passes at 2 substeps (incline-2)"},
    {"tip-load", scenarioTipLoad, "float: the damped XPBD relaxation (n^2 sweeps per step) does "
                                  "not settle in single precision; the float CPU build ends 0.81 L "
                                  "from the direct solve"},
    {"end-moment", scenarioEndMoment, "float: as tip-load; the float CPU build's arc radius is 31x "
                                      "the direct solve's"},
    {"twist-buckling", scenarioTwistBuckling,
     "float: at 16 n substeps per ms a substep moves the rod ~1e-9 m, below float spacing; the "
     "float CPU build's thresholds are 6-8% off with no convergence"},
    {"energy", scenarioEnergy, "float: 100 s of undamped free flight accumulates rounding; the "
                               "float CPU build gains 0.10 of the energy at 2 substeps and "
                               "5.4e3 at 16"},
    {"capstan", scenarioCapstan,
     "colour order: with red-black stretch sweeps a wrapped rope slides below exp(mu theta) "
     "even in double (the colour-ordered double CPU fails identically, sequential sweeps pass "
     "at 1%, and 24 or 96 sweeps change nothing)"},
};

std::string line(const char* f, const std::string& s, double a, double b = 0, double c = 0) {
    char buf[512];
    std::snprintf(buf, sizeof(buf), f, s.c_str(), a, b, c);
    return buf;
}

}  // namespace

CaseResult runGpuSuite(const std::string& outDir) {
    CaseResult res;
    res.name = "Phase 1-2 validation suite on the GPU path";

#ifdef CRS_WITH_CUDA
    const bool haveGpu = gpu::cudaAvailable();
    const std::string why = haveGpu ? "" : gpu::diagnostic();
#else
    const bool haveGpu = false;
    const std::string why = "built without CUDA";
#endif
    const bool cpuOnly = !haveGpu && std::getenv("CRS_GPU_SUITE_CPU") != nullptr;
    if (!haveGpu && !cpuOnly) {
        res.skipped = true;
        res.skipReason = why + " (CRS_GPU_SUITE_CPU=1 runs the CPU half)";
        return res;
    }
#ifdef CRS_WITH_CUDA
    if (haveGpu) res.notes.push_back("  device: " + gpu::deviceName());
#endif
    if (cpuOnly)
        res.notes.push_back(sizeof(Real) == 4 ? "  CPU half only, single precision"
                                              : "  CPU half only, double precision");

    Csv csv(outDir, "gpu_suite.csv", "scenario,quantity,gpu,cpu,gpu_minus_cpu,lo,hi,agree,limited");
    int limited = 0;
    using Clock = std::chrono::steady_clock;
    auto seconds = [](Clock::time_point a, Clock::time_point b) {
        return std::chrono::duration<double>(b - a).count();
    };

    // Debugging aids: CRS_GPU_SUITE_ONLY=<a,b,...> runs only those scenarios, and
    // CRS_GPU_SUITE_NO_CPU=1 skips the CPU half (and with it the agreement checks).
    const char* only = std::getenv("CRS_GPU_SUITE_ONLY");
    const bool noCpu = haveGpu && std::getenv("CRS_GPU_SUITE_NO_CPU") != nullptr;
    for (const Scenario& sc : kScenarios) {
        if (only && ("," + std::string(only) + ",").find("," + std::string(sc.name) + ",") ==
                        std::string::npos)
            continue;
        std::fprintf(stderr, "gpu-suite: %s\n", sc.name);
        const auto t0 = Clock::now();
        const Measures g = haveGpu ? sc.run(Backend::kGpu) : Measures();
        const auto t1 = Clock::now();
        const Measures c = noCpu ? g : sc.run(Backend::kCpu);
        const auto t2 = Clock::now();
        res.notes.push_back(line("  %s: %.1f s on the GPU, %.1f s on the CPU", sc.name,
                                 seconds(t0, t1), seconds(t1, t2)));
        if (haveGpu && g.size() != c.size()) {
            res.checks.push_back(makeCheck(std::string(sc.name) + ": GPU batch created", 0, 1, 0));
            continue;
        }
        if (sc.limit) {
            res.notes.push_back(std::string("    [LIMIT] ") + sc.limit);
            ++limited;
        }
        for (std::size_t i = 0; i < c.size(); ++i) {
            const Measure& m = haveGpu ? g[i] : c[i];
            if (haveGpu) {
                const double diff = g[i].value - c[i].value;
                res.notes.push_back(line("    %-46s GPU %.6g  CPU %.6g  (diff %.3g)", m.name,
                                         g[i].value, c[i].value, diff));
                csv.row(sc.name, m.name, g[i].value, c[i].value, diff, m.lo, m.hi, m.agree,
                        sc.limit ? 1 : 0);
                if (m.agree >= 0 && !noCpu && !sc.limit)
                    res.checks.push_back(makeCheck(m.name + ": GPU - CPU", std::abs(diff), 0.0,
                                                   m.agree, 1.0));
            } else {
                res.notes.push_back(line("    %-46s CPU %.9g", m.name, m.value));
            }
            if (!sc.limit) res.checks.push_back(makeRangeCheck(m.name, m.value, m.lo, m.hi));
        }
    }
    if (limited > 0)
        res.notes.push_back(line("  %s%.0f scenarios reported, not validated: see [LIMIT]", "",
                                 double(limited)));
    return res;
}

}  // namespace crs
