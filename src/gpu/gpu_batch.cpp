// Host orchestration for the GPU solver: allocation, layout, upload/download.
// Compiled by the host compiler, not nvcc -- everything that touches the device
// goes through the thin shims in gpu_kernels.h.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "gpu_kernels.h"
#include "gpu_solver.h"

namespace crs {
namespace gpu {

namespace {

Vec3f toF(Vec3 v) { return Vec3f(float(v.x), float(v.y), float(v.z)); }
Quatf toF(Quat q) { return Quatf(float(q.w), float(q.x), float(q.y), float(q.z)); }
Vec3 toD(Vec3f v) { return Vec3(v.x, v.y, v.z); }
Quat toD(Quatf q) { return Quat(q.w, q.x, q.y, q.z); }

template <typename T>
T* uploadArray(const std::vector<T>& host, std::vector<void*>& owned) {
    if (host.empty()) return nullptr;
    void* p = deviceMalloc(host.size() * sizeof(T));
    if (!p) return nullptr;
    copyToDevice(p, host.data(), host.size() * sizeof(T));
    owned.push_back(p);
    return static_cast<T*>(p);
}

// Shared-memory footprint of one rod under the fused strategy: two quaternion
// arrays per segment; position, previous position, velocity, force and
// kinematic velocity per particle; angular velocity and torque per segment; one
// multiplier per constraint; one contact slot per particle per primitive.
std::size_t fusedSharedBytes(int numParticles, int numSegments, int numStretch, int numBend,
                             int numPrims = 0, int selfTableSize = -1, int selfCapacity = 0) {
    std::size_t bytes =
        2 * std::size_t(numSegments) * sizeof(Quatf) +
        (5 * std::size_t(numParticles) + 2 * std::size_t(numSegments) + numStretch + numBend) *
            sizeof(Vec3f) +
        std::size_t(numParticles) * numPrims * sizeof(DevContactF);
    if (selfTableSize >= 0)  // self-collision: per-segment scratch, table, pool
        bytes += std::size_t(numStretch) * (sizeof(Vec3f) + 4 * sizeof(int)) + 2 * sizeof(int) +
                 (std::size_t(selfTableSize) + 1) * sizeof(int) + sizeof(float) +
                 std::size_t(selfCapacity) * sizeof(DevSelfContactF);
    return bytes;
}

Primitivef toF(const Primitive& p) {
    Primitivef f;
    f.type = p.type;
    f.a = toF(p.a);
    f.b = toF(p.b);
    f.normal = toF(p.normal);
    f.halfExtents = toF(p.halfExtents);
    f.rotation = toF(p.rotation);
    f.radius = float(p.radius);
    f.friction = float(p.friction);
    return f;
}

std::size_t particleAt(bool rodMajor, int rod, int i, int numParticles, int numRods) {
    return rodMajor ? std::size_t(rod) * numParticles + i : std::size_t(i) * numRods + rod;
}

std::size_t segmentAt(bool rodMajor, int rod, int j, int numSegments, int numRods) {
    return rodMajor ? std::size_t(rod) * numSegments + j : std::size_t(j) * numRods + rod;
}

}  // namespace

bool cudaAvailable() { return deviceCount() > 0; }

std::string diagnostic() {
    int rt = 0, dr = 0;
    const char* msg = "";
    const int err = deviceProbe(&rt, &dr, &msg);
    if (err == 0 && deviceCount() > 0) return "";
    // When the runtime cannot initialize at all, the version queries fail too and
    // report 0 -- print that as "unavailable", not as a real version 0.0.
    char versions[128];
    if (rt == 0 && dr == 0)
        std::snprintf(versions, sizeof(versions), "runtime and driver versions unavailable");
    else
        std::snprintf(versions, sizeof(versions), "runtime CUDA %d.%d, driver supports CUDA %d.%d",
                      rt / 1000, (rt % 1000) / 10, dr / 1000, (dr % 1000) / 10);
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "no usable CUDA device: error %d (%s); %s. The kernels compile; the most likely "
                  "cause is a driver older than the CUDA runtime they were built against (compare "
                  "`nvidia-smi` with the toolkit version).",
                  err, (msg && *msg) ? msg : "no message", versions);
    return buf;
}

std::string deviceName() {
    char name[256] = {0};
    int major = 0, minor = 0, sms = 0;
    std::size_t shared = 0;
    if (!deviceInfo(name, sizeof(name), &major, &minor, &sms, &shared)) return "unknown";
    char buf[384];
    std::snprintf(buf, sizeof(buf), "%s (sm_%d%d, %d SMs, %zu KB shared/block)", name, major, minor,
                  sms, shared / 1024);
    return buf;
}

int maxFusedSegments() {
    char name[256] = {0};
    int major = 0, minor = 0, sms = 0;
    std::size_t shared = 0;
    if (!deviceInfo(name, sizeof(name), &major, &minor, &sms, &shared)) return 0;
    // Solve fusedSharedBytes(n + 1, n, n, n - 1) <= shared for n.
    int best = 0;
    for (int n = 4; n <= 4096; ++n)
        if (fusedSharedBytes(n + 1, n, n, n - 1) <= shared) best = n;
    return best;
}

struct Batch::Impl {
    DevRodF rod;
    DevStateF state;
    std::vector<void*> owned;
    int fusedThreads = 128;

    ~Impl() {
        for (void* p : owned) deviceFree(p);
    }
};

Batch::~Batch() { destroy(); }

void Batch::destroy() {
    delete impl_;
    impl_ = nullptr;
    numRods_ = 0;
}

int Batch::launchesPerStep(const BatchParams& params) const {
    // The number this whole design is trying to shrink. The fused path issues
    // one launch for an entire step no matter how many substeps, sweeps or
    // colours it contains; the multi-kernel path pays per colour per sweep.
    if (strategy_ == Strategy::kFused) return 1;
    const int contactKinds = (numPrims_ > 0 ? 1 : 0) + (selfCollision_ ? 1 : 0);
    const int perSweep = stretchColoring_.numColors() + bendColoring_.numColors() + contactKinds;
    return params.substeps * (3 + params.iterations * perSweep + 1 + contactKinds);
}

bool Batch::create(const Rod& prototype, int numRods, Strategy strategy,
                   const CollisionWorld* world) {
    destroy();
    if (!cudaAvailable()) return false;
    const int numPrims = world ? static_cast<int>(world->primitives.size()) : 0;
    numPrims_ = numPrims;
    selfCollision_ = world && world->selfCollision;

    numRods_ = numRods;
    numParticles_ = static_cast<int>(prototype.state.numParticles());
    numSegments_ = static_cast<int>(prototype.state.numSegments());
    strategy_ = strategy;
    stretchColoring_ = colorStretchConstraints(prototype);
    bendColoring_ = colorBendConstraints(prototype);

    const int nStretch = static_cast<int>(prototype.stretch.size());
    const int nBend = static_cast<int>(prototype.bend.size());
    // Self contacts per rod: half the segment count, against a measured peak of
    // 37 for a 150-segment rope coiling into a pile. Overflow is counted.
    const int selfCapacity = selfCollision_ ? std::max(16, nStretch / 2) : 0;
    sharedBytes_ = fusedSharedBytes(numParticles_, numSegments_, nStretch, nBend, numPrims,
                                    selfCollision_ ? world->hashTableSize : -1, selfCapacity);

    if (strategy == Strategy::kFused) {
        char name[256] = {0};
        int major = 0, minor = 0, sms = 0;
        std::size_t shared = 0;
        deviceInfo(name, sizeof(name), &major, &minor, &sms, &shared);
        if (sharedBytes_ > shared) {
            numRods_ = 0;
            return false;
        }
    }

    impl_ = new Impl();
    Impl& im = *impl_;
    DevRodF& d = im.rod;
    d.numRods = numRods;
    d.numParticles = numParticles_;
    d.numSegments = numSegments_;
    d.numStretch = nStretch;
    d.numBend = nBend;

    std::vector<float> invMass(numParticles_);
    for (int i = 0; i < numParticles_; ++i) invMass[i] = float(prototype.state.invMass[i]);
    std::vector<Vec3f> invInertia(numSegments_), inertia(numSegments_);
    for (int j = 0; j < numSegments_; ++j) {
        invInertia[j] = toF(prototype.state.invInertia[j]);
        inertia[j] = toF(prototype.state.inertia[j]);
    }
    std::vector<float> sRest(nStretch), bRest(nBend);
    std::vector<Vec3f> sComp(nStretch), bComp(nBend), bRestD(nBend);
    for (int k = 0; k < nStretch; ++k) {
        sRest[k] = float(prototype.stretch.restLength[k]);
        sComp[k] = toF(prototype.stretch.compliance[k]);
    }
    for (int k = 0; k < nBend; ++k) {
        bRest[k] = float(prototype.bend.restLength[k]);
        bComp[k] = toF(prototype.bend.compliance[k]);
        bRestD[k] = toF(prototype.bend.restDarboux[k]);
    }

    d.invMass = uploadArray(invMass, im.owned);
    d.invInertia = uploadArray(invInertia, im.owned);
    d.inertia = uploadArray(inertia, im.owned);
    d.sP0 = uploadArray(prototype.stretch.p0, im.owned);
    d.sP1 = uploadArray(prototype.stretch.p1, im.owned);
    d.sSeg = uploadArray(prototype.stretch.seg, im.owned);
    d.sRest = uploadArray(sRest, im.owned);
    d.sCompliance = uploadArray(sComp, im.owned);
    d.bA = uploadArray(prototype.bend.segA, im.owned);
    d.bB = uploadArray(prototype.bend.segB, im.owned);
    d.bRest = uploadArray(bRest, im.owned);
    d.bCompliance = uploadArray(bComp, im.owned);
    d.bRestDarboux = uploadArray(bRestD, im.owned);
    d.sOrder = uploadArray(stretchColoring_.order, im.owned);
    d.sColorStart = uploadArray(stretchColoring_.colorStart, im.owned);
    d.sNumColors = stretchColoring_.numColors();
    d.bOrder = uploadArray(bendColoring_.order, im.owned);
    d.bColorStart = uploadArray(bendColoring_.colorStart, im.owned);
    d.bNumColors = bendColoring_.numColors();
    d.numPrims = numPrims;
    d.radius = float(prototype.material.radius);
    if (numPrims > 0) {
        std::vector<Primitivef> prims;
        for (const Primitive& p : world->primitives) prims.push_back(toF(p));
        d.prims = uploadArray(prims, im.owned);
    }
    if (selfCollision_) {
        d.selfCollision = 1;
        d.selfGap = selfCollisionIndexGap(prototype, *world);
        d.selfFriction = float(world->selfFriction);
        d.hashTableSize = world->hashTableSize;
        d.selfCapacity = selfCapacity;
    }

    const std::size_t pCount = std::size_t(numParticles_) * numRods;
    const std::size_t sCount = std::size_t(numSegments_) * numRods;
    auto alloc = [&](std::size_t bytes) {
        void* p = deviceMalloc(bytes);
        if (p) im.owned.push_back(p);
        return p;
    };
    im.state.x = static_cast<Vec3f*>(alloc(pCount * sizeof(Vec3f)));
    im.state.xPrev = static_cast<Vec3f*>(alloc(pCount * sizeof(Vec3f)));
    im.state.v = static_cast<Vec3f*>(alloc(pCount * sizeof(Vec3f)));
    im.state.q = static_cast<Quatf*>(alloc(sCount * sizeof(Quatf)));
    im.state.qPrev = static_cast<Quatf*>(alloc(sCount * sizeof(Quatf)));
    im.state.omega = static_cast<Vec3f*>(alloc(sCount * sizeof(Vec3f)));
    im.state.lamS = static_cast<Vec3f*>(alloc(std::size_t(nStretch) * numRods * sizeof(Vec3f)));
    im.state.lamB = static_cast<Vec3f*>(alloc(std::size_t(nBend) * numRods * sizeof(Vec3f)));
    im.state.force = static_cast<Vec3f*>(alloc(pCount * sizeof(Vec3f)));
    im.state.torque = static_cast<Vec3f*>(alloc(sCount * sizeof(Vec3f)));
    im.state.kinematicVelocity = static_cast<Vec3f*>(alloc(pCount * sizeof(Vec3f)));
    im.state.complianceScale = static_cast<float*>(alloc(std::size_t(numRods) * sizeof(float)));
    im.state.frictionScale = static_cast<float*>(alloc(std::size_t(numRods) * sizeof(float)));
    if (im.state.complianceScale && im.state.frictionScale) {
        const std::vector<float> ones(numRods, 1.0f);
        copyToDevice(im.state.complianceScale, ones.data(), numRods * sizeof(float));
        copyToDevice(im.state.frictionScale, ones.data(), numRods * sizeof(float));
    }
    if (numPrims > 0) {
        im.state.contacts =
            static_cast<DevContactF*>(alloc(pCount * numPrims * sizeof(DevContactF)));
        if (!im.state.contacts) {
            destroy();
            return false;
        }
    }
    if (selfCollision_) {
        const std::size_t rods = std::size_t(numRods);
        auto allocInts = [&](std::size_t count) {
            return static_cast<int*>(alloc(count * sizeof(int)));
        };
        im.state.selfCentres = static_cast<Vec3f*>(alloc(rods * nStretch * sizeof(Vec3f)));
        im.state.selfCellOf = allocInts(rods * nStretch);
        im.state.selfSorted = allocInts(rods * nStretch);
        im.state.selfCellStart = allocInts(rods * (std::size_t(d.hashTableSize) + 1));
        im.state.selfPool = static_cast<DevSelfContactF*>(
            alloc(rods * std::size_t(selfCapacity) * sizeof(DevSelfContactF)));
        im.state.selfSegCount = allocInts(rods * nStretch);
        im.state.selfSegStart = allocInts(rods * nStretch);
        im.state.selfLive = allocInts(rods);
        im.state.selfCellSize = static_cast<float*>(alloc(rods * sizeof(float)));
        im.state.selfOverflow = allocInts(rods);
        if (!im.state.selfCentres || !im.state.selfCellOf || !im.state.selfSorted ||
            !im.state.selfCellStart || !im.state.selfPool || !im.state.selfSegCount ||
            !im.state.selfSegStart || !im.state.selfLive || !im.state.selfCellSize ||
            !im.state.selfOverflow) {
            destroy();
            return false;
        }
        const std::vector<int> zeros(rods * nStretch, 0);
        copyToDevice(im.state.selfSegCount, zeros.data(), rods * nStretch * sizeof(int));
        copyToDevice(im.state.selfLive, zeros.data(), rods * sizeof(int));
        copyToDevice(im.state.selfOverflow, zeros.data(), rods * sizeof(int));
    }
    if (!im.state.x || !im.state.q || !im.state.force || !im.state.torque ||
        !im.state.kinematicVelocity || !im.state.complianceScale || !im.state.frictionScale) {
        destroy();
        return false;
    }

    this->upload(prototype);
    return true;
}

void Batch::upload(const Rod& in) {
    Impl& im = *impl_;
    const bool rodMajor = strategy_ == Strategy::kFused;
    const std::size_t pCount = std::size_t(numParticles_) * numRods_;
    const std::size_t sCount = std::size_t(numSegments_) * numRods_;

    std::vector<Vec3f> hx(pCount), hv(pCount), hforce(pCount), hkin(pCount), homega(sCount),
                       htorque(sCount);
    std::vector<Quatf> hq(sCount);

    for (int r = 0; r < numRods_; ++r) {
        for (int i = 0; i < numParticles_; ++i) {
            const std::size_t at = particleAt(rodMajor, r, i, numParticles_, numRods_);
            hx[at] = toF(in.state.x[i]);
            hv[at] = toF(in.state.v[i]);
            hforce[at] = toF(in.state.extForce[i]);
            hkin[at] = toF(in.state.kinematicVelocity[i]);
        }
        for (int j = 0; j < numSegments_; ++j) {
            const std::size_t at = segmentAt(rodMajor, r, j, numSegments_, numRods_);
            hq[at] = toF(in.state.q[j]);
            homega[at] = toF(in.state.omega[j]);
            htorque[at] = toF(in.state.extTorque[j]);
        }
    }

    copyToDevice(im.state.x, hx.data(), pCount * sizeof(Vec3f));
    copyToDevice(im.state.v, hv.data(), pCount * sizeof(Vec3f));
    copyToDevice(im.state.force, hforce.data(), pCount * sizeof(Vec3f));
    copyToDevice(im.state.kinematicVelocity, hkin.data(), pCount * sizeof(Vec3f));
    copyToDevice(im.state.q, hq.data(), sCount * sizeof(Quatf));
    copyToDevice(im.state.omega, homega.data(), sCount * sizeof(Vec3f));
    copyToDevice(im.state.torque, htorque.data(), sCount * sizeof(Vec3f));
}

void Batch::uploadRod(int rodIndex, const Rod& source) {
    Impl& im = *impl_;
    const bool rodMajor = strategy_ == Strategy::kFused;
    for (int i = 0; i < numParticles_; ++i) {
        const std::size_t at = particleAt(rodMajor, rodIndex, i, numParticles_, numRods_);
        const Vec3f x = toF(source.state.x[i]), v = toF(source.state.v[i]);
        copyToDevice(im.state.x + at, &x, sizeof(Vec3f));
        copyToDevice(im.state.v + at, &v, sizeof(Vec3f));
    }
    for (int j = 0; j < numSegments_; ++j) {
        const std::size_t at = segmentAt(rodMajor, rodIndex, j, numSegments_, numRods_);
        const Quatf q = toF(source.state.q[j]);
        const Vec3f w = toF(source.state.omega[j]);
        copyToDevice(im.state.q + at, &q, sizeof(Quatf));
        copyToDevice(im.state.omega + at, &w, sizeof(Vec3f));
    }
    setLoads(rodIndex, source);
}

void Batch::setLoads(int rodIndex, const Rod& source) {
    Impl& im = *impl_;
    const bool rodMajor = strategy_ == Strategy::kFused;
    for (int i = 0; i < numParticles_; ++i) {
        const std::size_t at = particleAt(rodMajor, rodIndex, i, numParticles_, numRods_);
        const Vec3f f = toF(source.state.extForce[i]);
        const Vec3f k = toF(source.state.kinematicVelocity[i]);
        copyToDevice(im.state.force + at, &f, sizeof(Vec3f));
        copyToDevice(im.state.kinematicVelocity + at, &k, sizeof(Vec3f));
    }
    for (int j = 0; j < numSegments_; ++j) {
        const std::size_t at = segmentAt(rodMajor, rodIndex, j, numSegments_, numRods_);
        const Vec3f t = toF(source.state.extTorque[j]);
        copyToDevice(im.state.torque + at, &t, sizeof(Vec3f));
        if (norm2(source.state.invInertia[j]) == Real(0)) {
            const Quatf q = toF(source.state.q[j]);
            copyToDevice(im.state.q + at, &q, sizeof(Quatf));
        }
    }
}

void Batch::download(int rodIndex, Rod& out) const {
    const Impl& im = *impl_;
    const bool rodMajor = strategy_ == Strategy::kFused;
    const std::size_t pCount = std::size_t(numParticles_) * numRods_;
    const std::size_t sCount = std::size_t(numSegments_) * numRods_;

    std::vector<Vec3f> hx(pCount), hv(pCount), homega(sCount);
    std::vector<Quatf> hq(sCount);
    copyToHost(hx.data(), im.state.x, pCount * sizeof(Vec3f));
    copyToHost(hv.data(), im.state.v, pCount * sizeof(Vec3f));
    copyToHost(hq.data(), im.state.q, sCount * sizeof(Quatf));
    copyToHost(homega.data(), im.state.omega, sCount * sizeof(Vec3f));

    for (int i = 0; i < numParticles_; ++i) {
        const std::size_t at = particleAt(rodMajor, rodIndex, i, numParticles_, numRods_);
        out.state.x[i] = toD(hx[at]);
        out.state.v[i] = toD(hv[at]);
    }
    for (int j = 0; j < numSegments_; ++j) {
        const std::size_t at = segmentAt(rodMajor, rodIndex, j, numSegments_, numRods_);
        out.state.q[j] = toD(hq[at]);
        out.state.omega[j] = toD(homega[at]);
    }
}

void Batch::setKinematicVelocities(const std::vector<Vec3f>& rodMajor) {
    const bool rodMajorLayout = strategy_ == Strategy::kFused;
    const std::size_t pCount = std::size_t(numParticles_) * numRods_;
    if (rodMajor.size() != pCount) return;
    if (rodMajorLayout) {
        copyToDevice(impl_->state.kinematicVelocity, rodMajor.data(), pCount * sizeof(Vec3f));
        return;
    }
    std::vector<Vec3f> layout(pCount);
    for (int r = 0; r < numRods_; ++r)
        for (int i = 0; i < numParticles_; ++i)
            layout[particleAt(false, r, i, numParticles_, numRods_)] =
                rodMajor[std::size_t(r) * numParticles_ + i];
    copyToDevice(impl_->state.kinematicVelocity, layout.data(), pCount * sizeof(Vec3f));
}

void Batch::setMaterialScales(const std::vector<float>& youngsScale,
                              const std::vector<float>& frictionScale) {
    if (youngsScale.size() == std::size_t(numRods_)) {
        std::vector<float> compliance(numRods_);
        for (int r = 0; r < numRods_; ++r) compliance[r] = 1.0f / youngsScale[r];
        copyToDevice(impl_->state.complianceScale, compliance.data(), numRods_ * sizeof(float));
    }
    if (frictionScale.size() == std::size_t(numRods_))
        copyToDevice(impl_->state.frictionScale, frictionScale.data(), numRods_ * sizeof(float));
}

void Batch::downloadPositions(std::vector<Vec3f>& out) const {
    const bool rodMajor = strategy_ == Strategy::kFused;
    const std::size_t pCount = std::size_t(numParticles_) * numRods_;
    std::vector<Vec3f> raw(pCount);
    copyToHost(raw.data(), impl_->state.x, pCount * sizeof(Vec3f));
    out.resize(pCount);
    for (int r = 0; r < numRods_; ++r)
        for (int i = 0; i < numParticles_; ++i)
            out[std::size_t(r) * numParticles_ + i] =
                raw[particleAt(rodMajor, r, i, numParticles_, numRods_)];
}

void Batch::step(const BatchParams& params) {
    Impl& im = *impl_;
    StepConfigF cfg;
    cfg.h = params.dt / params.substeps;
    cfg.substeps = params.substeps;
    cfg.iterations = params.iterations;
    cfg.gravity = Vec3f(params.gravityX, params.gravityY, params.gravityZ);
    cfg.linDecay = std::exp(-params.linearDamping * cfg.h);
    cfg.angDecay = std::exp(-params.angularDamping * cfg.h);
    cfg.contactInterval = params.contactInterval > 0 ? params.contactInterval : 1;

    if (strategy_ == Strategy::kFused)
        launchFusedStep(im.rod, im.state, cfg, sharedBytes_, im.fusedThreads);
    else
        launchMultiKernelStep(im.rod, im.state, cfg, stretchColoring_.colorStart.data(),
                              bendColoring_.colorStart.data());
}

void Batch::synchronize() const { deviceSynchronize(); }

long long Batch::selfContactOverflow() const {
    if (!selfCollision_) return 0;
    std::vector<int> overflow(numRods_);
    copyToHost(overflow.data(), impl_->state.selfOverflow, numRods_ * sizeof(int));
    long long total = 0;
    for (int v : overflow) total += v;
    return total;
}

}  // namespace gpu
}  // namespace crs
