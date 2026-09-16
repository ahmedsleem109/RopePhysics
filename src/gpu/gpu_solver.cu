// CUDA kernels for the XPBD Cosserat rod.
//
// The algebra below is line for line src/core/solver.cpp, reading through a
// View instead of std::vector. One derivation, two storage strategies -- rather
// than two derivations -- is what makes "the GPU reproduces the CPU trajectory"
// a claim about the port rather than about the physics.
#include <cuda_runtime.h>

#include "gpu_kernels.h"

namespace crs {
namespace gpu {

namespace {

// A view onto one rod's state. Every arrangement this solver uses --
// element-major global, rod-major global, and a block's shared memory -- is the
// same affine map base + element * stride, so one view type serves all three and
// the projection routines never learn which one they are running on.
struct View {
    Vec3f* x;
    Vec3f* xPrev;
    Vec3f* v;
    Quatf* q;
    Quatf* qPrev;
    Vec3f* omega;
    Vec3f* lamS;
    Vec3f* lamB;
    Vec3f* force;   // indexed like x
    Vec3f* torque;  // indexed like q
    DevContactF* contacts;  // indexed P(i) * numPrims + k

    int pBase, pStride;
    int sBase, sStride;
    int lsBase, lsStride;
    int lbBase, lbStride;

    __device__ int P(int i) const { return pBase + i * pStride; }
    __device__ int S(int j) const { return sBase + j * sStride; }
    __device__ int LS(int k) const { return lsBase + k * lsStride; }
    __device__ int LB(int k) const { return lbBase + k * lbStride; }
};

// ---------------------------------------------------------------- projections

__device__ void projectStretchOne(const DevRodF& d, const View& view, int k, float invH2) {
    const int i0 = d.sP0[k], i1 = d.sP1[k], j = d.sSeg[k];
    const float w0 = d.invMass[i0], w1 = d.invMass[i1];
    const Vec3f iI = d.invInertia[j];
    if (w0 == 0.0f && w1 == 0.0f && norm2(iI) == 0.0f) return;

    const float l = d.sRest[k];
    const Quatf q = view.q[view.S(j)];
    // Material-frame C, as in solver.cpp (see the note there).
    const Vec3f u = rotateInv(q, view.x[view.P(i1)] - view.x[view.P(i0)]) / l;
    const Vec3f C = u - Vec3f(0, 0, 1);

    const Vec3f alpha = d.sCompliance[k] * invH2;
    Mat3f A = Mat3f::identity((w0 + w1) / (l * l)) + sandwichDiag(Mat3f::skew(u), iI);
    A.m[0][0] += alpha.x;
    A.m[1][1] += alpha.y;
    A.m[2][2] += alpha.z;

    const int li = view.LS(k);
    Vec3f dLambda;
    if (!solveSPD3(A, -(C + cwise(alpha, view.lamS[li])), dLambda)) return;
    view.lamS[li] += dLambda;

    const Vec3f world = rotate(q, dLambda);
    view.x[view.P(i0)] -= world * (w0 / l);
    view.x[view.P(i1)] += world * (w1 / l);
    if (norm2(iI) > 0.0f) view.q[view.S(j)] = applyBodyDelta(q, cwise(iI, cross(dLambda, u)));
}

__device__ void projectBendOne(const DevRodF& d, const View& view, int k, float invH2) {
    const int a = d.bA[k], b = d.bB[k];
    const Vec3f iIa = d.invInertia[a], iIb = d.invInertia[b];
    if (norm2(iIa) == 0.0f && norm2(iIb) == 0.0f) return;

    const float lbar = d.bRest[k];
    const Quatf qa = view.q[view.S(a)];
    const Quatf qb = view.q[view.S(b)];
    const Quatf p = sameHemisphere(conj(qa) * qb, Quatf());
    const Vec3f C = p.im() * (2.0f / lbar) - d.bRestDarboux[k];

    const Mat3f skewPv = Mat3f::skew(p.im());
    const Mat3f Ja = (Mat3f::identity(-p.w) + skewPv) * (1.0f / lbar);
    const Mat3f Jb = (Mat3f::identity(p.w) + skewPv) * (1.0f / lbar);

    const Vec3f alpha = d.bCompliance[k] * invH2;
    Mat3f A = sandwichDiag(Ja, iIa) + sandwichDiag(Jb, iIb);
    A.m[0][0] += alpha.x;
    A.m[1][1] += alpha.y;
    A.m[2][2] += alpha.z;

    const int li = view.LB(k);
    Vec3f dLambda;
    if (!solveSPD3(A, -(C + cwise(alpha, view.lamB[li])), dLambda)) return;
    view.lamB[li] += dLambda;

    if (norm2(iIa) > 0.0f) {
        const Vec3f g = (dLambda * -p.w - cross(p.im(), dLambda)) * (1.0f / lbar);
        view.q[view.S(a)] = applyBodyDelta(qa, cwise(iIa, g));
    }
    if (norm2(iIb) > 0.0f) {
        const Vec3f g = (dLambda * p.w - cross(p.im(), dLambda)) * (1.0f / lbar);
        view.q[view.S(b)] = applyBodyDelta(qb, cwise(iIb, g));
    }
}

// ---- contacts, rod vs world ------------------------------------------------
//
// The single-particle case of collision.cpp / projectContacts, per particle,
// with the primitives visited in order. Contacts on different particles share
// no state, so running particles in parallel is the same Gauss-Seidel sweep the
// CPU does.

// Rebuild one particle's contacts from its position at the start of the
// substep, or, between regenerations, only reset their multipliers.
__device__ void generateParticleContacts(const DevRodF& d, const View& view, int i,
                                         bool regenerate) {
    const int base = view.P(i) * d.numPrims;
    for (int k = 0; k < d.numPrims; ++k) {
        DevContactF& c = view.contacts[base + k];
        c.lambdaN = 0.0f;
        c.appliedTangential = 0.0f;
        if (!regenerate) continue;
        c.active = 0;
        if (d.invMass[i] == 0.0f) continue;  // pinned: cannot respond
        const Vec3f x = view.x[view.P(i)];
        Vec3f n;
        const float gap = signedDistance(d.prims[k], x, n) - d.radius;
        if (gap >= 0.0f) continue;
        c.active = 1;
        c.normal = n;
        c.offset = dot(x, n) - gap;
        c.friction = d.prims[k].friction;
    }
}

__device__ void projectParticleContacts(const DevRodF& d, const View& view, int i) {
    const int pi = view.P(i);
    const float w = d.invMass[i];
    if (w <= 0.0f) return;
    const int base = pi * d.numPrims;
    for (int k = 0; k < d.numPrims; ++k) {
        DevContactF& c = view.contacts[base + k];
        if (!c.active) continue;

        // --- non-penetration ---
        const float C = dot(view.x[pi], c.normal) - c.offset;
        float dLambda = -C / w;
        const float clamped = c.lambdaN + dLambda > 0.0f ? c.lambdaN + dLambda : 0.0f;
        dLambda = clamped - c.lambdaN;
        c.lambdaN = clamped;
        if (dLambda != 0.0f) view.x[pi] += c.normal * (w * dLambda);

        if (c.friction <= 0.0f || c.lambdaN <= 0.0f) continue;

        // --- Coulomb friction, position level, total capped per substep ---
        const Vec3f dp = view.x[pi] - view.xPrev[pi];
        const Vec3f tangential = dp - c.normal * dot(dp, c.normal);
        const float slide = norm(tangential);
        if (slide < 1e-15f) continue;
        const float budget = c.friction * c.lambdaN * w - c.appliedTangential;
        if (budget <= 0.0f) continue;
        const float capped = slide < budget ? slide : budget;
        c.appliedTangential += capped;
        view.x[pi] += tangential * (-capped / slide);
    }
}

__device__ void predictParticle(const DevRodF& d, const View& view, int i, float h, Vec3f gravity) {
    const int pi = view.P(i);
    view.xPrev[pi] = view.x[pi];
    if (d.invMass[i] == 0.0f) return;
    view.v[pi] += (gravity + view.force[pi] * d.invMass[i]) * h;
    view.x[pi] += view.v[pi] * h;
}

__device__ void predictFrame(const DevRodF& d, const View& view, int j, float h) {
    const int si = view.S(j);
    view.qPrev[si] = view.q[si];
    const Vec3f iI = d.invInertia[j];
    if (norm2(iI) == 0.0f) return;
    // Free-body precession plus applied torque, in the body frame (the torque
    // is stored in world axes), exactly as solver.cpp.
    const Vec3f w = view.omega[si];
    const Vec3f tau = rotateInv(view.q[si], view.torque[si]);
    view.omega[si] = w + cwise(iI, tau - cross(w, cwise(d.inertia[j], w))) * h;
    view.q[si] = normalize(view.q[si] + (view.q[si] * Quatf(0.0f, view.omega[si])) * (0.5f * h));
}

__device__ void finishParticle(const DevRodF& d, const View& view, int i, float h, float linDecay) {
    const int pi = view.P(i);
    if (d.invMass[i] == 0.0f) {
        view.v[pi] = Vec3f();
        return;
    }
    view.v[pi] = (view.x[pi] - view.xPrev[pi]) / h * linDecay;
}

__device__ void finishFrame(const DevRodF& d, const View& view, int j, float h, float angDecay) {
    const int si = view.S(j);
    if (norm2(d.invInertia[j]) == 0.0f) {
        view.omega[si] = Vec3f();
        return;
    }
    const Quatf dq = sameHemisphere(conj(view.qPrev[si]) * view.q[si], Quatf());
    view.omega[si] = dq.im() * (2.0f / h) * angDecay;
}

// ---------------------------------------------------------- multi-kernel path
//
// Element-major storage: element e of rod r lives at e * numRods + r. Threads
// are (constraint, rod) pairs with the rod varying fastest, so consecutive
// threads touch consecutive addresses.

__device__ View globalView(const DevRodF& d, const DevStateF& s, int r) {
    View view;
    view.x = s.x;
    view.xPrev = s.xPrev;
    view.v = s.v;
    view.q = s.q;
    view.qPrev = s.qPrev;
    view.omega = s.omega;
    view.lamS = s.lamS;
    view.lamB = s.lamB;
    view.force = s.force;
    view.torque = s.torque;
    view.contacts = s.contacts;
    view.pBase = view.sBase = view.lsBase = view.lbBase = r;
    view.pStride = view.sStride = view.lsStride = view.lbStride = d.numRods;
    return view;
}

__global__ void kPredict(DevRodF d, DevStateF s, float h, Vec3f gravity) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d.numParticles * d.numRods) return;
    const int i = idx / d.numRods;
    predictParticle(d, globalView(d, s, idx - i * d.numRods), i, h, gravity);
}

__global__ void kPredictFrames(DevRodF d, DevStateF s, float h) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d.numSegments * d.numRods) return;
    const int j = idx / d.numRods;
    predictFrame(d, globalView(d, s, idx - j * d.numRods), j, h);
}

__global__ void kGenerateContacts(DevRodF d, DevStateF s, bool regenerate) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d.numParticles * d.numRods) return;
    const int i = idx / d.numRods;
    generateParticleContacts(d, globalView(d, s, idx - i * d.numRods), i, regenerate);
}

__global__ void kProjectContacts(DevRodF d, DevStateF s) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d.numParticles * d.numRods) return;
    const int i = idx / d.numRods;
    projectParticleContacts(d, globalView(d, s, idx - i * d.numRods), i);
}

__global__ void kClearMultipliers(DevRodF d, DevStateF s) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int nS = d.numStretch * d.numRods;
    const int total = nS + d.numBend * d.numRods;
    if (idx >= total) return;
    if (idx < nS)
        s.lamS[idx] = Vec3f();
    else
        s.lamB[idx - nS] = Vec3f();
}

__global__ void kProjectStretchColor(DevRodF d, DevStateF s, float invH2, int colorBegin,
                                     int colorCount) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= colorCount * d.numRods) return;
    const int c = idx / d.numRods;
    projectStretchOne(d, globalView(d, s, idx - c * d.numRods), d.sOrder[colorBegin + c], invH2);
}

__global__ void kProjectBendColor(DevRodF d, DevStateF s, float invH2, int colorBegin,
                                  int colorCount) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= colorCount * d.numRods) return;
    const int c = idx / d.numRods;
    projectBendOne(d, globalView(d, s, idx - c * d.numRods), d.bOrder[colorBegin + c], invH2);
}

__global__ void kFinish(DevRodF d, DevStateF s, float h, float linDecay, float angDecay) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int totalP = d.numParticles * d.numRods;
    const int totalS = d.numSegments * d.numRods;
    if (idx < totalP) {
        const int i = idx / d.numRods;
        finishParticle(d, globalView(d, s, idx - i * d.numRods), i, h, linDecay);
    } else if (idx < totalP + totalS) {
        const int t = idx - totalP;
        const int j = t / d.numRods;
        finishFrame(d, globalView(d, s, t - j * d.numRods), j, h, angDecay);
    }
}

// ---------------------------------------------------------------- fused path
//
// One block per rod, rod-major storage so the block's load and store are
// coalesced, and the entire substep loop inside the kernel. Between colours the
// block synchronizes; nothing leaves shared memory until the step is over.
//
// The shared allocation puts quaternions first so their 16-byte alignment is
// satisfied without padding; the 12-byte vectors that follow need only 4.
__global__ void kFusedStep(DevRodF d, DevStateF g, float h, int substeps, int iterations,
                           Vec3f gravity, float linDecay, float angDecay, int contactInterval) {
    extern __shared__ char smem[];
    const int r = blockIdx.x;
    if (r >= d.numRods) return;

    const int nP = d.numParticles, nS = d.numSegments;
    const int nStretch = d.numStretch, nBend = d.numBend;

    Quatf* sq = reinterpret_cast<Quatf*>(smem);
    Quatf* sqPrev = sq + nS;
    Vec3f* sx = reinterpret_cast<Vec3f*>(sqPrev + nS);
    Vec3f* sxPrev = sx + nP;
    Vec3f* sv = sxPrev + nP;
    Vec3f* somega = sv + nP;
    Vec3f* slamS = somega + nS;
    Vec3f* slamB = slamS + nStretch;
    Vec3f* sforce = slamB + nBend;
    Vec3f* storque = sforce + nP;
    // Contacts live only inside a step (regenerated at its first substep), so
    // they are never loaded from or stored back to global memory.
    DevContactF* scontacts = reinterpret_cast<DevContactF*>(storque + nS);

    View view;
    view.x = sx;
    view.xPrev = sxPrev;
    view.v = sv;
    view.q = sq;
    view.qPrev = sqPrev;
    view.omega = somega;
    view.lamS = slamS;
    view.lamB = slamB;
    view.force = sforce;
    view.torque = storque;
    view.contacts = scontacts;
    view.pBase = view.sBase = view.lsBase = view.lbBase = 0;
    view.pStride = view.sStride = view.lsStride = view.lbStride = 1;

    const int tid = threadIdx.x, nthreads = blockDim.x;
    const int pOff = r * nP, sOff = r * nS;

    for (int i = tid; i < nP; i += nthreads) {
        sx[i] = g.x[pOff + i];
        sv[i] = g.v[pOff + i];
        sforce[i] = g.force[pOff + i];
    }
    for (int j = tid; j < nS; j += nthreads) {
        sq[j] = g.q[sOff + j];
        somega[j] = g.omega[sOff + j];
        storque[j] = g.torque[sOff + j];
    }
    __syncthreads();

    const float invH2 = 1.0f / (h * h);
    for (int sub = 0; sub < substeps; ++sub) {
        for (int k = tid; k < nStretch; k += nthreads) slamS[k] = Vec3f();
        for (int k = tid; k < nBend; k += nthreads) slamB[k] = Vec3f();
        if (d.numPrims > 0) {
            const bool regenerate = sub % contactInterval == 0;
            for (int i = tid; i < nP; i += nthreads)
                generateParticleContacts(d, view, i, regenerate);
        }
        for (int i = tid; i < nP; i += nthreads) predictParticle(d, view, i, h, gravity);
        for (int j = tid; j < nS; j += nthreads) predictFrame(d, view, j, h);
        __syncthreads();

        for (int it = 0; it < iterations; ++it) {
            for (int c = 0; c < d.sNumColors; ++c) {
                for (int ci = d.sColorStart[c] + tid; ci < d.sColorStart[c + 1]; ci += nthreads)
                    projectStretchOne(d, view, d.sOrder[ci], invH2);
                __syncthreads();
            }
            for (int c = 0; c < d.bNumColors; ++c) {
                for (int ci = d.bColorStart[c] + tid; ci < d.bColorStart[c + 1]; ci += nthreads)
                    projectBendOne(d, view, d.bOrder[ci], invH2);
                __syncthreads();
            }
            if (d.numPrims > 0) {
                for (int i = tid; i < nP; i += nthreads) projectParticleContacts(d, view, i);
                __syncthreads();
            }
        }

        for (int i = tid; i < nP; i += nthreads) finishParticle(d, view, i, h, linDecay);
        for (int j = tid; j < nS; j += nthreads) finishFrame(d, view, j, h, angDecay);
        __syncthreads();
    }

    for (int i = tid; i < nP; i += nthreads) {
        g.x[pOff + i] = sx[i];
        g.v[pOff + i] = sv[i];
    }
    for (int j = tid; j < nS; j += nthreads) {
        g.q[sOff + j] = sq[j];
        g.omega[sOff + j] = somega[j];
    }
}

int blocksFor(int n, int threads) { return (n + threads - 1) / threads; }

constexpr int kThreads = 256;

}  // namespace

// ---------------------------------------------------------------- launchers

void launchMultiKernelStep(const DevRodF& d, const DevStateF& s, const StepConfigF& cfg,
                           const int* hostStretchColorStart, const int* hostBendColorStart) {
    const int totalP = d.numParticles * d.numRods;
    const int totalS = d.numSegments * d.numRods;
    const int totalLam = (d.numStretch + d.numBend) * d.numRods;
    const float invH2 = 1.0f / (cfg.h * cfg.h);

    for (int sub = 0; sub < cfg.substeps; ++sub) {
        if (d.numPrims > 0)
            kGenerateContacts<<<blocksFor(totalP, kThreads), kThreads>>>(
                d, s, sub % cfg.contactInterval == 0);
        kClearMultipliers<<<blocksFor(totalLam, kThreads), kThreads>>>(d, s);
        kPredict<<<blocksFor(totalP, kThreads), kThreads>>>(d, s, cfg.h, cfg.gravity);
        kPredictFrames<<<blocksFor(totalS, kThreads), kThreads>>>(d, s, cfg.h);

        for (int it = 0; it < cfg.iterations; ++it) {
            for (int c = 0; c < d.sNumColors; ++c) {
                const int begin = hostStretchColorStart[c];
                const int count = hostStretchColorStart[c + 1] - begin;
                kProjectStretchColor<<<blocksFor(count * d.numRods, kThreads), kThreads>>>(
                    d, s, invH2, begin, count);
            }
            for (int c = 0; c < d.bNumColors; ++c) {
                const int begin = hostBendColorStart[c];
                const int count = hostBendColorStart[c + 1] - begin;
                kProjectBendColor<<<blocksFor(count * d.numRods, kThreads), kThreads>>>(
                    d, s, invH2, begin, count);
            }
            if (d.numPrims > 0) kProjectContacts<<<blocksFor(totalP, kThreads), kThreads>>>(d, s);
        }

        kFinish<<<blocksFor(totalP + totalS, kThreads), kThreads>>>(d, s, cfg.h, cfg.linDecay,
                                                                   cfg.angDecay);
    }
}

void launchFusedStep(const DevRodF& d, const DevStateF& s, const StepConfigF& cfg,
                     std::size_t sharedBytes, int threadsPerBlock) {
    kFusedStep<<<d.numRods, threadsPerBlock, sharedBytes>>>(d, s, cfg.h, cfg.substeps,
                                                            cfg.iterations, cfg.gravity,
                                                            cfg.linDecay, cfg.angDecay,
                                                            cfg.contactInterval);
}

// ---------------------------------------------------------------- device shims

int deviceCount() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return 0;
    return n;
}

int deviceProbe(int* runtimeVersion, int* driverVersion, const char** message) {
    int n = 0;
    const cudaError_t err = cudaGetDeviceCount(&n);
    *runtimeVersion = 0;
    *driverVersion = 0;
    cudaRuntimeGetVersion(runtimeVersion);
    cudaDriverGetVersion(driverVersion);
    *message = cudaGetErrorString(err);
    return static_cast<int>(err);
}

bool deviceInfo(char* nameOut, int nameLen, int* major, int* minor, int* multiprocessors,
                std::size_t* sharedMemPerBlock) {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return false;
    for (int i = 0; i < nameLen - 1 && prop.name[i]; ++i) nameOut[i] = prop.name[i];
    nameOut[nameLen - 1] = 0;
    int n = 0;
    while (n < nameLen - 1 && prop.name[n]) {
        nameOut[n] = prop.name[n];
        ++n;
    }
    nameOut[n] = 0;
    *major = prop.major;
    *minor = prop.minor;
    *multiprocessors = prop.multiProcessorCount;
    *sharedMemPerBlock = prop.sharedMemPerBlock;
    return true;
}

void deviceSynchronize() { cudaDeviceSynchronize(); }

const char* lastErrorString() { return cudaGetErrorString(cudaGetLastError()); }

void* deviceMalloc(std::size_t bytes) {
    void* p = nullptr;
    if (bytes == 0) return nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) return nullptr;
    return p;
}

void deviceFree(void* p) {
    if (p) cudaFree(p);
}

bool copyToDevice(void* dst, const void* src, std::size_t bytes) {
    return cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice) == cudaSuccess;
}

bool copyToHost(void* dst, const void* src, std::size_t bytes) {
    return cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
}

}  // namespace gpu
}  // namespace crs
