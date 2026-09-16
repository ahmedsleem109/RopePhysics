// The CUDA/host boundary.
//
// Everything nvcc compiles lives behind this header, and it deliberately
// contains no standard library beyond <cmath> (pulled in by math3.h) and
// <cstddef>. Two reasons, one principled and one practical:
//
//   * The kernels and the host orchestration are genuinely different code with
//     different constraints, and keeping the translation unit nvcc sees down to
//     device code alone keeps compile times and error messages sane.
//   * nvcc pins the host compiler versions it will accept, and the failures
//     when that is stretched are almost always its front end choking on a newer
//     standard library rather than on anything to do with CUDA. Not handing it
//     <vector> and <string> removes that whole class of problem.
#pragma once

#include <cstddef>

#include "../core/geometry.h"
#include "../core/math3.h"

namespace crs {
namespace gpu {

// Topology, rest state and material: identical for every rod in the batch, so
// these arrays are indexed by constraint or element alone and cost no per-rod
// bandwidth.
struct DevRodF {
    int numRods = 0, numParticles = 0, numSegments = 0, numStretch = 0, numBend = 0;

    const float* invMass = nullptr;      // per particle
    const Vec3f* invInertia = nullptr;   // per segment
    const Vec3f* inertia = nullptr;      // per segment

    const int* sP0 = nullptr;
    const int* sP1 = nullptr;
    const int* sSeg = nullptr;
    const float* sRest = nullptr;
    const Vec3f* sCompliance = nullptr;

    const int* bA = nullptr;
    const int* bB = nullptr;
    const float* bRest = nullptr;
    const Vec3f* bCompliance = nullptr;
    const Vec3f* bRestDarboux = nullptr;

    const int* sOrder = nullptr;
    const int* sColorStart = nullptr;
    int sNumColors = 0;
    const int* bOrder = nullptr;
    const int* bColorStart = nullptr;
    int bNumColors = 0;

    // The static world every rod collides with, shared by the batch. Rod-vs-world
    // contact is per particle (a chain of spheres of the rod's radius), exactly
    // as generateContacts does on the CPU.
    const Primitivef* prims = nullptr;
    int numPrims = 0;
    float radius = 0;

    // Self-collision, as generateContacts does it on the CPU. The index gap and
    // table size come from the host (selfCollisionIndexGap, world.hashTableSize).
    int selfCollision = 0;
    int selfGap = 0;
    float selfFriction = 0;
    int hashTableSize = 0;
    int selfCapacity = 0;  // self contacts per rod; more are counted, not stored
};

// A rod-vs-rod contact: two segments, four particles, weights (1-u, u, -(1-v),
// -v), one normal. The four-particle case of the CPU Contact.
struct DevSelfContactF {
    int idx[4];
    float weight[4];
    Vec3f normal;
    float offset = 0;
    float friction = 0;
    float lambdaN = 0;
    float appliedTangential = 0;
};

// One potential contact: a (particle, primitive) pair of one rod. Slots are
// fixed, one per pair, so generation never allocates and projection never
// needs a list; `active` marks the ones currently penetrating. This is the
// single-particle case of the CPU Contact (weight 1).
struct DevContactF {
    Vec3f normal;
    float offset = 0;
    float friction = 0;
    float lambdaN = 0;
    float appliedTangential = 0;
    int active = 0;
};

// Per-rod state. Layout is decided by the caller: element-major
// (element * numRods + rod) for the multi-kernel path, rod-major
// (rod * numElements + element) for the fused one.
struct DevStateF {
    Vec3f* x = nullptr;
    Vec3f* xPrev = nullptr;
    Vec3f* v = nullptr;
    Vec3f* omega = nullptr;
    Vec3f* lamS = nullptr;
    Vec3f* lamB = nullptr;
    Quatf* q = nullptr;
    Quatf* qPrev = nullptr;
    // Applied loads, per rod: force on each particle [N] and torque on each
    // segment [N m] in WORLD axes, as on the CPU. Same layout as x and q.
    Vec3f* force = nullptr;
    Vec3f* torque = nullptr;
    Vec3f* kinematicVelocity = nullptr;  // pinned particles move at this, as on the CPU
    // numPrims contact slots per particle, indexed particle-slot * numPrims + k.
    DevContactF* contacts = nullptr;

    // Self-collision scratch, always rod-major (rod * size + item) whatever the
    // strategy, because each rod's self-collision runs sequentially in one
    // thread. See gpu_solver.cu for why.
    Vec3f* selfCentres = nullptr;          // numStretch per rod
    int* selfCellOf = nullptr;             // numStretch per rod
    int* selfSorted = nullptr;             // numStretch per rod
    int* selfCellStart = nullptr;          // hashTableSize + 1 per rod
    DevSelfContactF* selfPool = nullptr;   // selfCapacity per rod
    int* selfSegCount = nullptr;           // numStretch per rod
    int* selfSegStart = nullptr;           // numStretch per rod
    int* selfLive = nullptr;               // one per rod
    float* selfCellSize = nullptr;         // one per rod
    int* selfOverflow = nullptr;           // one per rod, contacts dropped (cumulative)
};

struct StepConfigF {
    float h = 1e-4f;
    int substeps = 8;
    int iterations = 1;
    Vec3f gravity;
    float linDecay = 1.0f;
    float angDecay = 1.0f;
    int contactInterval = 1;  // regenerate contacts every N substeps
};

// One full step. The colour offsets are needed on the host to issue one launch
// per colour, so they are passed as host arrays alongside the device copies.
void launchMultiKernelStep(const DevRodF& rod, const DevStateF& state, const StepConfigF& cfg,
                           const int* hostStretchColorStart, const int* hostBendColorStart);

// One full step in a single launch: one block per rod, state resident in shared
// memory for the whole step.
void launchFusedStep(const DevRodF& rod, const DevStateF& state, const StepConfigF& cfg,
                     std::size_t sharedBytes, int threadsPerBlock);

// Thin device-query shims so the host side never includes a CUDA header either.
int deviceCount();
// Why the device is unusable, when it is. Returns the cudaError_t of the device
// query plus the runtime and driver CUDA versions, which is the pair that
// actually explains most "no CUDA device" failures.
int deviceProbe(int* runtimeVersion, int* driverVersion, const char** message);
bool deviceInfo(char* nameOut, int nameLen, int* major, int* minor, int* multiprocessors,
                std::size_t* sharedMemPerBlock);
void deviceSynchronize();
const char* lastErrorString();

// Memory helpers, so allocation stays on this side of the boundary too.
void* deviceMalloc(std::size_t bytes);
void deviceFree(void* p);
bool copyToDevice(void* dst, const void* src, std::size_t bytes);
bool copyToHost(void* dst, const void* src, std::size_t bytes);

}  // namespace gpu
}  // namespace crs
