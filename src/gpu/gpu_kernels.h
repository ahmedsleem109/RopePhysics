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
};

struct StepConfigF {
    float h = 1e-4f;
    int substeps = 8;
    int iterations = 1;
    Vec3f gravity;
    float linDecay = 1.0f;
    float angDecay = 1.0f;
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
