// CUDA solver: the same XPBD Cosserat rod, batched across independent
// environments.
//
// This header is plain C++ and is safe to include from non-CUDA translation
// units. When the project is configured without a CUDA compiler the
// implementation compiles to stubs and `cudaAvailable()` returns false.
//
// Two execution strategies are provided, and they are genuinely different
// designs rather than a fast path and a slow path:
//
//   kMultiKernel  One kernel launch per constraint colour per sweep. General:
//                 it does not care how large a rod is or how many colours the
//                 graph needs. Pays a launch per colour per sweep.
//
//   kFused        One block per rod, the entire rod's state resident in shared
//                 memory, and the whole substep loop -- predict, every colour of
//                 every sweep, velocity update -- inside a single kernel with
//                 __syncthreads() between colours. No launches inside the step
//                 and no round trip to global memory between sweeps. Limited to
//                 rods whose state fits in a block's shared memory.
//
// The two want opposite memory layouts, which is why the layout is a parameter
// rather than a constant:
//
//   Multi-kernel threads are (constraint, rod) pairs with the rod varying
//   fastest, so consecutive threads touch consecutive rods -- element-major
//   (element * numRods + rod) makes that a coalesced load.
//
//   A fused block owns one rod and walks its elements, so consecutive threads
//   touch consecutive elements of the same rod -- rod-major
//   (rod * numElements + element) is the coalesced choice there.
#pragma once

#include <string>
#include <vector>

#include "../core/collision.h"
#include "../core/coloring.h"
#include "../core/rod.h"

namespace crs {
namespace gpu {

bool cudaAvailable();
std::string deviceName();
// Human-readable reason the device is unusable, including the runtime and
// driver CUDA versions. Empty when a device is available.
std::string diagnostic();

enum class Strategy { kMultiKernel, kFused };

struct BatchParams {
    float dt = 1e-3f;
    int substeps = 8;
    int iterations = 1;
    float gravityX = 0, gravityY = 0, gravityZ = -9.81f;
    float linearDamping = 0;
    float angularDamping = 0;
    int contactInterval = 1;  // regenerate world contacts every N substeps
};

// One batch of identical rods. Topology, rest state and material are shared
// across the batch; positions, orientations, velocities and multipliers are
// per rod. That is the shape a reinforcement-learning batch actually has, and
// it keeps the per-rod bandwidth down to the state that genuinely differs.
class Batch {
   public:
    Batch() = default;
    ~Batch();
    Batch(const Batch&) = delete;
    Batch& operator=(const Batch&) = delete;

    // Replicate `prototype` across `numRods` environments, optionally colliding
    // with the static primitives of `world`. Returns false if the batch cannot
    // run: for kFused, if a rod's state and contacts do not fit in shared
    // memory; for any strategy, if `world` asks for self-collision, which the
    // GPU path does not support yet.
    bool create(const Rod& prototype, int numRods, Strategy strategy,
                const CollisionWorld* world = nullptr);
    void destroy();

    void step(const BatchParams& params);
    void synchronize() const;

    // Copy one environment's state back into a host rod, which must have the
    // same topology as the prototype.
    void download(int rodIndex, Rod& out) const;
    // Re-upload every environment from a host rod (all environments identical),
    // including its applied loads.
    void upload(const Rod& in);
    // Set one environment's applied loads from a host rod with the same
    // topology: extForce per particle, extTorque per segment, and the
    // orientation of every fixed frame (zero inertia). Updating a fixed frame's
    // orientation between steps is how an end is driven, e.g. twisted.
    void setLoads(int rodIndex, const Rod& source);

    int numRods() const { return numRods_; }
    int numSegments() const { return numSegments_; }
    Strategy strategy() const { return strategy_; }
    int stretchColors() const { return stretchColoring_.numColors(); }
    int bendColors() const { return bendColoring_.numColors(); }
    std::size_t sharedBytesPerBlock() const { return sharedBytes_; }
    // Kernel launches issued per step, the number the fused path is trying to
    // drive to one.
    int launchesPerStep(const BatchParams& params) const;

   private:
    struct Impl;
    Impl* impl_ = nullptr;

    int numRods_ = 0;
    int numSegments_ = 0;
    int numParticles_ = 0;
    int numPrims_ = 0;
    Strategy strategy_ = Strategy::kMultiKernel;
    Coloring stretchColoring_;
    Coloring bendColoring_;
    std::size_t sharedBytes_ = 0;
};

// Largest rod the fused strategy can run, given the device's shared memory per
// block. Reported rather than assumed, because it is the limit that decides
// whether a scene is batchable at all.
int maxFusedSegments();

}  // namespace gpu
}  // namespace crs
