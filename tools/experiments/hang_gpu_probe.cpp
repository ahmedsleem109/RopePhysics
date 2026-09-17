// Exploratory probe: do the GPU and the CPU agree on one cable-hanging
// placement near the hold/slip boundary?
//
// Runs the placement on the GPU (float) and on the CPU in the GPU's constraint
// order, both in double and, if built with CRS_REAL_FLOAT, float. Prints the
// short end's largest rise over 8 s on each.
//
//   rodhang <leg ratio> <mu> [margin radii = 0.25] [substeps] [iterations] [cpu]
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "apps/cable_hanging.h"
#include "core/coloring.h"
#include "gpu/gpu_solver.h"

using namespace crs;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: rodhang <leg ratio> <mu> [margin radii]\n");
        return 2;
    }
    const Real ratio = Real(std::atof(argv[1]));
    const Real mu = Real(std::atof(argv[2]));
    const apps::HangingCable task;
    CollisionWorld world = task.world(mu);
    if (argc > 3) world.contactMarginRadii = Real(std::atof(argv[3]));
    SolverParams p = task.params();
    if (argc > 4) p.substeps = std::atoi(argv[4]);
    if (argc > 5) p.iterations = std::atoi(argv[5]);
    const bool cpuOnly = argc > 6;

    // CPU, GPU constraint order.
    {
        Rod rod = task.build(ratio, Real(1e6));
        const Coloring stretch = colorStretchConstraints(rod);
        const Coloring bend = colorBendConstraints(rod);
        SolverParams q = p;
        q.stretchColoring = &stretch;
        q.bendColoring = &bend;
        SolverContext ctx;
        const double z0 = double(rod.state.x.front().z);
        double rise = 0;
        for (int s = 0; s < task.steps(); ++s) {
            step(rod, q, world, ctx);
            rise = std::max(rise, double(rod.state.x.front().z) - z0);
        }
        std::printf("ratio %.2f mu %.2f, CPU (%s, %d x %d): max rise %.1f mm\n", double(ratio),
                    double(mu), sizeof(Real) == 8 ? "double" : "float", p.substeps, p.iterations,
                    1e3 * rise);
    }

    if (cpuOnly) return 0;
    if (!gpu::cudaAvailable()) {
        std::printf("GPU: %s\n", gpu::diagnostic().c_str());
        return 0;
    }
    const Rod rod = task.build(ratio, Real(1e6));
    gpu::Batch batch;
    if (!batch.create(rod, 1, gpu::Strategy::kFused, &world)) return 1;
    gpu::BatchParams bp;
    bp.dt = float(p.dt);
    bp.substeps = p.substeps;
    bp.iterations = p.iterations;
    bp.gravityZ = float(p.gravity.z);
    std::vector<Vec3f> pos;
    const double z0 = double(rod.state.x.front().z);
    double rise = 0;
    for (int s = 1; s <= task.steps(); ++s) {
        batch.step(bp);
        if (s % 50 == 0) {
            batch.downloadPositions(pos);
            rise = std::max(rise, double(pos.front().z) - z0);
        }
    }
    std::printf("GPU (float): max rise %.1f mm\n", 1e3 * rise);
    return 0;
}
