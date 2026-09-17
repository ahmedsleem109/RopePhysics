// Exploratory probe: why does a cable resting on a bar creep?
//
// The cable-hanging drape at 2:1 with mu = 0.5, more than twice the 0.22 the
// capstan equation needs, on the CPU. Reports how fast the short end rises
// between 2 s and 8 s, for a contact margin (see CollisionWorld::contactMarginRadii)
// and substep count given on the command line.
//
//   rodcreep <margin in cable radii> [substeps = 8] [iterations = 2] [ratio = 2] [mu = 0.5]
#include <algorithm>
#include <cstdio>
#include <cstdlib>

#include "apps/cable_hanging.h"

using namespace crs;

int main(int argc, char** argv) {
    const double marginRadii = argc > 1 ? std::atof(argv[1]) : 0.0;
    const apps::HangingCable task;
    SolverParams p = task.params();
    if (argc > 2) p.substeps = std::atoi(argv[2]);
    if (argc > 3) p.iterations = std::atoi(argv[3]);
    const Real ratio = Real(argc > 4 ? std::atof(argv[4]) : 2.0);
    const Real mu = Real(argc > 5 ? std::atof(argv[5]) : 0.5);

    Rod rod = task.build(ratio, Real(1e6));
    CollisionWorld world = task.world(mu);
    world.contactMarginRadii = Real(marginRadii);
    SolverContext ctx;

    const int stepsPerSecond = int(1.0 / double(p.dt) + 0.5);
    const double z0 = double(rod.state.x.front().z);
    double z2 = 0, maxRise = 0;
    for (int sec = 1; sec <= 8; ++sec) {
        for (int i = 0; i < stepsPerSecond; ++i) {
            step(rod, p, world, ctx);
            maxRise = std::max(maxRise, double(rod.state.x.front().z) - z0);
        }
        if (sec == 2) z2 = double(rod.state.x.front().z);
    }
    const double z8 = double(rod.state.x.front().z);
    std::printf("margin %.2f r, %d x %d, ratio %.2f, mu %.2f: max rise %.1f mm (%s), rate 2-8 s %.3f mm/s, "
                "ends at z %.3f / %.3f\n",
                marginRadii, p.substeps, p.iterations, double(ratio), double(mu), 1e3 * maxRise,
                maxRise > 0.05 ? "SLID" : "held", 1e3 * (z8 - z2) / 6.0, z8,
                double(rod.state.x.back().z));
    return 0;
}
