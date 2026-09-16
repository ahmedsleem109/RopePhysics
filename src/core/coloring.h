// Constraint graph coloring.
//
// Two constraints may be solved in parallel exactly when they touch no common
// state. Grouping constraints into colors of mutually non-conflicting members
// is what lets a Gauss-Seidel sweep run as a sequence of parallel passes with
// no atomics and no races, which is the whole basis of the GPU solver.
//
// The coloring is also used by the CPU solver, and that is deliberate: a
// colored sweep is a genuinely different iteration order from a sequential one
// (red-black Gauss-Seidel is not sequential Gauss-Seidel), so the two produce
// different trajectories from the same initial state even when both are
// correct. Being able to run the CPU in the GPU's order is what turns
// "GPU matches CPU" from a fuzzy claim into a numerical one.
#pragma once

#include <vector>

namespace crs {

struct Rod;

struct Coloring {
    // Constraint indices, grouped by color and contiguous within each group.
    std::vector<int> order;
    // numColors + 1 offsets into `order`.
    std::vector<int> colorStart;

    int numColors() const { return static_cast<int>(colorStart.size()) - 1; }
    int colorSize(int c) const { return colorStart[c + 1] - colorStart[c]; }
    bool empty() const { return order.empty(); }
};

// Greedy coloring of `count` constraints, where `resources[k]` lists the ids of
// the state each constraint writes. Ids live in one shared space, so a caller
// that mixes particles and orientation frames must offset one of them.
//
// Greedy is the right choice here rather than an optimal coloring: for a rod the
// conflict graph is a path, greedy finds the optimal two colors immediately, and
// for contact graphs the difference between greedy and optimal is a color or
// two against a scheduling cost that is linear in the color count.
Coloring colorConstraints(int count, const std::vector<std::vector<int>>& resources);

// The two colorings a rod needs. Stretch constraints write both end particles
// and the segment frame; bend constraints write two segment frames.
Coloring colorStretchConstraints(const Rod& rod);
Coloring colorBendConstraints(const Rod& rod);

// True when no color contains two constraints sharing a resource. Cheap enough
// to assert in tests, and the only thing standing between a coloring bug and a
// silent race on the GPU.
bool verifyColoring(const Coloring& coloring, const std::vector<std::vector<int>>& resources);

}  // namespace crs
