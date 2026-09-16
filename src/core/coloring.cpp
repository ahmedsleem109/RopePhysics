#include "coloring.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

#include "rod.h"

namespace crs {

Coloring colorConstraints(int count, const std::vector<std::vector<int>>& resources) {
    Coloring coloring;
    if (count == 0) {
        coloring.colorStart.assign(1, 0);
        return coloring;
    }

    // Colour of each resource's most recent assignment, per colour: rather than
    // building the conflict graph explicitly (which is quadratic in the worst
    // case), track for each resource the set of colours already used on it.
    std::unordered_map<int, std::vector<char>> usedByResource;
    std::vector<int> colorOf(count, -1);
    int numColors = 0;

    for (int k = 0; k < count; ++k) {
        // Find the lowest colour that no resource of this constraint has taken.
        int c = 0;
        for (;; ++c) {
            bool clash = false;
            for (int rid : resources[k]) {
                auto it = usedByResource.find(rid);
                if (it != usedByResource.end() && c < static_cast<int>(it->second.size()) &&
                    it->second[c]) {
                    clash = true;
                    break;
                }
            }
            if (!clash) break;
        }
        colorOf[k] = c;
        numColors = std::max(numColors, c + 1);
        for (int rid : resources[k]) {
            std::vector<char>& used = usedByResource[rid];
            if (static_cast<int>(used.size()) <= c) used.resize(c + 1, 0);
            used[c] = 1;
        }
    }

    coloring.colorStart.assign(static_cast<std::size_t>(numColors) + 1, 0);
    for (int k = 0; k < count; ++k) ++coloring.colorStart[colorOf[k] + 1];
    for (int c = 0; c < numColors; ++c) coloring.colorStart[c + 1] += coloring.colorStart[c];

    coloring.order.resize(count);
    std::vector<int> cursor(coloring.colorStart.begin(), coloring.colorStart.end() - 1);
    for (int k = 0; k < count; ++k) coloring.order[cursor[colorOf[k]]++] = k;
    return coloring;
}

namespace {

// Particles and orientation frames are separate arrays, so give frames ids
// above every particle id to keep them from colliding in the shared id space.
int frameId(const Rod& rod, int segment) {
    return static_cast<int>(rod.state.numParticles()) + segment;
}

}  // namespace

Coloring colorStretchConstraints(const Rod& rod) {
    const int n = static_cast<int>(rod.stretch.size());
    std::vector<std::vector<int>> resources(n);
    for (int k = 0; k < n; ++k)
        resources[k] = {rod.stretch.p0[k], rod.stretch.p1[k], frameId(rod, rod.stretch.seg[k])};
    return colorConstraints(n, resources);
}

Coloring colorBendConstraints(const Rod& rod) {
    const int n = static_cast<int>(rod.bend.size());
    std::vector<std::vector<int>> resources(n);
    for (int k = 0; k < n; ++k)
        resources[k] = {frameId(rod, rod.bend.segA[k]), frameId(rod, rod.bend.segB[k])};
    return colorConstraints(n, resources);
}

bool verifyColoring(const Coloring& coloring, const std::vector<std::vector<int>>& resources) {
    for (int c = 0; c < coloring.numColors(); ++c) {
        std::unordered_set<int> seen;
        for (int i = coloring.colorStart[c]; i < coloring.colorStart[c + 1]; ++i)
            for (int rid : resources[coloring.order[i]])
                if (!seen.insert(rid).second) return false;
    }
    return static_cast<int>(coloring.order.size()) == static_cast<int>(resources.size());
}

}  // namespace crs
