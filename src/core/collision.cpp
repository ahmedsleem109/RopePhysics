#include "collision.h"

#include <algorithm>
#include <cmath>

namespace crs {

// ---------------------------------------------------------------- hash

void SpatialHash::build(const std::vector<Vec3>& centres, Real cell, int table) {
    cellSize = cell;
    tableSize = table;
    const std::size_t n = centres.size();

    cellOf.resize(n);
    sortedItems.resize(n);
    cellStart.assign(static_cast<std::size_t>(tableSize) + 1, 0);

    for (std::size_t i = 0; i < n; ++i) {
        const Vec3 c = centres[i];
        cellOf[i] = hashCell(cellCoord(c.x, cellSize), cellCoord(c.y, cellSize),
                             cellCoord(c.z, cellSize), tableSize);
        ++cellStart[cellOf[i] + 1];
    }
    // Exclusive scan: the same prefix sum the GPU version will run.
    for (int i = 0; i < tableSize; ++i) cellStart[i + 1] += cellStart[i];

    std::vector<int> cursor(cellStart.begin(), cellStart.end() - 1);
    for (std::size_t i = 0; i < n; ++i) sortedItems[cursor[cellOf[i]]++] = static_cast<int>(i);
}

void SpatialHash::queryNeighbourhood(Vec3 p, std::vector<int>& out) const {
    const int cx = cellCoord(p.x, cellSize), cy = cellCoord(p.y, cellSize),
              cz = cellCoord(p.z, cellSize);
    for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                const uint32_t h = hashCell(cx + dx, cy + dy, cz + dz, tableSize);
                for (int k = cellStart[h]; k < cellStart[h + 1]; ++k) out.push_back(sortedItems[k]);
            }
}

// ---------------------------------------------------------------- generation

namespace {

// Build a contact from its already-evaluated gap, storing the offset that makes
// C(x) reproduce that gap at the current positions.
Contact makeContact(const RodState& s, const int* idx, const Real* weight, int count, Vec3 normal,
                    Real gap, Real friction) {
    Contact c;
    c.count = count;
    c.normal = normal;
    c.friction = friction;
    Real dotSum = 0;
    for (int k = 0; k < count; ++k) {
        c.idx[k] = idx[k];
        c.weight[k] = weight[k];
        dotSum += weight[k] * dot(s.x[idx[k]], normal);
    }
    c.offset = dotSum - gap;
    return c;
}

}  // namespace

int selfCollisionIndexGap(const Rod& rod, const CollisionWorld& world) {
    Real lMin = rod.stretch.restLength.empty() ? Real(1) : rod.stretch.restLength[0];
    for (Real l : rod.stretch.restLength) lMin = std::min(lMin, l);
    // (gap - 1) whole segments lie strictly between j and k; require that rest
    // length to reach one diameter.
    const int geometric = static_cast<int>(std::floor(2 * rod.material.radius / lMin)) + 2;
    return std::max(world.selfSkip, geometric);
}

void generateContacts(const Rod& rod, const CollisionWorld& world, ContactSet& out,
                      SpatialHash& hash) {
    out.clear();
    const RodState& s = rod.state;
    const Real r = rod.material.radius;

    // --- rod vs primitives ---
    //
    // Contacts are generated per particle: the rod is a chain of spheres of its
    // own radius as far as the world is concerned. That is exact for a plane,
    // and for smooth convex primitives it is accurate as long as the segment
    // length does not much exceed 2r, which is the regime every contact case
    // here runs in. It also keeps one contact per particle per primitive, which
    // is the shape the GPU kernel wants.
    for (std::size_t i = 0; i < s.numParticles(); ++i) {
        if (s.invMass[i] == Real(0)) continue;  // pinned: cannot respond
        for (const Primitive& prim : world.primitives) {
            Vec3 n;
            const Real d = signedDistance(prim, s.x[i], n);
            const Real gap = d - r;
            if (gap >= Real(0)) continue;
            const int idx[1] = {static_cast<int>(i)};
            const Real w[1] = {Real(1)};
            out.contacts.push_back(makeContact(s, idx, w, 1, n, gap, prim.friction));
        }
    }

    if (!world.selfCollision) return;

    // --- rod vs itself ---
    const int nSeg = static_cast<int>(rod.stretch.size());
    const int gap = selfCollisionIndexGap(rod, world);
    if (nSeg <= gap) return;

    std::vector<Vec3> centres(nSeg);
    Real maxSegLen = 0;
    for (int j = 0; j < nSeg; ++j) {
        const Vec3 a = s.x[rod.stretch.p0[j]], b = s.x[rod.stretch.p1[j]];
        centres[j] = (a + b) * Real(0.5);
        maxSegLen = std::max(maxSegLen, norm(b - a));
    }

    // A 3x3x3 neighbourhood reaches exactly one cell width, so the cell has to
    // be at least as large as the furthest midpoint separation that can still
    // produce a contact: half of each segment plus the two radii.
    const Real cell = maxSegLen + 2 * r;
    hash.build(centres, cell, world.hashTableSize);

    std::vector<int> candidates;
    for (int j = 0; j < nSeg; ++j) {
        candidates.clear();
        hash.queryNeighbourhood(centres[j], candidates);
        for (int k : candidates) {
            // Each pair once, and never segments too close along the rod to
            // be separated by a diameter at rest.
            if (k - j < gap) continue;

            Real u, v;
            const Vec3 a0 = s.x[rod.stretch.p0[j]], a1 = s.x[rod.stretch.p1[j]];
            const Vec3 b0 = s.x[rod.stretch.p0[k]], b1 = s.x[rod.stretch.p1[k]];
            closestPointsBetweenSegments(a0, a1, b0, b1, u, v);

            const Vec3 pa = a0 + (a1 - a0) * u;
            const Vec3 pb = b0 + (b1 - b0) * v;
            const Vec3 delta = pa - pb;
            const Real dist = norm(delta);
            const Real separation = dist - 2 * r;
            if (separation >= Real(0)) continue;

            const Vec3 n = dist > Real(1e-12) ? delta / dist : geometry_detail::fallbackNormal<Real>();
            const int idx[4] = {rod.stretch.p0[j], rod.stretch.p1[j], rod.stretch.p0[k],
                                rod.stretch.p1[k]};
            const Real w[4] = {Real(1) - u, u, -(Real(1) - v), -v};
            out.contacts.push_back(makeContact(s, idx, w, 4, n, separation, world.selfFriction));
        }
    }
}

}  // namespace crs
