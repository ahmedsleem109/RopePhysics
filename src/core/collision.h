// Contact against analytic primitives, rod self-collision, and the uniform
// spatial hash that feeds both.
//
// Design notes that matter for the CUDA port:
//   * Primitives are flat PODs with a type tag, not a class hierarchy.
//   * Contacts are a flat array of PODs with a uniform shape: up to four
//     particles, a weight each, one normal. Rod-vs-world and rod-vs-rod contacts
//     are the same struct and the same solver code.
//   * The broadphase is key generation -> sort -> cell-start offsets, which is
//     the same three steps the GPU version will run as key kernel -> radix sort
//     -> scan. The CPU sort here is a counting sort for exactly that reason.
#pragma once

#include <cstdint>
#include <vector>

#include "geometry.h"
#include "rod.h"

namespace crs {

// Primitives, signed distances and segment closest points live in geometry.h,
// shared with the CUDA kernels.

// ---------------------------------------------------------------- contacts

// One inequality constraint C >= 0, linearized about a fixed normal:
//
//     C(x) = sum_k weight[k] * dot(x[idx[k]], normal) - offset
//
// with the offset chosen at generation time so C equals the true signed gap.
// A rod-vs-world contact uses one particle (weight 1); a rod-vs-rod contact uses
// four, with the two on the second segment carrying negative weights, so the
// same expression is the relative gap.
struct Contact {
    int idx[4] = {-1, -1, -1, -1};
    Real weight[4] = {0, 0, 0, 0};
    Vec3 normal;
    Real offset = 0;
    Real friction = 0;
    Real lambdaN = 0;
    // Tangential correction already spent this substep, as a length. The
    // Coulomb cone bounds the TOTAL, not each sweep's share, so this has to be
    // carried across sweeps -- see projectContacts.
    Real appliedTangential = 0;

    int count = 0;
};

struct ContactSet {
    std::vector<Contact> contacts;

    void clear() { contacts.clear(); }
    std::size_t size() const { return contacts.size(); }
    void resetMultipliers() {
        for (Contact& c : contacts) {
            c.lambdaN = 0;
            c.appliedTangential = 0;
        }
    }
};

// ---------------------------------------------------------------- broadphase

// Uniform spatial hash over item centres. Cells are hashed into a fixed-size
// table, so a query returns candidates that must still be distance-checked --
// which the narrowphase does anyway.
struct SpatialHash {
    Real cellSize = 1;
    int tableSize = 0;
    std::vector<uint32_t> cellOf;    // per item
    std::vector<int> sortedItems;    // items ordered by cell
    std::vector<int> cellStart;      // tableSize + 1 offsets into sortedItems

    void build(const std::vector<Vec3>& centres, Real cellSize, int tableSize);

    // Append every item whose cell touches the 3x3x3 neighbourhood of `p`.
    void queryNeighbourhood(Vec3 p, std::vector<int>& out) const;
};

uint32_t hashCell(int ix, int iy, int iz, int tableSize);

// ---------------------------------------------------------------- generation

struct CollisionWorld {
    std::vector<Primitive> primitives;

    bool selfCollision = false;
    Real selfFriction = Real(0);
    // Minimum index gap between segments that may generate a self contact.
    // The gap actually used is the larger of this and what the rod's geometry
    // demands -- see selfCollisionIndexGap.
    int selfSkip = 2;

    int hashTableSize = 4096;
};

// Smallest index gap k - j at which segments j and k may generate a self
// contact.
//
// Skipping only direct neighbours is not enough. Segments j and j+2 are
// separated by exactly one segment, so at rest they are one segment length
// apart -- and when segments are shorter than the rope's diameter that is
// already closer than 2r. The solver would then push apart a rope that is not
// touching itself at all. The gap therefore has to span at least a diameter of
// rest length.
int selfCollisionIndexGap(const Rod& rod, const CollisionWorld& world);

// Rebuild `out` from scratch for the rod's current configuration.
void generateContacts(const Rod& rod, const CollisionWorld& world, ContactSet& out,
                      SpatialHash& hash);

}  // namespace crs
