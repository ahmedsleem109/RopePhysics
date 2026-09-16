#include "collision.h"

#include <algorithm>
#include <cmath>

namespace crs {

// ---------------------------------------------------------------- primitives

Primitive Primitive::makePlane(Vec3 point, Vec3 unitNormal, Real friction) {
    Primitive p;
    p.type = kPlane;
    p.a = point;
    p.normal = normalize(unitNormal);
    p.friction = friction;
    return p;
}

Primitive Primitive::makeSphere(Vec3 centre, Real radius, Real friction) {
    Primitive p;
    p.type = kSphere;
    p.a = centre;
    p.radius = radius;
    p.friction = friction;
    return p;
}

Primitive Primitive::makeCapsule(Vec3 endA, Vec3 endB, Real radius, Real friction) {
    Primitive p;
    p.type = kCapsule;
    p.a = endA;
    p.b = endB;
    p.radius = radius;
    p.friction = friction;
    return p;
}

Primitive Primitive::makeBox(Vec3 centre, Vec3 halfExtents, Quat rotation, Real friction) {
    Primitive p;
    p.type = kBox;
    p.a = centre;
    p.halfExtents = halfExtents;
    p.rotation = rotation;
    p.friction = friction;
    return p;
}

namespace {

Real clamp01(Real v) { return v < Real(0) ? Real(0) : (v > Real(1) ? Real(1) : v); }

// Nearest point on segment [a, b] to p, as the parameter along the segment.
Real closestParamOnSegment(Vec3 a, Vec3 b, Vec3 p) {
    const Vec3 ab = b - a;
    const Real len2 = norm2(ab);
    if (len2 < Real(1e-30)) return Real(0);
    return clamp01(dot(p - a, ab) / len2);
}

// A normal that is at least well-defined when p sits exactly on the axis or
// centre of a primitive, where the true normal is undefined.
Vec3 fallbackNormal() { return Vec3(0, 0, 1); }

Real distanceToSphereLike(Vec3 centre, Real radius, Vec3 p, Vec3& outNormal) {
    const Vec3 d = p - centre;
    const Real len = norm(d);
    outNormal = len > Real(1e-12) ? d / len : fallbackNormal();
    return len - radius;
}

Real distanceToBox(const Primitive& prim, Vec3 p, Vec3& outNormal) {
    // Work in the box's frame, where the problem is axis-aligned.
    const Vec3 local = rotateInv(prim.rotation, p - prim.a);
    const Vec3 h = prim.halfExtents;

    Vec3 clamped(std::min(std::max(local.x, -h.x), h.x), std::min(std::max(local.y, -h.y), h.y),
                 std::min(std::max(local.z, -h.z), h.z));
    const Vec3 delta = local - clamped;
    const Real outside = norm(delta);

    if (outside > Real(1e-12)) {
        outNormal = rotate(prim.rotation, delta / outside);
        return outside;
    }

    // Inside: the nearest surface is the closest face, and the distance is
    // negative. Pick the face with the smallest penetration.
    const Real dx = h.x - std::abs(local.x);
    const Real dy = h.y - std::abs(local.y);
    const Real dz = h.z - std::abs(local.z);
    Vec3 n;
    Real depth;
    if (dx <= dy && dx <= dz) {
        n = Vec3(local.x >= Real(0) ? Real(1) : Real(-1), 0, 0);
        depth = dx;
    } else if (dy <= dz) {
        n = Vec3(0, local.y >= Real(0) ? Real(1) : Real(-1), 0);
        depth = dy;
    } else {
        n = Vec3(0, 0, local.z >= Real(0) ? Real(1) : Real(-1));
        depth = dz;
    }
    outNormal = rotate(prim.rotation, n);
    return -depth;
}

}  // namespace

Real signedDistance(const Primitive& prim, Vec3 p, Vec3& outNormal) {
    switch (prim.type) {
        case Primitive::kPlane:
            outNormal = prim.normal;
            return dot(p - prim.a, prim.normal);
        case Primitive::kSphere:
            return distanceToSphereLike(prim.a, prim.radius, p, outNormal);
        case Primitive::kCapsule: {
            const Real t = closestParamOnSegment(prim.a, prim.b, p);
            const Vec3 axisPoint = prim.a + (prim.b - prim.a) * t;
            return distanceToSphereLike(axisPoint, prim.radius, p, outNormal);
        }
        case Primitive::kBox:
        default:
            return distanceToBox(prim, p, outNormal);
    }
}

// ---------------------------------------------------------------- segments

void closestPointsBetweenSegments(Vec3 p1, Vec3 q1, Vec3 p2, Vec3 q2, Real& s, Real& t) {
    const Vec3 d1 = q1 - p1, d2 = q2 - p2, r = p1 - p2;
    const Real a = norm2(d1), e = norm2(d2), f = dot(d2, r);
    const Real eps = Real(1e-30);

    if (a <= eps && e <= eps) {  // both degenerate
        s = t = Real(0);
        return;
    }
    if (a <= eps) {
        s = Real(0);
        t = clamp01(f / e);
        return;
    }
    const Real c = dot(d1, r);
    if (e <= eps) {
        t = Real(0);
        s = clamp01(-c / a);
        return;
    }

    const Real b = dot(d1, d2);
    const Real denom = a * e - b * b;
    // Parallel segments leave s free; pick 0 and let the clamping below fix t.
    s = denom > eps ? clamp01((b * f - c * e) / denom) : Real(0);
    t = (b * s + f) / e;

    // Clamping t may invalidate s, so recompute it against the clamped t.
    if (t < Real(0)) {
        t = Real(0);
        s = clamp01(-c / a);
    } else if (t > Real(1)) {
        t = Real(1);
        s = clamp01((b - c) / a);
    }
}

// ---------------------------------------------------------------- hash

uint32_t hashCell(int ix, int iy, int iz, int tableSize) {
    // Three large primes; the classic Teschner et al. spatial hash. The xor of
    // the scaled coordinates is cheap on a GPU and spreads well enough that the
    // bucket occupancy stays flat for rod-shaped inputs.
    const uint32_t h = (static_cast<uint32_t>(ix) * 73856093u) ^
                       (static_cast<uint32_t>(iy) * 19349663u) ^
                       (static_cast<uint32_t>(iz) * 83492791u);
    return h % static_cast<uint32_t>(tableSize);
}

namespace {

int cellCoord(Real v, Real cellSize) {
    return static_cast<int>(std::floor(v / cellSize));
}

}  // namespace

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

            const Vec3 n = dist > Real(1e-12) ? delta / dist : fallbackNormal();
            const int idx[4] = {rod.stretch.p0[j], rod.stretch.p1[j], rod.stretch.p0[k],
                                rod.stretch.p1[k]};
            const Real w[4] = {Real(1) - u, u, -(Real(1) - v), -v};
            out.contacts.push_back(makeContact(s, idx, w, 4, n, separation, world.selfFriction));
        }
    }
}

}  // namespace crs
