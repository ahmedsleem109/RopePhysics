// Collision geometry shared by the CPU solver and the CUDA kernels.
//
// Primitives, signed distances and segment closest points are written once,
// templated on the scalar, the same way math3.h is: double on the host, float
// on the device. Keeping a single copy is what lets a GPU contact trajectory be
// compared against the CPU one as a statement about the port.
#pragma once

#include "math3.h"

namespace crs {

// ---------------------------------------------------------------- primitives

// Flat POD with a type tag rather than a class hierarchy, so an array of them
// uploads to the device as-is.
template <typename T>
struct TPrimitive {
    enum Type { kPlane = 0, kSphere, kCapsule, kBox };

    int type = kPlane;
    TVec3<T> a;            // plane: a point on it; sphere: centre; capsule: end A; box: centre
    TVec3<T> b;            // capsule: end B
    TVec3<T> normal;       // plane: unit outward normal
    TVec3<T> halfExtents;  // box
    TQuat<T> rotation;     // box orientation (body -> world)
    T radius = 0;          // sphere / capsule
    T friction = 0;

    static TPrimitive makePlane(TVec3<T> point, TVec3<T> unitNormal, T friction) {
        TPrimitive p;
        p.type = kPlane;
        p.a = point;
        p.normal = normalize(unitNormal);
        p.friction = friction;
        return p;
    }
    static TPrimitive makeSphere(TVec3<T> centre, T radius, T friction) {
        TPrimitive p;
        p.type = kSphere;
        p.a = centre;
        p.radius = radius;
        p.friction = friction;
        return p;
    }
    static TPrimitive makeCapsule(TVec3<T> endA, TVec3<T> endB, T radius, T friction) {
        TPrimitive p;
        p.type = kCapsule;
        p.a = endA;
        p.b = endB;
        p.radius = radius;
        p.friction = friction;
        return p;
    }
    static TPrimitive makeBox(TVec3<T> centre, TVec3<T> halfExtents, TQuat<T> rotation,
                              T friction) {
        TPrimitive p;
        p.type = kBox;
        p.a = centre;
        p.halfExtents = halfExtents;
        p.rotation = rotation;
        p.friction = friction;
        return p;
    }
};

namespace geometry_detail {

template <typename T>
CRS_HD inline T clamp01(T v) {
    return v < T(0) ? T(0) : (v > T(1) ? T(1) : v);
}
template <typename T>
CRS_HD inline T absT(T v) {
    return v < T(0) ? -v : v;
}
template <typename T>
CRS_HD inline T clampT(T v, T lo, T hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// A normal that is at least well-defined when p sits exactly on the axis or
// centre of a primitive, where the true normal is undefined.
template <typename T>
CRS_HD inline TVec3<T> fallbackNormal() {
    return TVec3<T>(T(0), T(0), T(1));
}

// Nearest point on segment [a, b] to p, as the parameter along the segment.
template <typename T>
CRS_HD inline T closestParamOnSegment(TVec3<T> a, TVec3<T> b, TVec3<T> p) {
    const TVec3<T> ab = b - a;
    const T len2 = norm2(ab);
    if (len2 < T(1e-30)) return T(0);
    return clamp01(dot(p - a, ab) / len2);
}

template <typename T>
CRS_HD inline T distanceToSphereLike(TVec3<T> centre, T radius, TVec3<T> p, TVec3<T>& outNormal) {
    const TVec3<T> d = p - centre;
    const T len = norm(d);
    outNormal = len > T(1e-12) ? d / len : fallbackNormal<T>();
    return len - radius;
}

template <typename T>
CRS_HD inline T distanceToBox(const TPrimitive<T>& prim, TVec3<T> p, TVec3<T>& outNormal) {
    // Work in the box's frame, where the problem is axis-aligned.
    const TVec3<T> local = rotateInv(prim.rotation, p - prim.a);
    const TVec3<T> h = prim.halfExtents;

    const TVec3<T> clamped(clampT(local.x, -h.x, h.x), clampT(local.y, -h.y, h.y),
                           clampT(local.z, -h.z, h.z));
    const TVec3<T> delta = local - clamped;
    const T outside = norm(delta);

    if (outside > T(1e-12)) {
        outNormal = rotate(prim.rotation, delta / outside);
        return outside;
    }

    // Inside: the nearest surface is the closest face, and the distance is
    // negative. Pick the face with the smallest penetration.
    const T dx = h.x - absT(local.x);
    const T dy = h.y - absT(local.y);
    const T dz = h.z - absT(local.z);
    TVec3<T> n;
    T depth;
    if (dx <= dy && dx <= dz) {
        n = TVec3<T>(local.x >= T(0) ? T(1) : T(-1), T(0), T(0));
        depth = dx;
    } else if (dy <= dz) {
        n = TVec3<T>(T(0), local.y >= T(0) ? T(1) : T(-1), T(0));
        depth = dy;
    } else {
        n = TVec3<T>(T(0), T(0), local.z >= T(0) ? T(1) : T(-1));
        depth = dz;
    }
    outNormal = rotate(prim.rotation, n);
    return -depth;
}

}  // namespace geometry_detail

// Signed distance from `p` to the primitive's surface (positive outside), with
// the outward unit normal at the nearest surface point.
template <typename T>
CRS_HD inline T signedDistance(const TPrimitive<T>& prim, TVec3<T> p, TVec3<T>& outNormal) {
    using namespace geometry_detail;
    switch (prim.type) {
        case TPrimitive<T>::kPlane:
            outNormal = prim.normal;
            return dot(p - prim.a, prim.normal);
        case TPrimitive<T>::kSphere:
            return distanceToSphereLike(prim.a, prim.radius, p, outNormal);
        case TPrimitive<T>::kCapsule: {
            const T t = closestParamOnSegment(prim.a, prim.b, p);
            const TVec3<T> axisPoint = prim.a + (prim.b - prim.a) * t;
            return distanceToSphereLike(axisPoint, prim.radius, p, outNormal);
        }
        case TPrimitive<T>::kBox:
        default:
            return distanceToBox(prim, p, outNormal);
    }
}

// Closest points between two segments, returned as the parameters along each.
// Standard clamped-parametric solution; handles parallel and degenerate cases.
template <typename T>
CRS_HD inline void closestPointsBetweenSegments(TVec3<T> p1, TVec3<T> q1, TVec3<T> p2, TVec3<T> q2,
                                                T& s, T& t) {
    using geometry_detail::clamp01;
    const TVec3<T> d1 = q1 - p1, d2 = q2 - p2, r = p1 - p2;
    const T a = norm2(d1), e = norm2(d2), f = dot(d2, r);
    const T eps = T(1e-30);

    if (a <= eps && e <= eps) {  // both degenerate
        s = t = T(0);
        return;
    }
    if (a <= eps) {
        s = T(0);
        t = clamp01(f / e);
        return;
    }
    const T c = dot(d1, r);
    if (e <= eps) {
        t = T(0);
        s = clamp01(-c / a);
        return;
    }

    const T b = dot(d1, d2);
    const T denom = a * e - b * b;
    // Parallel segments leave s free; pick 0 and let the clamping below fix t.
    s = denom > eps ? clamp01((b * f - c * e) / denom) : T(0);
    t = (b * s + f) / e;

    // Clamping t may invalidate s, so recompute it against the clamped t.
    if (t < T(0)) {
        t = T(0);
        s = clamp01(-c / a);
    } else if (t > T(1)) {
        t = T(1);
        s = clamp01((b - c) / a);
    }
}

// ---------------------------------------------------------------- spatial hash

// Three large primes; the classic Teschner et al. spatial hash. The xor of the
// scaled coordinates is cheap on a GPU and spreads well enough that bucket
// occupancy stays flat for rod-shaped inputs.
CRS_HD inline unsigned hashCell(int ix, int iy, int iz, int tableSize) {
    const unsigned h = (static_cast<unsigned>(ix) * 73856093u) ^
                       (static_cast<unsigned>(iy) * 19349663u) ^
                       (static_cast<unsigned>(iz) * 83492791u);
    return h % static_cast<unsigned>(tableSize);
}

template <typename T>
CRS_HD inline int cellCoord(T v, T cellSize) {
    return static_cast<int>(crsFloor(v / cellSize));
}

using Primitive = TPrimitive<Real>;
using Primitivef = TPrimitive<float>;

}  // namespace crs
