// Small fixed-size linear algebra for the Cosserat rod solver.
//
// Templated on the scalar type so that exactly this code compiles as host
// double precision for the CPU reference and as __device__ float for the CUDA
// port. That is not tidiness for its own sake: the Phase 3 gate is that the GPU
// reproduces CPU trajectories, and two hand-written copies of quaternion
// algebra is the surest way to fail that gate for reasons having nothing to do
// with the GPU.
#pragma once

#include <cmath>

namespace crs {

// The CPU reference scalar; device code uses TVec3<float>. CRS_REAL_FLOAT builds
// the reference in single precision, which is how GPU-vs-CPU differences are
// split into "float rounding" and "kernel bug".
#ifdef CRS_REAL_FLOAT
using Real = float;
#else
using Real = double;
#endif

constexpr Real kPi = Real(3.14159265358979323846);

#ifdef __CUDACC__
#define CRS_HD __host__ __device__
#else
#define CRS_HD
#endif

// Scalar helpers that pick the right precision on both sides.
CRS_HD inline float crsSqrt(float v) { return sqrtf(v); }
CRS_HD inline double crsSqrt(double v) { return sqrt(v); }
CRS_HD inline float crsFloor(float v) { return floorf(v); }
CRS_HD inline double crsFloor(double v) { return floor(v); }
CRS_HD inline float crsCos(float v) { return cosf(v); }
CRS_HD inline double crsCos(double v) { return cos(v); }
CRS_HD inline float crsSin(float v) { return sinf(v); }
CRS_HD inline double crsSin(double v) { return sin(v); }

// ---------------------------------------------------------------- Vec3

template <typename T>
struct TVec3 {
    T x, y, z;

    CRS_HD TVec3() : x(0), y(0), z(0) {}
    CRS_HD TVec3(T x_, T y_, T z_) : x(x_), y(y_), z(z_) {}

    CRS_HD T operator[](int i) const { return (&x)[i]; }
    CRS_HD T& operator[](int i) { return (&x)[i]; }
};

template <typename T>
CRS_HD inline TVec3<T> operator+(TVec3<T> a, TVec3<T> b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}
template <typename T>
CRS_HD inline TVec3<T> operator-(TVec3<T> a, TVec3<T> b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}
template <typename T>
CRS_HD inline TVec3<T> operator-(TVec3<T> a) {
    return {-a.x, -a.y, -a.z};
}
template <typename T>
CRS_HD inline TVec3<T> operator*(TVec3<T> a, T s) {
    return {a.x * s, a.y * s, a.z * s};
}
template <typename T>
CRS_HD inline TVec3<T> operator*(T s, TVec3<T> a) {
    return a * s;
}
template <typename T>
CRS_HD inline TVec3<T> operator/(TVec3<T> a, T s) {
    return a * (T(1) / s);
}
template <typename T>
CRS_HD inline TVec3<T>& operator+=(TVec3<T>& a, TVec3<T> b) {
    a = a + b;
    return a;
}
template <typename T>
CRS_HD inline TVec3<T>& operator-=(TVec3<T>& a, TVec3<T> b) {
    a = a - b;
    return a;
}
template <typename T>
CRS_HD inline TVec3<T>& operator*=(TVec3<T>& a, T s) {
    a = a * s;
    return a;
}

template <typename T>
CRS_HD inline T dot(TVec3<T> a, TVec3<T> b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
template <typename T>
CRS_HD inline TVec3<T> cross(TVec3<T> a, TVec3<T> b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
template <typename T>
CRS_HD inline T norm2(TVec3<T> a) {
    return dot(a, a);
}
template <typename T>
CRS_HD inline T norm(TVec3<T> a) {
    return crsSqrt(dot(a, a));
}
template <typename T>
CRS_HD inline TVec3<T> normalize(TVec3<T> a) {
    return a / norm(a);
}
// Componentwise product; inverse inertia is diagonal in the body frame so this
// is how M^-1 * v is spelled everywhere below.
template <typename T>
CRS_HD inline TVec3<T> cwise(TVec3<T> a, TVec3<T> b) {
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

// ---------------------------------------------------------------- Mat3
// Row-major 3x3. Used only for the small dense system each constraint solves.

template <typename T>
struct TMat3 {
    T m[3][3];

    CRS_HD static TMat3 zero() {
        TMat3 r;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) r.m[i][j] = T(0);
        return r;
    }
    CRS_HD static TMat3 identity(T s = T(1)) {
        TMat3 r = zero();
        r.m[0][0] = r.m[1][1] = r.m[2][2] = s;
        return r;
    }
    // Columns are the basis vectors of the frame this matrix rotates into.
    CRS_HD static TMat3 fromCols(TVec3<T> c0, TVec3<T> c1, TVec3<T> c2) {
        TMat3 r;
        for (int i = 0; i < 3; ++i) {
            r.m[i][0] = c0[i];
            r.m[i][1] = c1[i];
            r.m[i][2] = c2[i];
        }
        return r;
    }
    // Cross-product matrix: skew(v) * u == cross(v, u).
    CRS_HD static TMat3 skew(TVec3<T> v) {
        TMat3 r = zero();
        r.m[0][1] = -v.z;
        r.m[0][2] = v.y;
        r.m[1][0] = v.z;
        r.m[1][2] = -v.x;
        r.m[2][0] = -v.y;
        r.m[2][1] = v.x;
        return r;
    }
    CRS_HD TVec3<T> col(int j) const { return {m[0][j], m[1][j], m[2][j]}; }
};

template <typename T>
CRS_HD inline TVec3<T> operator*(const TMat3<T>& A, TVec3<T> v) {
    return {A.m[0][0] * v.x + A.m[0][1] * v.y + A.m[0][2] * v.z,
            A.m[1][0] * v.x + A.m[1][1] * v.y + A.m[1][2] * v.z,
            A.m[2][0] * v.x + A.m[2][1] * v.y + A.m[2][2] * v.z};
}
template <typename T>
CRS_HD inline TMat3<T> operator+(const TMat3<T>& A, const TMat3<T>& B) {
    TMat3<T> r;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) r.m[i][j] = A.m[i][j] + B.m[i][j];
    return r;
}
template <typename T>
CRS_HD inline TMat3<T> operator*(const TMat3<T>& A, T s) {
    TMat3<T> r;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) r.m[i][j] = A.m[i][j] * s;
    return r;
}
template <typename T>
CRS_HD inline TMat3<T> transpose(const TMat3<T>& A) {
    TMat3<T> r;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) r.m[i][j] = A.m[j][i];
    return r;
}
// A * diag(d) * A^T, the shape every J M^-1 J^T block takes.
template <typename T>
CRS_HD inline TMat3<T> sandwichDiag(const TMat3<T>& A, TVec3<T> d) {
    TMat3<T> r;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            r.m[i][j] = A.m[i][0] * d.x * A.m[j][0] + A.m[i][1] * d.y * A.m[j][1] +
                        A.m[i][2] * d.z * A.m[j][2];
    return r;
}

// Solve the symmetric positive-definite system A y = b by 3x3 Cholesky.
// Every system built below is SPD: J M^-1 J^T is positive semi-definite and the
// compliance added to the diagonal is strictly positive, so a failure here is a
// modelling bug rather than a numerical accident.
template <typename T>
CRS_HD inline bool solveSPD3(const TMat3<T>& A, TVec3<T> b, TVec3<T>& y) {
    T l00 = A.m[0][0];
    if (l00 <= T(0)) return false;
    l00 = crsSqrt(l00);
    const T l10 = A.m[1][0] / l00;
    const T l20 = A.m[2][0] / l00;
    T l11 = A.m[1][1] - l10 * l10;
    if (l11 <= T(0)) return false;
    l11 = crsSqrt(l11);
    const T l21 = (A.m[2][1] - l20 * l10) / l11;
    T l22 = A.m[2][2] - l20 * l20 - l21 * l21;
    if (l22 <= T(0)) return false;
    l22 = crsSqrt(l22);

    const T z0 = b.x / l00;
    const T z1 = (b.y - l10 * z0) / l11;
    const T z2 = (b.z - l20 * z0 - l21 * z1) / l22;

    y.z = z2 / l22;
    y.y = (z1 - l21 * y.z) / l11;
    y.x = (z0 - l10 * y.y - l20 * y.z) / l00;
    return true;
}

// ---------------------------------------------------------------- Quat
// (w, v) convention, unit length. Maps body -> world.

template <typename T>
struct TQuat {
    T w, x, y, z;

    CRS_HD TQuat() : w(1), x(0), y(0), z(0) {}
    CRS_HD TQuat(T w_, T x_, T y_, T z_) : w(w_), x(x_), y(y_), z(z_) {}
    CRS_HD TQuat(T w_, TVec3<T> v) : w(w_), x(v.x), y(v.y), z(v.z) {}

    CRS_HD TVec3<T> im() const { return {x, y, z}; }
};

template <typename T>
CRS_HD inline TQuat<T> operator*(TQuat<T> a, TQuat<T> b) {
    const TVec3<T> av = a.im(), bv = b.im();
    return TQuat<T>(a.w * b.w - dot(av, bv), a.w * bv + b.w * av + cross(av, bv));
}
template <typename T>
CRS_HD inline TQuat<T> conj(TQuat<T> q) {
    return {q.w, -q.x, -q.y, -q.z};
}
template <typename T>
CRS_HD inline TQuat<T> operator+(TQuat<T> a, TQuat<T> b) {
    return {a.w + b.w, a.x + b.x, a.y + b.y, a.z + b.z};
}
template <typename T>
CRS_HD inline TQuat<T> operator*(TQuat<T> q, T s) {
    return {q.w * s, q.x * s, q.y * s, q.z * s};
}
template <typename T>
CRS_HD inline T dot(TQuat<T> a, TQuat<T> b) {
    return a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z;
}

template <typename T>
CRS_HD inline TQuat<T> normalize(TQuat<T> q) {
    const T n = crsSqrt(dot(q, q));
    return q * (T(1) / n);
}

// Rotate a body-frame vector into world.
template <typename T>
CRS_HD inline TVec3<T> rotate(TQuat<T> q, TVec3<T> v) {
    const TVec3<T> u = q.im();
    const TVec3<T> t = cross(u, v) * T(2);
    return v + t * q.w + cross(u, t);
}
template <typename T>
CRS_HD inline TVec3<T> rotateInv(TQuat<T> q, TVec3<T> v) {
    return rotate(conj(q), v);
}

template <typename T>
CRS_HD inline TMat3<T> toMat3(TQuat<T> q) {
    return TMat3<T>::fromCols(rotate(q, TVec3<T>(1, 0, 0)), rotate(q, TVec3<T>(0, 1, 0)),
                              rotate(q, TVec3<T>(0, 0, 1)));
}

// Right-multiplied incremental rotation: q <- q * exp(theta/2), theta in the
// BODY frame. This is the perturbation every constraint Jacobian is taken with
// respect to, which is what lets body-frame inverse inertia be used raw.
template <typename T>
CRS_HD inline TQuat<T> applyBodyDelta(TQuat<T> q, TVec3<T> theta) {
    return normalize(q * TQuat<T>(T(1), theta * T(0.5)));
}

// Quaternion rotating unit vector `from` onto unit vector `to`.
template <typename T>
CRS_HD inline TQuat<T> quatFromTo(TVec3<T> from, TVec3<T> to) {
    const T d = dot(from, to);
    if (d > T(1) - T(1e-12)) return TQuat<T>();
    if (d < T(-1) + T(1e-12)) {
        // Antiparallel: any perpendicular axis works, pick a stable one.
        TVec3<T> axis = cross(from, TVec3<T>(1, 0, 0));
        if (norm2(axis) < T(1e-12)) axis = cross(from, TVec3<T>(0, 1, 0));
        axis = normalize(axis);
        return TQuat<T>(T(0), axis);
    }
    const TVec3<T> c = cross(from, to);
    const T s = crsSqrt((T(1) + d) * T(2));
    return normalize(TQuat<T>(s * T(0.5), c / s));
}

// Rotation matrix -> quaternion (Shepperd's method: pick the largest pivot so
// the square root never hits a near-zero denominator).
template <typename T>
CRS_HD inline TQuat<T> quatFromMat3(const TMat3<T>& R) {
    const T t = R.m[0][0] + R.m[1][1] + R.m[2][2];
    TQuat<T> q;
    if (t > T(0)) {
        T s = crsSqrt(t + T(1)) * T(2);
        q = TQuat<T>(T(0.25) * s, (R.m[2][1] - R.m[1][2]) / s, (R.m[0][2] - R.m[2][0]) / s,
                     (R.m[1][0] - R.m[0][1]) / s);
    } else if (R.m[0][0] > R.m[1][1] && R.m[0][0] > R.m[2][2]) {
        T s = crsSqrt(T(1) + R.m[0][0] - R.m[1][1] - R.m[2][2]) * T(2);
        q = TQuat<T>((R.m[2][1] - R.m[1][2]) / s, T(0.25) * s, (R.m[0][1] + R.m[1][0]) / s,
                     (R.m[0][2] + R.m[2][0]) / s);
    } else if (R.m[1][1] > R.m[2][2]) {
        T s = crsSqrt(T(1) + R.m[1][1] - R.m[0][0] - R.m[2][2]) * T(2);
        q = TQuat<T>((R.m[0][2] - R.m[2][0]) / s, (R.m[0][1] + R.m[1][0]) / s, T(0.25) * s,
                     (R.m[1][2] + R.m[2][1]) / s);
    } else {
        T s = crsSqrt(T(1) + R.m[2][2] - R.m[0][0] - R.m[1][1]) * T(2);
        q = TQuat<T>((R.m[1][0] - R.m[0][1]) / s, (R.m[0][2] + R.m[2][0]) / s,
                     (R.m[1][2] + R.m[2][1]) / s, T(0.25) * s);
    }
    return normalize(q);
}

template <typename T>
CRS_HD inline TQuat<T> quatAxisAngle(TVec3<T> axis, T angle) {
    const T h = angle * T(0.5);
    return TQuat<T>(crsCos(h), normalize(axis) * crsSin(h));
}

// Pick the representative on the same hemisphere as `ref`; the double cover
// otherwise makes the Darboux vector flip sign mid-simulation.
template <typename T>
CRS_HD inline TQuat<T> sameHemisphere(TQuat<T> q, TQuat<T> ref) {
    return dot(q, ref) < T(0) ? q * T(-1) : q;
}

// ---------------------------------------------------------------- aliases
// The CPU reference works in double; these names are what the rest of the
// non-device code uses. The float aliases are what the CUDA kernels use.

using Vec3 = TVec3<Real>;
using Mat3 = TMat3<Real>;
using Quat = TQuat<Real>;

using Vec3f = TVec3<float>;
using Mat3f = TMat3<float>;
using Quatf = TQuat<float>;

}  // namespace crs
