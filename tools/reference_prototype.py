"""NumPy mirror of src/core -- an independent check on the C++ reference.

A port of the same XPBD Cosserat formulation: same constraints, same Jacobians,
same compliance parameterization, same substepping, same explicit constraint
arrays. It exists for two reasons:

  1. It validates the derivation without needing a C++ toolchain.
  2. Two independent implementations agreeing on the validation numbers is a
     much stronger statement than one implementation agreeing with itself.

Deliberately slow and readable. The C++ is the reference; this is the second
opinion. Run it directly to print the validation table.
"""

from __future__ import annotations

import math

import numpy as np

E3 = np.array([0.0, 0.0, 1.0])


# ----------------------------------------------------------------- quaternions
# (w, x, y, z), unit length, body -> world.

def qmul(a, b):
    aw, av = a[0], a[1:]
    bw, bv = b[0], b[1:]
    return np.concatenate(([aw * bw - av @ bv], aw * bv + bw * av + np.cross(av, bv)))


def qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def qnorm(q):
    return q / np.linalg.norm(q)


def qrot(q, v):
    u = q[1:]
    t = 2.0 * np.cross(u, v)
    return v + q[0] * t + np.cross(u, t)


def qmat(q):
    return np.column_stack([qrot(q, np.eye(3)[i]) for i in range(3)])


def qapply_body_delta(q, theta):
    return qnorm(qmul(q, np.concatenate(([1.0], 0.5 * theta))))


def qfrom_to(a, b):
    d = float(a @ b)
    if d > 1 - 1e-12:
        return np.array([1.0, 0, 0, 0])
    if d < -1 + 1e-12:
        axis = np.cross(a, [1.0, 0, 0])
        if axis @ axis < 1e-12:
            axis = np.cross(a, [0.0, 1, 0])
        return np.concatenate(([0.0], axis / np.linalg.norm(axis)))
    c = np.cross(a, b)
    s = math.sqrt((1 + d) * 2)
    return qnorm(np.concatenate(([0.5 * s], c / s)))


def same_hemisphere(q):
    return -q if q[0] < 0 else q


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


# ----------------------------------------------------------------- material

class Material:
    def __init__(self, E=1e7, nu=0.35, rho=1000.0, r=5e-3):
        self.E, self.nu, self.rho, self.r = E, nu, rho, r

    G = property(lambda s: s.E / (2 * (1 + s.nu)))
    A = property(lambda s: math.pi * s.r ** 2)
    I = property(lambda s: math.pi * s.r ** 4 / 4)
    J = property(lambda s: 2 * s.I)
    ks = property(lambda s: 6 * (1 + s.nu) / (7 + 6 * s.nu))
    EI = property(lambda s: s.E * s.I)
    GJ = property(lambda s: s.G * s.J)


# ----------------------------------------------------------------- rod

class Rod:
    """Particles carry the centerline, segment frames carry the material frame.

    Constraints are stored as explicit index arrays so that boundary elements
    (ghost frames, half-length elements) are ordinary entries rather than
    special cases in the solver.
    """

    def __init__(self, n, length, mat: Material, direction=(1.0, 0.0, 0.0)):
        d = np.asarray(direction, float)
        d = d / np.linalg.norm(d)
        self.mat = mat
        self.l = length / n
        self.x = np.outer(np.arange(n + 1) * self.l, d)
        self.v = np.zeros_like(self.x)
        self.ext_force = np.zeros_like(self.x)

        seg_mass = mat.rho * mat.A * self.l
        self.mass = np.full(n + 1, seg_mass)
        self.mass[0] = self.mass[-1] = 0.5 * seg_mass
        self.inv_mass = 1.0 / self.mass

        self.q = np.tile(qfrom_to(E3, d), (n, 1))
        self.omega = np.zeros((n, 3))
        self.ext_torque = np.zeros((n, 3))
        it = seg_mass * (3 * mat.r ** 2 + self.l ** 2) / 12
        ia = 0.5 * seg_mass * mat.r ** 2
        self.inertia = np.tile([it, it, ia], (n, 1))
        self.inv_inertia = 1.0 / self.inertia

        self.s_p0 = np.arange(n)
        self.s_p1 = np.arange(1, n + 1)
        self.s_seg = np.arange(n)
        self.s_l = np.full(n, self.l)
        self.s_alpha = np.array([self._alpha_s(self.l)] * n)
        self.s_lam = np.zeros((n, 3))

        self.b_a = np.arange(n - 1)
        self.b_b = np.arange(1, n)
        self.b_l = np.full(n - 1, self.l)
        self.b_rest = np.zeros((n - 1, 3))
        self.b_alpha = np.array([self._alpha_b(self.l)] * (n - 1))
        self.b_lam = np.zeros((n - 1, 3))

    # Compliance = (stiffness per unit length)^-1 / element length.
    def _alpha_s(self, l):
        m = self.mat
        EA, GA = m.E * m.A, m.ks * m.G * m.A
        return [1 / (GA * l), 1 / (GA * l), 1 / (EA * l)]

    def _alpha_b(self, l):
        m = self.mat
        return [1 / (m.EI * l), 1 / (m.EI * l), 1 / (m.GJ * l)]

    @property
    def n_seg(self):
        return len(self.q)

    def add_frame(self, q, fixed=True):
        """Append an orientation element. A fixed one is a kinematic boundary
        frame: it has no inertia, never moves, and carries no kinetic energy."""
        assert fixed, "only fixed ghost frames are needed so far"
        self.q = np.vstack([self.q, qnorm(np.asarray(q, float))])
        self.omega = np.vstack([self.omega, np.zeros(3)])
        self.ext_torque = np.vstack([self.ext_torque, np.zeros(3)])
        self.inertia = np.vstack([self.inertia, np.zeros(3)])
        self.inv_inertia = np.vstack([self.inv_inertia, np.zeros(3)])
        return len(self.q) - 1

    def add_bend(self, a, b, lbar, rest=(0.0, 0.0, 0.0)):
        self.b_a = np.append(self.b_a, a)
        self.b_b = np.append(self.b_b, b)
        self.b_l = np.append(self.b_l, lbar)
        self.b_rest = np.vstack([self.b_rest, np.asarray(rest, float)])
        self.b_alpha = np.vstack([self.b_alpha, self._alpha_b(lbar)])
        self.b_lam = np.vstack([self.b_lam, np.zeros(3)])

    # --- boundary conditions ---

    def clamp_root_rigid(self):
        """Fix particle 0 and freeze segment 0's orientation.

        The naive clamp. Segment 0's frame lives at s = l/2, so freezing it
        clamps the rod half an element in from the end: the effective length is
        L - l/2 and the tip deflection carries an O(h) error of -1.5 h / L.
        """
        self.inv_mass[0] = 0.0
        self.inv_inertia[0] = 0.0

    def clamp_root_ghost(self):
        """Fix particle 0 and attach segment 0 to a fixed frame AT s = 0 through
        a half-length bend element.

        This puts the clamped boundary condition where it belongs, at s = 0, and
        restores second-order convergence of the tip deflection.
        """
        self.inv_mass[0] = 0.0
        g = self.add_frame(self.q[0])
        self.add_bend(g, 0, 0.5 * self.l)


# ----------------------------------------------------------------- solver

def project(rod: Rod, h: float):
    a_scale = 1.0 / (h * h)

    # stretch / shear: C = (x1 - x0)/l - R e3
    for k in range(len(rod.s_seg)):
        i0, i1, j = rod.s_p0[k], rod.s_p1[k], rod.s_seg[k]
        w0, w1 = rod.inv_mass[i0], rod.inv_mass[i1]
        iI = rod.inv_inertia[j]
        if w0 == 0 and w1 == 0 and not iI.any():
            continue
        l = rod.s_l[k]
        q = rod.q[j]
        R = qmat(q)
        C = (rod.x[i1] - rod.x[i0]) / l - R[:, 2]
        alpha = rod.s_alpha[k] * a_scale
        A = (w0 + w1) / l ** 2 * np.eye(3)
        A += R @ np.diag([iI[1], iI[0], 0.0]) @ R.T
        A += np.diag(alpha)
        dl = np.linalg.solve(A, -(C + alpha * rod.s_lam[k]))
        rod.s_lam[k] += dl
        rod.x[i0] -= w0 / l * dl
        rod.x[i1] += w1 / l * dl
        if iI.any():
            body = qrot(qconj(q), dl)
            rod.q[j] = qapply_body_delta(q, iI * (-np.cross(E3, body)))

    # bend / twist: C = 2 Im(conj(qa) qb)/lbar - Omega0
    for k in range(len(rod.b_a)):
        a, b = rod.b_a[k], rod.b_b[k]
        iIa, iIb = rod.inv_inertia[a], rod.inv_inertia[b]
        if not iIa.any() and not iIb.any():
            continue
        lbar = rod.b_l[k]
        p = same_hemisphere(qmul(qconj(rod.q[a]), rod.q[b]))
        pw, pv = p[0], p[1:]
        C = 2 * pv / lbar - rod.b_rest[k]
        Ja = (-pw * np.eye(3) + skew(pv)) / lbar
        Jb = (pw * np.eye(3) + skew(pv)) / lbar
        alpha = rod.b_alpha[k] * a_scale
        A = Ja @ np.diag(iIa) @ Ja.T + Jb @ np.diag(iIb) @ Jb.T + np.diag(alpha)
        dl = np.linalg.solve(A, -(C + alpha * rod.b_lam[k]))
        rod.b_lam[k] += dl
        if iIa.any():
            rod.q[a] = qapply_body_delta(rod.q[a], iIa * (Ja.T @ dl))
        if iIb.any():
            rod.q[b] = qapply_body_delta(rod.q[b], iIb * (Jb.T @ dl))


def step(rod: Rod, dt, substeps, gravity=np.zeros(3), damping=0.0, iterations=1):
    h = dt / substeps
    free_p = rod.inv_mass > 0
    free_s = np.flatnonzero(rod.inv_inertia.any(axis=1))
    fixed_s = np.flatnonzero(~rod.inv_inertia.any(axis=1))
    for _ in range(substeps):
        rod.s_lam[:] = 0.0
        rod.b_lam[:] = 0.0

        x_prev = rod.x.copy()
        q_prev = rod.q.copy()
        rod.v[free_p] += h * (gravity + rod.ext_force[free_p] * rod.inv_mass[free_p, None])
        rod.x[free_p] += h * rod.v[free_p]
        for j in free_s:
            w = rod.omega[j]
            tau = qrot(qconj(rod.q[j]), rod.ext_torque[j])
            rod.omega[j] = w + h * rod.inv_inertia[j] * (tau - np.cross(w, rod.inertia[j] * w))
            rod.q[j] = qnorm(rod.q[j] + 0.5 * h * qmul(rod.q[j],
                                                       np.concatenate(([0.0], rod.omega[j]))))

        for _ in range(iterations):
            project(rod, h)

        decay = math.exp(-damping * h)
        rod.v[free_p] = (rod.x[free_p] - x_prev[free_p]) / h * decay
        rod.v[~free_p] = 0.0
        for j in free_s:
            dq = same_hemisphere(qmul(qconj(q_prev[j]), rod.q[j]))
            rod.omega[j] = 2 * dq[1:] / h * decay
        rod.omega[fixed_s] = 0.0


def relax(rod: Rod, dt, substeps, damping, max_steps, tol, gravity=np.zeros(3), iterations=1):
    for i in range(max_steps):
        step(rod, dt, substeps, gravity, damping, iterations)
        if np.abs(rod.v).max() < tol and np.abs(rod.omega).max() < tol:
            return i + 1, True
    return max_steps, False


def omega1(mat, L):
    """First bending eigenfrequency of a cantilever; the relaxation rate scale."""
    return 3.516 * math.sqrt(mat.EI / (mat.rho * mat.A * L ** 4))


# ----------------------------------------------------------------- cross-check
#
# Reproduces the trajectory that `rodsim crosscheck` writes to
# docs/data/crosscheck.csv, using the same setup, the same step parameters and
# no adaptivity anywhere. Agreement to round-off between two independently
# written implementations is the point of this file.

CROSSCHECK_SEGMENTS = 16
CROSSCHECK_STEPS = 200


def crosscheck_state():
    rod = Rod(CROSSCHECK_SEGMENTS, 1.0, Material())
    rod.clamp_root_ghost()
    gravity = np.array([0.0, 0.0, -9.81])
    for _ in range(CROSSCHECK_STEPS):
        step(rod, 1e-3, 4, gravity, damping=0.0, iterations=1)
    return rod


def crosscheck(path="docs/data/crosscheck.csv"):
    import csv as _csv

    rod = crosscheck_state()
    with open(path, newline="") as f:
        rows = list(_csv.DictReader(f))

    dx = 0.0
    dq = 0.0
    for i, r in enumerate(rows):
        ref = np.array([float(r["x"]), float(r["y"]), float(r["z"])])
        dx = max(dx, float(np.max(np.abs(rod.x[i] - ref))))
        if i < rod.n_seg:
            refq = np.array([float(r["qw"]), float(r["qx"]), float(r["qy"]), float(r["qz"])])
            q = rod.q[i]
            # Compare on the same hemisphere: +q and -q are the same rotation.
            if q @ refq < 0:
                refq = -refq
            dq = max(dq, float(np.max(np.abs(q - refq))))
    return dx, dq


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "docs/data/crosscheck.csv"
    print(f"cross-checking against {path}")
    print(f"  {CROSSCHECK_SEGMENTS} segments, {CROSSCHECK_STEPS} steps, dt=1e-3, 4 substeps")
    dx, dq = crosscheck(path)
    print(f"  max |dx| = {dx:.3e} m")
    print(f"  max |dq| = {dq:.3e}")
    ok = dx < 1e-9 and dq < 1e-9
    print("  PASS" if ok else "  FAIL -- the two implementations disagree beyond round-off")
    sys.exit(0 if ok else 1)
