#include "solver.h"

#include <algorithm>
#include <cmath>

namespace crs {

namespace {

const Vec3 kE3(0, 0, 1);

// ---- stretch / shear -------------------------------------------------------
//
//   C = R(q)^T (x1 - x0) / l - e3                                  [3 vector]
//
// C lives in the MATERIAL frame, because that is the frame the compliance
// diag(shear, shear, axial) is written in. An earlier version measured C in
// world components, which applied the axial stiffness to whichever world axis
// happened to be z: a cantilever along x carried EA in shear and converged to
// a third of the Timoshenko shear deflection, while the same rod along z was
// correct. The static cantilever sweep to n = 256 is what exposed it.
//
// Jacobians, with the orientation perturbed by a BODY-frame rotation vector
// (q <- q exp(dtheta/2)), so body-frame inverse inertia can be used directly.
// With u = R^T (x1 - x0) / l, and R^T -> (I - skew(dtheta)) R^T:
//
//   dC/dx0 = -R^T/l      dC/dx1 = +R^T/l      dC/dtheta = skew(u)
//
// and therefore
//
//   J M^-1 J^T = (w0 + w1)/l^2 I + skew(u) diag(Iinv) skew(u)^T
//
// (spinning about the tangent u does not move the tangent, which is exactly
// why twist must come from the bend term).
void projectStretch(Rod& rod, Real h, const Coloring* coloring) {
    RodState& s = rod.state;
    StretchConstraints& c = rod.stretch;
    const Real invH2 = Real(1) / (h * h);
    const int n = static_cast<int>(c.size());

    for (int ii = 0; ii < n; ++ii) {
        const int k = coloring ? coloring->order[ii] : ii;
        const int i0 = c.p0[k], i1 = c.p1[k], j = c.seg[k];
        const Real w0 = s.invMass[i0], w1 = s.invMass[i1];
        const Vec3 iI = s.invInertia[j];
        if (w0 == Real(0) && w1 == Real(0) && norm2(iI) == Real(0)) continue;

        const Real l = c.restLength[k];
        const Quat q = s.q[j];
        const Vec3 u = rotateInv(q, s.x[i1] - s.x[i0]) / l;
        const Vec3 C = u - kE3;

        const Vec3 alpha = c.compliance[k] * invH2;
        Mat3 A = Mat3::identity((w0 + w1) / (l * l)) + sandwichDiag(Mat3::skew(u), iI);
        A.m[0][0] += alpha.x;
        A.m[1][1] += alpha.y;
        A.m[2][2] += alpha.z;

        Vec3 dLambda;
        if (!solveSPD3(A, -(C + cwise(alpha, c.lambda[k])), dLambda)) continue;
        c.lambda[k] += dLambda;

        const Vec3 world = rotate(q, dLambda);
        s.x[i0] -= world * (w0 / l);
        s.x[i1] += world * (w1 / l);
        if (norm2(iI) > Real(0)) {
            // J_theta^T dLambda = skew(u)^T dLambda = dLambda x u
            s.q[j] = applyBodyDelta(q, cwise(iI, cross(dLambda, u)));
        }
    }
}

// ---- bend / twist ----------------------------------------------------------
//
//   p = conj(q_a) q_b,    Omega = (2/lbar) Im(p),    C = Omega - Omega_rest
//
// Differentiating the quaternion product under body-frame perturbations:
//
//   dC/dtheta_a = (1/lbar) (-p_w I + skew(p_v))
//   dC/dtheta_b = (1/lbar) (+p_w I + skew(p_v))
void projectBend(Rod& rod, Real h, const Coloring* coloring) {
    RodState& s = rod.state;
    BendConstraints& c = rod.bend;
    const Real invH2 = Real(1) / (h * h);
    const int n = static_cast<int>(c.size());

    for (int ii = 0; ii < n; ++ii) {
        const int k = coloring ? coloring->order[ii] : ii;
        const int a = c.segA[k], b = c.segB[k];
        const Vec3 iIa = s.invInertia[a], iIb = s.invInertia[b];
        if (norm2(iIa) == Real(0) && norm2(iIb) == Real(0)) continue;

        const Real lbar = c.restLength[k];
        // Same hemisphere as the identity, so Im(p) stays a small rotation
        // vector rather than jumping to the antipodal representative.
        const Quat p = sameHemisphere(conj(s.q[a]) * s.q[b], Quat());
        const Vec3 C = p.im() * (Real(2) / lbar) - c.restDarboux[k];

        const Mat3 skewPv = Mat3::skew(p.im());
        const Mat3 Ja = (Mat3::identity(-p.w) + skewPv) * (Real(1) / lbar);
        const Mat3 Jb = (Mat3::identity(p.w) + skewPv) * (Real(1) / lbar);

        const Vec3 alpha = c.compliance[k] * invH2;
        Mat3 A = sandwichDiag(Ja, iIa) + sandwichDiag(Jb, iIb);
        A.m[0][0] += alpha.x;
        A.m[1][1] += alpha.y;
        A.m[2][2] += alpha.z;

        Vec3 dLambda;
        if (!solveSPD3(A, -(C + cwise(alpha, c.lambda[k])), dLambda)) continue;
        c.lambda[k] += dLambda;

        // J^T dLambda, using skew^T = -skew.
        if (norm2(iIa) > Real(0)) {
            const Vec3 g = (dLambda * -p.w - cross(p.im(), dLambda)) * (Real(1) / lbar);
            s.q[a] = applyBodyDelta(s.q[a], cwise(iIa, g));
        }
        if (norm2(iIb) > Real(0)) {
            const Vec3 g = (dLambda * p.w - cross(p.im(), dLambda)) * (Real(1) / lbar);
            s.q[b] = applyBodyDelta(s.q[b], cwise(iIb, g));
        }
    }
}

}  // namespace

int stableSubsteps(const Rod& rod, Vec3 gravity, Real dt, Real fraction, Real maxAngle) {
    const RodState& s = rod.state;
    Real hMax = dt;

    Real aMax = 0;
    for (std::size_t i = 0; i < s.numParticles(); ++i) {
        if (s.invMass[i] == Real(0)) continue;
        aMax = std::max(aMax, norm(gravity + s.extForce[i] * s.invMass[i]));
    }
    if (aMax > Real(0)) {
        Real lMin = rod.stretch.restLength.empty() ? Real(1) : rod.stretch.restLength[0];
        for (Real l : rod.stretch.restLength) lMin = std::min(lMin, l);
        hMax = std::min(hMax, std::sqrt(fraction * lMin / aMax));
    }

    Real alphaMax = 0;
    for (std::size_t j = 0; j < s.numSegments(); ++j) {
        if (norm2(s.invInertia[j]) == Real(0)) continue;
        alphaMax = std::max(alphaMax, norm(cwise(s.invInertia[j], s.extTorque[j])));
    }
    if (alphaMax > Real(0)) hMax = std::min(hMax, std::sqrt(maxAngle / alphaMax));

    return std::max(1, static_cast<int>(std::ceil(dt / hMax)));
}

void projectConstraints(Rod& rod, Real h, const Coloring* stretchColoring,
                        const Coloring* bendColoring) {
    projectStretch(rod, h, stretchColoring);
    projectBend(rod, h, bendColoring);
}

// Contacts are rigid and unilateral, so they carry no compliance and their
// multiplier is clamped at zero: a contact may push, never pull. Friction is
// applied straight after the normal solve for the same contact, so it always
// sees an up-to-date normal force -- doing every normal first and every
// friction afterwards makes a stack of contacts creep.
void projectContacts(Rod& rod, ContactSet& set) {
    RodState& s = rod.state;

    for (Contact& c : set.contacts) {
        Real wEff = 0;
        for (int k = 0; k < c.count; ++k)
            wEff += c.weight[k] * c.weight[k] * s.invMass[c.idx[k]];
        if (wEff <= Real(0)) continue;  // every participant pinned

        // --- non-penetration ---
        Real C = -c.offset;
        for (int k = 0; k < c.count; ++k) C += c.weight[k] * dot(s.x[c.idx[k]], c.normal);

        Real dLambda = -C / wEff;
        const Real clamped = std::max(c.lambdaN + dLambda, Real(0));
        dLambda = clamped - c.lambdaN;
        c.lambdaN = clamped;
        if (dLambda != Real(0))
            for (int k = 0; k < c.count; ++k)
                s.x[c.idx[k]] += c.normal * (s.invMass[c.idx[k]] * c.weight[k] * dLambda);

        if (c.friction <= Real(0) || c.lambdaN <= Real(0)) continue;

        // --- Coulomb friction, at the position level ---
        //
        // The tangential displacement the contact point has accumulated since
        // the substep began is what friction must undo. Undoing all of it is
        // sticking; the Coulomb cone caps the correction at mu * lambda_n, and
        // whatever slides past that cap is the dynamic-friction regime. One
        // expression covers both, with no branch on stick vs slip.
        //
        // The cap bounds the TOTAL tangential correction applied during this
        // substep, not each sweep's share. Applying a fresh full-size cap on
        // every sweep lets a solver spend `iterations` times the Coulomb limit,
        // and the symptom is not subtle: a block sits motionless on a slope far
        // steeper than atan(mu), because friction four times too strong still
        // looks exactly like friction.
        Vec3 dp;
        for (int k = 0; k < c.count; ++k)
            dp += (s.x[c.idx[k]] - s.xPrev[c.idx[k]]) * c.weight[k];
        const Vec3 tangential = dp - c.normal * dot(dp, c.normal);
        const Real slide = norm(tangential);
        if (slide < Real(1e-15)) continue;

        const Real budget = c.friction * c.lambdaN * wEff - c.appliedTangential;
        if (budget <= Real(0)) continue;
        const Real capped = std::min(slide, budget);
        c.appliedTangential += capped;

        const Vec3 correction = tangential * (-capped / slide);
        for (int k = 0; k < c.count; ++k)
            s.x[c.idx[k]] += correction * (s.invMass[c.idx[k]] * c.weight[k] / wEff);
    }
}

void substep(Rod& rod, const SolverParams& p, Real h, ContactSet* contacts) {
    RodState& s = rod.state;
    const std::size_t nP = s.numParticles(), nS = s.numSegments();

    // --- predict ---
    for (std::size_t i = 0; i < nP; ++i) {
        s.xPrev[i] = s.x[i];
        if (s.invMass[i] == Real(0)) continue;
        s.v[i] += (p.gravity + s.extForce[i] * s.invMass[i]) * h;
        s.x[i] += s.v[i] * h;
    }
    for (std::size_t j = 0; j < nS; ++j) {
        s.qPrev[j] = s.q[j];
        if (norm2(s.invInertia[j]) == Real(0)) continue;
        // Free-body precession plus applied torque: I w' = tau - w x (I w),
        // all in the body frame (the applied torque is stored in world).
        const Vec3 w = s.omega[j];
        const Vec3 tau = rotateInv(s.q[j], s.extTorque[j]);
        s.omega[j] = w + cwise(s.invInertia[j], tau - cross(w, cwise(s.inertia[j], w))) * h;
        s.q[j] = normalize(s.q[j] + (s.q[j] * Quat(Real(0), s.omega[j])) * (Real(0.5) * h));
    }

    // --- solve ---
    for (int it = 0; it < p.iterations; ++it) {
        projectConstraints(rod, h, p.stretchColoring, p.bendColoring);
        if (contacts) projectContacts(rod, *contacts);
    }

    // --- recover velocities ---
    const Real linDecay = std::exp(-p.linearDamping * h);
    const Real angDecay = std::exp(-p.angularDamping * h);
    for (std::size_t i = 0; i < nP; ++i) {
        if (s.invMass[i] == Real(0)) {
            s.v[i] = Vec3();
            continue;
        }
        s.v[i] = (s.x[i] - s.xPrev[i]) / h * linDecay;
    }
    for (std::size_t j = 0; j < nS; ++j) {
        if (norm2(s.invInertia[j]) == Real(0)) {
            s.omega[j] = Vec3();
            continue;
        }
        const Quat dq = sameHemisphere(conj(s.qPrev[j]) * s.q[j], Quat());
        s.omega[j] = dq.im() * (Real(2) / h) * angDecay;
    }
}

void step(Rod& rod, const SolverParams& p) {
    const Real h = p.dt / p.substeps;
    for (int i = 0; i < p.substeps; ++i) {
        // XPBD multipliers are per-substep: each substep is its own implicit
        // solve, so a stale lambda from the previous one would double-count.
        rod.stretch.clearMultipliers();
        rod.bend.clearMultipliers();
        substep(rod, p, h, nullptr);
    }
}

void step(Rod& rod, const SolverParams& p, const CollisionWorld& world, SolverContext& ctx) {
    const Real h = p.dt / p.substeps;
    const int interval = std::max(1, p.contactInterval);
    for (int i = 0; i < p.substeps; ++i) {
        if (i % interval == 0) generateContacts(rod, world, ctx.contacts, ctx.hash);
        rod.stretch.clearMultipliers();
        rod.bend.clearMultipliers();
        ctx.contacts.resetMultipliers();
        substep(rod, p, h, &ctx.contacts);
    }
}

Residual residual(const Rod& rod) {
    Residual r;
    for (const Vec3& v : rod.state.v) r.maxSpeed = std::max(r.maxSpeed, norm(v));
    for (const Vec3& w : rod.state.omega) r.maxAngularSpeed = std::max(r.maxAngularSpeed, norm(w));
    return r;
}

RelaxReport relaxToEquilibrium(Rod& rod, SolverParams params, int maxSteps, Real speedTol) {
    // Damping only sets the path to equilibrium, not the equilibrium itself,
    // but it sets it strongly: too little and the rod rings for a long time,
    // too much and it creeps. Callers that know their lowest mode should pass
    // roughly 2*omega_1; this is a fallback, not a good default.
    if (params.linearDamping == Real(0)) params.linearDamping = Real(5);
    if (params.angularDamping == Real(0)) params.angularDamping = Real(5);

    RelaxReport report;
    for (int i = 0; i < maxSteps; ++i) {
        step(rod, params);
        report.steps = i + 1;
        const Residual r = residual(rod);
        if (r.maxSpeed < speedTol && r.maxAngularSpeed < speedTol) {
            report.converged = true;
            report.final = r;
            return report;
        }
    }
    report.final = residual(rod);
    return report;
}

RelaxReport relaxToEquilibrium(Rod& rod, SolverParams params, const CollisionWorld& world,
                               SolverContext& ctx, int maxSteps, Real speedTol) {
    if (params.linearDamping == Real(0)) params.linearDamping = Real(5);
    if (params.angularDamping == Real(0)) params.angularDamping = Real(5);

    RelaxReport report;
    for (int i = 0; i < maxSteps; ++i) {
        step(rod, params, world, ctx);
        report.steps = i + 1;
        const Residual r = residual(rod);
        if (r.maxSpeed < speedTol && r.maxAngularSpeed < speedTol) {
            report.converged = true;
            report.final = r;
            return report;
        }
    }
    report.final = residual(rod);
    return report;
}

}  // namespace crs
