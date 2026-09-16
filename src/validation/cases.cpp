#include "cases.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>

#include "../core/solver.h"
#include "../core/statics.h"
#include "support.h"

namespace crs {

// ---------------------------------------------------------------- helpers

bool CaseResult::pass() const {
    for (const Check& c : checks)
        if (!c.pass) return false;
    return true;
}

Check makeCheck(const std::string& name, double value, double reference, double tol, double scale) {
    Check c;
    c.name = name;
    c.value = value;
    c.reference = reference;
    c.tol = tol;
    const double denom = scale > 0 ? scale : std::abs(reference);
    c.relError = denom > 0 ? std::abs(value - reference) / denom : std::abs(value - reference);
    c.pass = c.relError <= tol;
    return c;
}

Check makeRangeCheck(const std::string& name, double value, double lo, double hi) {
    Check c;
    c.name = name;
    c.value = value;
    c.reference = 0.5 * (lo + hi);
    c.tol = 0.5 * (hi - lo);
    c.relError = std::abs(value - c.reference);
    c.pass = value >= lo && value <= hi;
    return c;
}

namespace {

// Least-squares slope of log10(y) against log10(x). Used to report the observed
// convergence order; reported, never asserted away.
double logLogSlope(const std::vector<double>& xs, const std::vector<double>& ys) {
    const std::size_t n = xs.size();
    if (n < 2) return 0;
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const double lx = std::log10(xs[i]), ly = std::log10(ys[i]);
        sx += lx;
        sy += ly;
        sxx += lx * lx;
        sxy += lx * ly;
    }
    const double d = n * sxx - sx * sx;
    return d == 0 ? 0 : (n * sxy - sx * sy) / d;
}

}  // namespace

// ------------------------------------------------------- solver convergence
//
// Before any mesh-convergence claim means anything, the solver itself has to be
// converged: the static equilibrium of XPBD is independent of the timestep only
// if the implicit system is actually solved each substep. Gauss-Seidel on a rod
// is a chain, so information travels one element per sweep -- a 128-element rod
// needs a sweep budget of that order before the tip load is felt at the root.
//
// This case sweeps the (substeps x iterations) budget at fixed mesh resolution
// and checks the richest budget against the direct static solve of the same
// discrete rod (statics.h). The mesh-convergence cases use that direct solve,
// so this is the link that says the XPBD solver actually reaches the
// equilibrium they validate. Self-consistency alone is not enough: an
// under-swept relaxation settles, stops moving, and is wrong.
CaseResult runSolverConvergence(const std::string& outDir) {
    CaseResult res;
    res.name = "solver parameter convergence (static tip deflection)";

    const RodMaterial mat = referenceMaterial();
    const Real L = Real(1);
    const Real EI = mat.bendStiffness();
    const Real P = Real(3) * EI * Real(0.005) / (L * L);
    const int n = 32;

    Csv csv(outDir, "solver_convergence.csv",
            "segments,substeps,iterations,budget,tip_deflection,direct_tip,rel_gap,steps,converged");

    Rod direct = makeStraightRod(n, L, mat, Vec3(0, 0, 0), Vec3(1, 0, 0));
    clampRootExact(direct);
    direct.state.extForce.back() = Vec3(0, 0, -P);
    const StaticReport directRep = solveStatic(direct);
    const double directTip = -direct.state.x.back().z;

    const std::vector<int> substepList = {1, 4, 16};
    const std::vector<int> iterList = {1, 4, 16, 64};

    double finest = 0, prev = 0;
    for (int sub : substepList) {
        for (int iter : iterList) {
            Rod rod = makeStraightRod(n, L, mat, Vec3(0, 0, 0), Vec3(1, 0, 0));
            clampRootExact(rod);
            rod.state.extForce.back() = Vec3(0, 0, -P);

            SolverParams p = staticParams(rod, 2 * cantileverOmega1(mat, L));
            p.substeps = sub;
            p.iterations = iter;
            const RelaxReport rep = relaxToEquilibrium(rod, p, 4000, Real(1e-10));
            const double tip = -rod.state.x.back().z;
            csv.row(n, sub, iter, sub * iter, tip, directTip, std::abs(tip - directTip) / directTip,
                    rep.steps, rep.converged ? 1 : 0);
            prev = finest;
            finest = tip;
        }
    }

    res.notes.push_back(fmt("  n=%.0f, richest budget tip = %.9g m (previous budget %.9g m)",
                            double(n), finest, prev));
    res.notes.push_back(fmt("  direct static solve tip = %.9g m", directTip));
    if (!directRep.converged) res.notes.push_back("  direct static solve did not converge");
    res.checks.push_back(makeCheck("tip deflection settled across budget", finest, prev, 0.01));
    res.checks.push_back(makeCheck("XPBD relaxation vs direct static solve", finest, directTip, 1e-6));
    return res;
}

// ---------------------------------------------------------------- cantilever
//
// Small-deflection cantilever with a transverse tip load, gravity off.
//   Euler-Bernoulli:  delta = P L^3 / (3 EI)
//   Timoshenko:       delta = P L^3 / (3 EI) + P L / (kappa G A)
// The Cosserat rod is shearable, so Timoshenko is what the discrete model
// actually converges to -- at second order, to 7.6e-6 at n = 256 -- and
// Euler-Bernoulli is the headline number.
CaseResult runCantilever(const std::string& outDir) {
    CaseResult res;
    res.name = "cantilever (Euler-Bernoulli convergence)";

    const RodMaterial mat = referenceMaterial();
    const Real L = Real(1);
    const Real EI = mat.bendStiffness();
    // Tip deflection / L. Small enough that the geometric nonlinearity of the
    // discrete model (relative size ~ ratio^2) stays below the discretization
    // error at the finest mesh; at 0.005 it floored the convergence by n = 64.
    const Real targetRatio = Real(1e-4);
    const Real P = Real(3) * EI * targetRatio * L / (L * L * L);

    const double deltaEB = P * L * L * L / (3 * EI);
    const double deltaTim = deltaEB + P * L / (mat.shearCorrection() * mat.shear() * mat.area());

    // Solved directly (statics.h), so refinement costs O(n), not the O(n^2)
    // sweep budget a converged XPBD relaxation needs.
    const std::vector<int> counts = {4, 8, 16, 32, 64, 128, 256};
    std::vector<double> hs, errs;
    double finestEB = 0;

    Csv csv(outDir, "cantilever_convergence.csv",
            "segments,h,tip_deflection,euler_bernoulli,timoshenko,rel_err_eb,rel_err_tim,"
            "newton_iterations,converged");

    for (int n : counts) {
        Rod rod = makeStraightRod(n, L, mat, Vec3(0, 0, 0), Vec3(1, 0, 0));
        clampRootExact(rod);
        rod.state.extForce.back() = Vec3(0, 0, -P);

        const StaticReport rep = solveStatic(rod);

        const double tip = -rod.state.x.back().z;
        const double errEB = std::abs(tip - deltaEB) / deltaEB;
        const double errTim = std::abs(tip - deltaTim) / deltaTim;
        hs.push_back(L / n);
        errs.push_back(errTim);
        finestEB = errEB;

        csv.row(n, L / n, tip, deltaEB, deltaTim, errEB, errTim, rep.newtonIterations,
                rep.converged ? 1 : 0);
        if (!rep.converged) res.notes.push_back(fmt("  n=%.0f: static solve did not converge", n));
    }

    const double slope = logLogSlope(hs, errs);
    res.notes.push_back(fmt("  error vs Timoshenko: slope %.2f in h (log-log fit over all n)", slope));
    res.notes.push_back(fmt("  P = %.6g N, delta_EB = %.6g m, delta_Timoshenko = %.6g m", P, deltaEB,
                            deltaTim));

    // Frame invariance: the same rod and load, turned to an oblique axis, must
    // deflect by the same amount. This is the check that would have caught the
    // stretch constraint once measuring its strain in world components (see
    // solver.cpp), which only an axis-aligned rod along z got right.
    {
        const int n = 32;
        const Vec3 axis = normalize(Vec3(1, 2, 3));
        const Vec3 loadDir = normalize(cross(axis, Vec3(1, 0, 0)));
        auto tipAlong = [&](Vec3 dir, Vec3 load) {
            Rod rod = makeStraightRod(n, L, mat, Vec3(0, 0, 0), dir);
            clampRootExact(rod);
            rod.state.extForce.back() = load * P;
            solveStatic(rod);
            return dot(rod.state.x.back(), load);
        };
        const double aligned = tipAlong(Vec3(1, 0, 0), Vec3(0, 0, -1));
        const double oblique = tipAlong(axis, loadDir);
        res.checks.push_back(makeCheck("same deflection on an oblique axis", oblique, aligned, 1e-9));
    }

    res.checks.push_back(
        makeCheck("tip deflection vs Euler-Bernoulli (finest mesh)", finestEB, 0.0, 0.01, 1.0));
    res.checks.push_back(makeCheck("tip deflection vs Timoshenko (finest mesh)", errs.back(), 0.0,
                                   2e-5, 1.0));
    res.checks.push_back(makeRangeCheck("convergence order in h", slope, 1.8, 2.2));
    return res;
}

// ---------------------------------------------------------------- pure moment
//
// A clamped rod loaded by a pure moment M at the free end takes a circular arc
// of radius EI/M, at any deflection magnitude. This is the cleanest fully
// nonlinear bending test there is: no small-angle assumption anywhere.
CaseResult runElasticaMoment(const std::string& outDir) {
    CaseResult res;
    res.name = "elastica: pure end moment -> circular arc";

    const RodMaterial mat = referenceMaterial();
    const Real L = Real(1);
    const Real EI = mat.bendStiffness();
    const Real kappa = Real(kPi) / L;  // bend the rod into a half circle
    const Real M = EI * kappa;
    const double Rref = 1.0 / kappa;

    // Solved directly with load continuation (statics.h). The XPBD relaxation
    // this used to take capped the sweep at n = 32, because a full end moment
    // on one small segment forces a very small predictor step.
    const std::vector<int> counts = {8, 16, 32, 64, 128};
    std::vector<double> hs, errs;
    double finestErr = 0;

    Csv csv(outDir, "elastica_moment.csv",
            "segments,h,radius,radius_ref,rel_err,load_steps,newton_iterations,converged");

    for (int n : counts) {
        Rod rod = makeStraightRod(n, L, mat, Vec3(0, 0, 0), Vec3(1, 0, 0));
        // The ghost frame is appended to the segment arrays, so the last REAL
        // segment has to be named before the clamp is applied.
        const int lastSeg = static_cast<int>(rod.state.numSegments()) - 1;
        clampRootExact(rod);
        rod.state.extTorque[lastSeg] = Vec3(0, -M, 0);  // bends the rod up, in the x-z plane

        const StaticReport rep = solveStatic(rod);
        if (!rep.converged) res.notes.push_back(fmt("  n=%.0f: static solve did not converge", n));

        // Arc radius from the total turning of the material frame: the rod is a
        // regular polygon inscribed in the arc, so measuring the turn per unit
        // arclength avoids any circle-fitting bias.
        const Vec3 t0 = rotate(rod.state.q.front(), Vec3(0, 0, 1));
        const Vec3 t1 = rotate(rod.state.q[lastSeg], Vec3(0, 0, 1));
        const double turn = std::atan2(norm(cross(t0, t1)), dot(t0, t1));
        const double arc = L * (n - 1.0) / n;  // midpoint of first to midpoint of last segment
        const double radius = arc / turn;

        const double err = std::abs(radius - Rref) / Rref;
        hs.push_back(L / n);
        errs.push_back(err);
        finestErr = err;
        csv.row(n, L / n, radius, Rref, err, rep.loadSteps, rep.newtonIterations,
                rep.converged ? 1 : 0);
    }

    const double slope = logLogSlope(hs, errs);
    res.notes.push_back(fmt("  arc radius error: slope %.2f in h; M = %.6g N m, R_ref = %.6g m",
                            slope, M, Rref));
    res.checks.push_back(makeCheck("arc radius (finest mesh)", finestErr, 0.0, 0.005, 1.0));
    res.checks.push_back(makeRangeCheck("convergence order in h", slope, 0.9, 2.5));
    return res;
}

// ---------------------------------------------------------------- tip-load elastica
namespace {

// Reference solution of the inextensible planar elastica under a transverse tip
// load, by shooting on the exact BVP rather than elliptic integrals:
//
//   EI theta'' = -P cos(theta),   theta(0) = 0,   theta'(L) = 0
//
// theta is measured from the undeformed axis toward the load. Integrating the
// tangent afterwards gives the tip position.
struct ElasticaRef {
    double tipX = 0, tipZ = 0, tipAngle = 0;
};

ElasticaRef solveElasticaRef(double P, double L, double EI, int steps = 20000) {
    const double a = P / EI;
    auto shoot = [&](double kappa0, ElasticaRef* out) {
        const double ds = L / steps;
        double th = 0, dth = kappa0, x = 0, z = 0;
        for (int i = 0; i < steps; ++i) {
            // RK4 on (theta, theta') with theta'' = -a cos(theta).
            const double k1t = dth, k1d = -a * std::cos(th);
            const double k2t = dth + 0.5 * ds * k1d, k2d = -a * std::cos(th + 0.5 * ds * k1t);
            const double k3t = dth + 0.5 * ds * k2d, k3d = -a * std::cos(th + 0.5 * ds * k2t);
            const double k4t = dth + ds * k3d, k4d = -a * std::cos(th + ds * k3t);
            // Trapezoid the tangent over the same interval for x and z.
            const double thNext = th + ds * (k1t + 2 * k2t + 2 * k3t + k4t) / 6.0;
            x += 0.5 * ds * (std::cos(th) + std::cos(thNext));
            z -= 0.5 * ds * (std::sin(th) + std::sin(thNext));
            dth += ds * (k1d + 2 * k2d + 2 * k3d + k4d) / 6.0;
            th = thNext;
        }
        if (out) {
            out->tipX = x;
            out->tipZ = z;
            out->tipAngle = th;
        }
        return dth;  // theta'(L), zero at the solution
    };

    // theta'(L) grows monotonically with the shooting parameter, so bisect.
    double lo = 0, hi = 10 * a * L + 10;
    for (int i = 0; i < 64; ++i) {  // halves a bracket of O(10) past double precision
        const double mid = 0.5 * (lo + hi);
        if (shoot(mid, nullptr) > 0)
            hi = mid;
        else
            lo = mid;
    }
    ElasticaRef ref;
    shoot(0.5 * (lo + hi), &ref);
    return ref;
}

}  // namespace

CaseResult runElasticaTipLoad(const std::string& outDir) {
    CaseResult res;
    res.name = "elastica: large-deflection tip load";

    const RodMaterial mat = referenceMaterial();
    const Real L = Real(1);
    const Real EI = mat.bendStiffness();

    // alpha = P L^2 / EI is the only parameter of the inextensible problem.
    const std::vector<double> alphas = {0.5, 1.0, 2.0, 3.0, 5.0};
    const int n = 32;

    Csv csv(outDir, "elastica_tipload.csv",
            "alpha,load,tip_x,tip_z,ref_x,ref_z,err_x,err_z,err_dist,newton_iterations,converged");

    double worst = 0;
    for (double alpha : alphas) {
        const Real P = Real(alpha) * EI / (L * L);
        Rod rod = makeStraightRod(n, L, mat, Vec3(0, 0, 0), Vec3(1, 0, 0));
        clampRootExact(rod);
        rod.state.extForce.back() = Vec3(0, 0, -P);

        const StaticReport rep = solveStatic(rod);
        if (!rep.converged)
            res.notes.push_back(fmt("  alpha=%.1f: static solve did not converge", alpha));

        const ElasticaRef ref = solveElasticaRef(P, L, EI);
        const double tipX = rod.state.x.back().x, tipZ = rod.state.x.back().z;
        const double errX = std::abs(tipX - ref.tipX) / L;
        const double errZ = std::abs(tipZ - ref.tipZ) / L;
        const double dist = std::sqrt((tipX - ref.tipX) * (tipX - ref.tipX) +
                                      (tipZ - ref.tipZ) * (tipZ - ref.tipZ)) / L;
        worst = std::max(worst, dist);
        csv.row(alpha, P, tipX, tipZ, ref.tipX, ref.tipZ, errX, errZ, dist, rep.newtonIterations,
                rep.converged ? 1 : 0);
    }

    res.notes.push_back(fmt("  n=%.0f, worst tip position error %.3g L over alpha in [0.5, 5]",
                            double(n), worst));
    res.checks.push_back(makeCheck("tip position vs exact elastica", worst, 0.0, 0.005, 1.0));
    return res;
}

// ---------------------------------------------------------------- helix
//
// A rod given a uniform intrinsic Darboux vector Omega = (0, kappa, tau) has a
// helix as its zero-energy state, with
//     R = kappa / (kappa^2 + tau^2),   pitch = 2 pi tau / (kappa^2 + tau^2).
// Starting from a straight rod and relaxing must find that helix, and the
// geometry is measured from the centerline alone -- never from the constraint
// that produced it.
CaseResult runHelix(const std::string& outDir) {
    CaseResult res;
    res.name = "helical equilibrium (intrinsic curvature + twist)";

    const RodMaterial mat = referenceMaterial();
    const Real L = Real(1);
    const int n = 100;
    const Real kappa = Real(10), tau = Real(5);
    const double Rref = kappa / (kappa * kappa + tau * tau);
    const double pitchRef = 2 * kPi * tau / (kappa * kappa + tau * tau);

    Rod rod = makeStraightRod(n, L, mat, Vec3(0, 0, 0), Vec3(0, 0, 1));
    for (std::size_t k = 0; k < rod.bend.size(); ++k) rod.bend.restDarboux[k] = Vec3(0, kappa, tau);
    const int nSegRef = static_cast<int>(rod.state.numSegments());

    // No external load here: the target is the rod's own stress-free shape, so
    // there is no end load to propagate along the chain and the quadratic sweep
    // budget the loaded cases need would be pure waste. A fixed modest budget
    // converges this case.
    SolverParams p;
    p.dt = Real(0.005);
    p.substeps = 16;
    p.iterations = 32;
    p.gravity = Vec3();
    p.linearDamping = p.angularDamping = Real(20);
    const RelaxReport rep = relaxToEquilibrium(rod, p, 60000, Real(1e-9));

    // Axis: a rod with a constant Darboux vector is a rigid screw, so the
    // material frame rotates about the helix axis and nothing else. The axis of
    // the world-frame rotation carrying the first frame onto the last is
    // therefore the helix axis exactly.
    //
    // (Averaging the tangents also "works", but only over a whole number of
    // turns: over 1.78 turns it tilts the axis enough to bias a circle fit by
    // 8%, which is how this measurement first went wrong.)
    const Quat rel = sameHemisphere(rod.state.q[nSegRef - 1] * conj(rod.state.q[0]), Quat());
    Vec3 axis = normalize(rel.im());

    // Two in-plane basis vectors perpendicular to the axis.
    Vec3 e1 = std::abs(axis.x) < Real(0.9) ? Vec3(1, 0, 0) : Vec3(0, 1, 0);
    e1 = normalize(e1 - axis * dot(e1, axis));
    const Vec3 e2 = cross(axis, e1);

    // Algebraic (Kasa) circle fit of the projected centerline.
    double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0, sxz = 0, syz = 0, sz = 0;
    std::vector<double> us, vs, ws;
    const std::size_t nP = rod.state.numParticles();
    for (std::size_t i = 0; i < nP; ++i) {
        const Vec3 x = rod.state.x[i];
        const double u = dot(x, e1), v = dot(x, e2);
        us.push_back(u);
        vs.push_back(v);
        ws.push_back(dot(x, axis));
        const double zz = u * u + v * v;
        sx += u; sy += v; sxx += u * u; syy += v * v; sxy += u * v;
        sxz += u * zz; syz += v * zz; sz += zz;
    }
    const double N = double(nP);
    const double a11 = 2 * (sxx - sx * sx / N), a12 = 2 * (sxy - sx * sy / N);
    const double a22 = 2 * (syy - sy * sy / N);
    const double b1 = sxz - sx * sz / N, b2 = syz - sy * sz / N;
    const double det = a11 * a22 - a12 * a12;
    const double cu = (b1 * a22 - b2 * a12) / det, cv = (a11 * b2 - a12 * b1) / det;
    double radius = 0;
    for (std::size_t i = 0; i < nP; ++i)
        radius += std::sqrt((us[i] - cu) * (us[i] - cu) + (vs[i] - cv) * (vs[i] - cv));
    radius /= N;

    // Pitch: unwrap the angle about the axis and regress axial position on it.
    double prev = 0, unwrapped = 0;
    std::vector<double> phis;
    for (std::size_t i = 0; i < nP; ++i) {
        const double ang = std::atan2(vs[i] - cv, us[i] - cu);
        if (i > 0) {
            double d = ang - prev;
            while (d > kPi) d -= 2 * kPi;
            while (d < -kPi) d += 2 * kPi;
            unwrapped += d;
        }
        prev = ang;
        phis.push_back(unwrapped);
    }
    double pp = 0, pw = 0, ppp = 0, ppw = 0;
    for (std::size_t i = 0; i < nP; ++i) {
        pp += phis[i]; pw += ws[i]; ppp += phis[i] * phis[i]; ppw += phis[i] * ws[i];
    }
    const double slope = (N * ppw - pp * pw) / (N * ppp - pp * pp);
    const double pitch = std::abs(slope) * 2 * kPi;

    Csv csv(outDir, "helix.csv", "segments,radius,radius_ref,pitch,pitch_ref,turns,steps,converged");
    csv.row(n, radius, Rref, pitch, pitchRef, std::abs(phis.back()) / (2 * kPi), rep.steps,
            rep.converged ? 1 : 0);

    // The target helix is the rod's zero-energy state, so residual elastic
    // energy is the direct measure of whether the relaxation actually found it.
    // Scaled against the energy the same rod holds when forced straight, which
    // is the natural unit for "how far from its rest shape is this rod".
    Rod straight = makeStraightRod(n, L, mat, Vec3(0, 0, 0), Vec3(0, 0, 1));
    for (std::size_t k = 0; k < straight.bend.size(); ++k)
        straight.bend.restDarboux[k] = Vec3(0, kappa, tau);
    const double eScale = straight.elasticEnergy();
    const double eResidual = rod.elasticEnergy() / eScale;

    res.notes.push_back(fmt("  kappa = %.4g 1/m, tau = %.4g 1/m, %.4f turns over L = 1 m",
                            double(kappa), double(tau), std::abs(phis.back()) / (2 * kPi)));
    res.notes.push_back(fmt("  turns expected %.4f; residual elastic energy %.3g of straight-rod",
                            L * std::sqrt(double(kappa * kappa + tau * tau)) / (2 * kPi),
                            eResidual));
    res.checks.push_back(makeCheck("residual elastic energy at rest shape", eResidual, 0.0, 0.01,
                                   1.0));
    if (!rep.converged) res.notes.push_back("  relaxation did not reach the residual tolerance");
    res.checks.push_back(makeCheck("helix radius", radius, Rref, 0.02));
    res.checks.push_back(makeCheck("helix pitch", pitch, pitchRef, 0.02));
    return res;
}

// ------------------------------------------------------------- twist buckling
//
// Michell / Greenhill instability: a naturally straight rod carrying a twisting
// moment M buckles out of its straight state above a critical M. Linearizing
// the Kirchhoff equations about the straight twisted state gives, for the
// complex lateral deflection w = u + i v,
//
//     EI w_ssss - i M w_sss = 0
//
// and with clamped ends (w = w_s = 0 at both) the solvability condition reduces
// to  theta = 2 atan(theta / 2) + 2 pi, i.e.
//
//     M_crit L / EI = 8.986819...
//
// (The textbook 2 pi is the pinned-pinned case; clamping raises it.) Imposing
// the twist kinematically rather than by applied moment, the rod carries
// M = GJ Phi / L before it buckles, so the critical end rotation is
//
//     Phi_crit = 8.986819 EI / GJ = 8.986819 (1 + nu)
//
// Fixing both end positions makes the bifurcation supercritical: above the
// threshold the amplitude is limited by the axial strain that bowing costs, so
// it grows smoothly like sqrt(Phi - Phi_crit) instead of running away. That is
// what makes the threshold measurable rather than merely bracketable: a^2 is
// linear in Phi near onset, so extrapolating it to zero locates Phi_crit
// without ever having to pick an arbitrary "has buckled" amplitude.
CaseResult runTwistBuckling(const std::string& outDir) {
    CaseResult res;
    res.name = "Michell/Greenhill twist buckling threshold";

    const RodMaterial mat = referenceMaterial();
    const Real L = Real(1);
    const double thetaCrit = 8.986819;  // clamped-clamped
    const double phiCritRef = thetaCrit * mat.bendStiffness() / mat.twistStiffness();

    // Clamped-clamped fundamental, for the damping rate. Using the cantilever
    // value here (6x lower) leaves the rod ringing for the whole run.

    const Real seed = Real(1e-7);

    // Buckling is a statement about dynamics, so this case runs dynamics.
    //
    // The first attempt used the static relaxation every other case uses, and it
    // was doubly wrong. With light damping it reported a threshold 18% low, and
    // that "buckling" turned out to be numerical: raising the damping made it
    // disappear entirely, which no physical instability does. And a well-damped,
    // heavily-iterated XPBD step is very nearly a backward-Euler solve, which is
    // unconditionally stable -- it damps genuinely unstable modes right along
    // with everything else. A quasi-static solver is the wrong instrument for
    // finding an instability no matter how carefully it is tuned.
    //
    // So: small substeps, one sweep, no damping. The substep count is set by
    // the requirement that alpha/h^2 dominate J M^-1 J^T, which for this rod
    // means h of a few microseconds; that is affordable precisely because
    // honest dynamics does not need the quadratic sweep budget a converged
    // static solve does.
    const int steps = 4000;
    const int substeps = 256;

    auto twistedRod = [&](int n, double phi) {
        Rod rod = makeStraightRod(n, L, mat, Vec3(0, 0, 0), Vec3(0, 0, 1));
        clampEndExact(rod, false);  // root ghost stays at angle 0, as s = 0 should
        const int tipGhost = clampEndExact(rod, true);

        // Build the straight, uniformly twisted base state directly.
        //
        // The discrete twist lives in Im(conj(q_a) q_b), which only represents
        // relative rotations inside (-pi, pi). Rotating the end frame to
        // Phi = 12 rad in one go asks a single half-length element to hold 1.9
        // turns; the constraint wraps it modulo 2 pi and the rod ends up storing
        // no twist at all. Spread over the rod it is half a radian per joint.
        const Real l = rod.stretch.restLength.front();
        const int nSeg = static_cast<int>(rod.stretch.size());
        for (int j = 0; j < nSeg; ++j) {
            const Real sMid = (Real(j) + Real(0.5)) * l;
            rod.state.q[j] =
                normalize(quatAxisAngle(Vec3(0, 0, 1), Real(phi) * sMid / L) * rod.state.q[j]);
        }
        rod.state.q[tipGhost] =
            normalize(quatAxisAngle(Vec3(0, 0, 1), Real(phi)) * rod.state.q[tipGhost]);
        rod.state.qPrev = rod.state.q;
        return rod;
    };

    auto amplitudeAt = [&](int n, double phi) {
        Rod rod = twistedRod(n, phi);
        for (std::size_t i = 1; i + 1 < rod.state.numParticles(); ++i)
            rod.state.x[i].x += seed * std::sin(kPi * rod.state.x[i].z / L);
        rod.state.xPrev = rod.state.x;

        SolverParams p;
        p.dt = Real(1e-3);
        p.substeps = substeps;
        p.iterations = 1;
        p.gravity = Vec3();

        double amp = 0;
        for (int i = 0; i < steps; ++i) {
            step(rod, p);
            // Track the largest excursion over the whole window, not just the
            // final one: below threshold the seed oscillates rather than decays,
            // so the endpoint alone is a noisy thing to threshold on.
            for (const Vec3& x : rod.state.x)
                amp = std::max(amp, std::sqrt(x.x * x.x + x.y * x.y));
            if (amp > 0.05) break;  // unambiguously buckled; no need to continue
        }
        return amp;
    };

    Csv csv(outDir, "twist_buckling.csv",
            "segments,h,phi_crit,phi_crit_ref,rel_err,amp_below,amp_above");

    // Guard on the base state before trusting any threshold read off it. A
    // uniformly twisted rod's stored energy has a closed form in the DISCRETE
    // model, and that is what to compare against: the discrete twist measure is
    // 2 sin(phi/2) / lbar, not phi / lbar, so its energy sits below the
    // continuum value by a factor (sin(x)/x)^2 with x = phi/2 -- about 2% at
    // half a radian per joint, vanishing as O(h^2). Checking against the
    // continuum number instead would flag that convergent difference as a bug.
    {
        const int n = 24;
        const double phi = phiCritRef;
        Rod rod = twistedRod(n, phi);
        const double stored = rod.elasticEnergy();

        const double GJ = double(mat.twistStiffness());
        const double l = double(L) / n;
        auto jointEnergy = [&](double lbar) {
            const double jointPhi = phi * lbar / double(L);
            const double omega = 2 * std::sin(0.5 * jointPhi) / lbar;
            return 0.5 * GJ * lbar * omega * omega;
        };
        // n-1 interior joints at full element length, plus a half-length
        // boundary element at each clamped end.
        const double discrete = (n - 1) * jointEnergy(l) + 2 * jointEnergy(0.5 * l);
        const double continuum = 0.5 * GJ * phi * phi / double(L);

        res.notes.push_back(fmt("  twist energy %.6g J vs discrete prediction %.6g J (%.3g rel)",
                                stored, discrete, std::abs(stored - discrete) / discrete));
        res.notes.push_back(fmt("  discrete sits %.2f%% under the continuum GJ Phi^2 / 2L = %.6g J, "
                                "an O(h^2) effect of the sin(phi/2) twist measure",
                                100 * (continuum - discrete) / continuum, continuum));
        res.checks.push_back(
            makeCheck("twisted base state stores the right energy", stored, discrete, 0.002));
    }

    const double buckled = 100.0 * double(seed);  // four orders below the saturated amplitude
    const std::vector<int> counts = {12, 16, 24, 32};
    std::vector<double> hs, errs;
    double finest = 0;

    for (int n : counts) {
        // Bracket, then bisect. The lower bracket is known not to buckle and the
        // upper is known to.
        double lo = 0.5 * phiCritRef, hi = 1.6 * phiCritRef;
        const double ampLo = amplitudeAt(n, lo), ampHi = amplitudeAt(n, hi);
        for (int it = 0; it < 9; ++it) {
            const double mid = 0.5 * (lo + hi);
            if (amplitudeAt(n, mid) > buckled)
                hi = mid;
            else
                lo = mid;
        }
        const double phiCrit = 0.5 * (lo + hi);
        const double err = std::abs(phiCrit - phiCritRef) / phiCritRef;
        hs.push_back(L / n);
        errs.push_back(err);
        finest = err;
        csv.row(n, L / n, phiCrit, phiCritRef, err, ampLo, ampHi);
    }

    const double slope = logLogSlope(hs, errs);
    res.notes.push_back(fmt("  clamped-clamped: M_crit L / EI = %.6f, so Phi_crit = %.5f rad "
                            "(%.4f turns)", thetaCrit, phiCritRef, phiCritRef / (2 * kPi)));
    res.notes.push_back(fmt("  threshold error %.3g at the finest mesh, converging at slope %.2f "
                            "in h", finest, slope));
    res.checks.push_back(makeCheck("buckling threshold (finest mesh)", finest, 0.0, 0.05, 1.0));
    res.checks.push_back(makeRangeCheck("threshold convergence order in h", slope, 0.8, 3.0));
    return res;
}

// ---------------------------------------------------------------- energy drift
//
// Free flight, no gravity, no damping, 1e5 steps. XPBD is not symplectic, and
// what it actually does here is worth stating precisely rather than hiding
// behind the word "drift": it DISSIPATES the elastic oscillation over the first
// few seconds and then conserves what is left essentially exactly, because what
// is left is rigid-body motion with nothing for the constraint solver to damp.
//
// Energy conservation is therefore not a property this integrator has, and
// asserting it would only produce a tolerance chosen to make the number pass.
// What IS assertable, and what this case checks:
//
//   * dissipation must FALL with the substep count. It is a discretization
//     error, so refining the substep has to reduce it -- that is the statement
//     that the dissipation is numerical rather than a modelling mistake.
//   * the solver must never ADD energy. A dissipative integrator is usable; one
//     that injects energy is not, and that is what blows rods up in practice.
//   * linear and angular momentum must be conserved to round-off, because every
//     constraint force here is internal. These are the invariants XPBD really
//     does hold, and they are the ones worth a tight tolerance.
CaseResult runEnergyDrift(const std::string& outDir) {
    CaseResult res;
    res.name = "energy / momentum drift (1e5 steps, free flight)";

    const RodMaterial mat = referenceMaterial();
    const Real L = Real(1);
    const int n = 32;
    const int steps = 100000;
    const Real dt = Real(1e-3);
    const int transientSteps = 20000;  // 20 s: well past the elastic transient

    Csv summary(outDir, "energy_dissipation.csv",
                "substeps,energy0,dissipation,late_drift,max_energy_gain,"
                "linear_momentum_err,angular_momentum_err");
    Csv series(outDir, "energy_drift.csv",
               "step,time,kinetic,elastic,total,rel_energy_drift,linear_momentum_err,"
               "angular_momentum_err");

    const std::vector<int> substepList = {2, 4, 8, 16};
    std::vector<double> dissipation;
    std::vector<double> gains;
    double worstLate = 0, worstP = 0, worstL = 0;

    for (std::size_t si = 0; si < substepList.size(); ++si) {
        const int sub = substepList[si];
        const bool finest = (si + 1 == substepList.size());

        Rod rod = makeStraightRod(n, L, mat, Vec3(0, 0, 0), Vec3(1, 0, 0));
        // Excite the first bending mode with a zero-mean transverse velocity.
        for (std::size_t i = 0; i < rod.state.numParticles(); ++i) {
            const Real s01 = rod.state.x[i].x / L;
            rod.state.v[i] = Vec3(0, 0, Real(0.1) * std::sin(2 * kPi * s01));
        }

        SolverParams p;
        p.dt = dt;
        p.substeps = sub;
        p.gravity = Vec3();

        const double e0 = rod.kineticEnergy() + rod.elasticEnergy();
        const Vec3 p0 = rod.linearMomentum();
        const Vec3 l0 = rod.angularMomentum();
        // The initial linear momentum is zero by construction, so scale the
        // momentum error by a momentum that actually exists: sum m_i |v_i|.
        double pScale = 0;
        for (std::size_t i = 0; i < rod.state.numParticles(); ++i)
            pScale += rod.state.mass[i] * norm(rod.state.v[i]);
        const double lScale = std::max(norm(l0), pScale * L);

        double eAfterTransient = 0, maxLate = 0, maxP = 0, maxL = 0, maxGain = 0;
        for (int i = 0; i <= steps; ++i) {
            if (i % 100 == 0) {
                const double ke = rod.kineticEnergy(), ee = rod.elasticEnergy();
                const double total = ke + ee;
                const double dp = norm(rod.linearMomentum() - p0) / pScale;
                const double dl = norm(rod.angularMomentum() - l0) / lScale;
                maxP = std::max(maxP, dp);
                maxL = std::max(maxL, dl);
                if (i == transientSteps) eAfterTransient = total;
                if (i > transientSteps && eAfterTransient > 0)
                    maxLate = std::max(maxLate, std::abs(total - eAfterTransient) / eAfterTransient);
                maxGain = std::max(maxGain, (total - e0) / e0);
                if (finest)
                    series.row(i, i * double(dt), ke, ee, total, std::abs(total - e0) / e0, dp, dl);
            }
            if (i < steps) step(rod, p);
        }

        const double diss = (e0 - eAfterTransient) / e0;
        dissipation.push_back(diss);
        worstLate = std::max(worstLate, maxLate);
        worstP = std::max(worstP, maxP);
        gains.push_back(maxGain);
        worstL = std::max(worstL, maxL);
        summary.row(sub, e0, diss, maxLate, maxGain, maxP, maxL);
    }

    // Both dissipation and any spurious energy gain are discretization errors:
    // halving the substep must reduce them.
    bool monotone = true;
    for (std::size_t i = 1; i < dissipation.size(); ++i)
        if (dissipation[i] > dissipation[i - 1] || gains[i] > gains[i - 1]) monotone = false;

    res.notes.push_back(fmt("  dissipation over the elastic transient: %.3f at 2 substeps -> "
                            "%.3f at 16", dissipation.front(), dissipation.back()));
    res.notes.push_back(fmt("  energy decays throughout: further loss after the first 20 s is "
                            "%.3g of the energy remaining there", worstLate));
    res.notes.push_back(fmt("  spurious energy gain vanishes with substepping: %.3g at 2 substeps "
                            "-> %.3g at 16", gains.front(), gains.back()));
    res.notes.push_back(fmt("  momentum error: linear %.3g, angular %.3g", worstP, worstL));

    res.checks.push_back(makeCheck("dissipation and gain fall with substep count",
                                   monotone ? 1.0 : 0.0, 1.0, 0.0));
    res.checks.push_back(makeCheck("no energy injected at 16 substeps", gains.back(), 0.0, 1e-3,
                                   1.0));
    res.checks.push_back(makeCheck("linear momentum conservation", worstP, 0.0, 1e-8, 1.0));
    res.checks.push_back(makeCheck("angular momentum conservation", worstL, 0.0, 1e-3, 1.0));
    return res;
}

// ---------------------------------------------------------------- cross-check
//
// A short, fully specified trajectory: same rod, same boundary condition, same
// step parameters, no damping, no adaptive anything. The NumPy mirror in
// tools/reference_prototype.py runs the identical setup and compares. Any
// disagreement beyond round-off means one of the two derivations is wrong, and
// that is worth far more than either code agreeing with itself.
namespace crs_crosscheck_config {
constexpr int kSegments = 16;
constexpr int kSteps = 200;
}  // namespace crs_crosscheck_config

CaseResult runCrossCheck(const std::string& outDir) {
    using namespace crs_crosscheck_config;
    CaseResult res;
    res.name = "cross-check trajectory (C++ vs NumPy mirror)";

    const RodMaterial mat = referenceMaterial();
    Rod rod = makeStraightRod(kSegments, Real(1), mat, Vec3(0, 0, 0), Vec3(1, 0, 0));
    clampRootExact(rod);

    SolverParams p;
    p.dt = Real(1e-3);
    p.substeps = 4;
    p.iterations = 1;
    p.gravity = Vec3(0, 0, Real(-9.81));
    for (int i = 0; i < kSteps; ++i) step(rod, p);

    Csv csv(outDir, "crosscheck.csv", "index,x,y,z,qw,qx,qy,qz");
    const std::size_t nP = rod.state.numParticles();
    for (std::size_t i = 0; i < nP; ++i) {
        const Vec3 x = rod.state.x[i];
        // Particles and frames are dumped in one table; frames past the particle
        // count are the trailing ghost, which is emitted too so the comparison
        // covers every value the solver touched.
        const Quat q = i < rod.state.numSegments() ? rod.state.q[i] : Quat();
        csv.row(i, x.x, x.y, x.z, q.w, q.x, q.y, q.z);
    }

    const double tip = rod.state.x.back().z;
    res.notes.push_back(fmt("  %.0f segments, %.0f steps, tip z = %.12g m", double(kSegments),
                            double(kSteps), tip));
    res.notes.push_back("  compare with: python tools/reference_prototype.py");
    res.checks.push_back(makeCheck("trajectory finite", std::isfinite(tip) ? 1.0 : 0.0, 1.0, 0.0));
    return res;
}

}  // namespace crs
