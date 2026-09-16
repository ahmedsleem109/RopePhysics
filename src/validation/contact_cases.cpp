// Phase 2 validation: contact against analytic primitives, Coulomb friction,
// and rod self-collision.
//
// The two quantitative cases both locate a threshold the same way: measure a
// quantity that is identically zero below the threshold and linear above it,
// then extrapolate the linear part back to zero. That avoids ever having to
// choose an arbitrary "has it started moving yet" cutoff, which is the usual
// way friction tests quietly become tests of the tolerance rather than the
// physics.
#include <algorithm>
#include <cmath>

#include "../core/collision.h"
#include "../core/solver.h"
#include "cases.h"
#include "support.h"

namespace crs {

namespace {

// A rope: thin and floppy enough that bending stiffness is negligible next to
// tension, but stiff axially so it does not behave like a spring.
//
// The density is deliberately unphysical. The capstan threshold is a STATIC
// balance of tension and friction and does not contain the mass at all; mass
// only sets the dynamics used to detect slip. At a real 1000 kg/m^3 a 0.25 mm
// rope lumps about 1e-7 kg per particle, a 1 N end load accelerates it at
// 1e7 m/s^2, and the predictor group chi = h^2 a / l comes out near 40 -- three
// orders past where the failure study shows the solver blows up, and blow up
// it did. Scaling the density by 1e4 brings chi under the 0.05 guard at an
// affordable substep without touching anything the answer depends on.
RodMaterial ropeMaterial() {
    RodMaterial m;
    m.youngs = Real(5e9);
    m.poisson = Real(0.35);
    m.density = Real(1e7);
    m.radius = Real(2.5e-4);
    return m;
}

// Least-squares fit y = a x + b, returned as the x-intercept -b/a. Used to read
// a threshold off the linear regime above it.
double zeroCrossing(const std::vector<double>& xs, const std::vector<double>& ys) {
    const double n = double(xs.size());
    if (n < 2) return 0;
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (std::size_t i = 0; i < xs.size(); ++i) {
        sx += xs[i];
        sy += ys[i];
        sxx += xs[i] * xs[i];
        sxy += xs[i] * ys[i];
    }
    const double denom = n * sxx - sx * sx;
    if (denom == 0) return 0;
    const double slope = (n * sxy - sx * sy) / denom;
    if (slope == 0) return 0;
    return -((sy - slope * sx) / n) / slope;
}

}  // namespace

// ---------------------------------------------------------------- primitives
//
// Drop a rod onto each primitive in turn and check where it comes to rest. The
// support height is exact and known for all four shapes, so this catches a sign
// error or a wrong radius offset in any of them, and the residual penetration
// says whether the rigid contact is actually being enforced.
CaseResult runContactPrimitives(const std::string& outDir) {
    CaseResult res;
    res.name = "contact against analytic primitives";

    const RodMaterial mat = referenceMaterial();
    const Real L = Real(0.4);
    const int n = 32;
    const Real r = mat.radius;

    struct Scenario {
        const char* name;
        Primitive prim;
        Real supportZ;  // height of the rod centerline at rest
    };

    // All four carry friction. Resting on a sphere or on the crown of a
    // cylinder is an unstable balance, and although the setup is exactly
    // symmetric, the Gauss-Seidel sweep is not -- it runs from one end of the
    // rod to the other, which breaks the symmetry at round-off and is quite
    // enough to tip a frictionless rod off. Friction is not what is being
    // measured here; it is what makes the thing being measured hold still.
    const Real mu = Real(0.8);
    const Real sphereR = Real(0.25), capsuleR = Real(0.15), boxH = Real(0.2);
    const std::vector<Scenario> scenarios = {
        {"plane", Primitive::makePlane(Vec3(0, 0, 0), Vec3(0, 0, 1), mu), r},
        // Draped over the sphere: only the middle particle touches, and it ends
        // up at the north pole plus one rod radius.
        {"sphere", Primitive::makeSphere(Vec3(0, 0, 0), sphereR, mu), sphereR + r},
        // Capsule axis PARALLEL to the rod, so the rod lies along the top of the
        // cylinder and every particle is in contact -- a line contact, which is
        // what exercises the capsule's axis projection rather than just its cap.
        {"capsule", Primitive::makeCapsule(Vec3(-1, 0, 0), Vec3(1, 0, 0), capsuleR, mu),
         capsuleR + r},
        {"box", Primitive::makeBox(Vec3(0, 0, 0), Vec3(1, 1, boxH), Quat(), mu), boxH + r},
    };

    Csv csv(outDir, "contact_primitives.csv",
            "shape,rest_height,expected,error,max_penetration,steps,converged");

    double worstHeight = 0, worstPenetration = 0;
    for (const Scenario& sc : scenarios) {
        CollisionWorld world;
        world.primitives.push_back(sc.prim);

        // Start just above the support height and let it settle.
        // Start barely clear of the surface: a long drop only adds a bounce
        // whose energy has to be damped away again.
        Rod rod = makeStraightRod(n, L, mat, Vec3(-L / 2, 0, sc.supportZ + Real(0.004)),
                                  Vec3(1, 0, 0));
        SolverParams p;
        p.dt = Real(2e-3);
        p.substeps = 16;
        p.iterations = 8;
        p.gravity = Vec3(0, 0, Real(-9.81));
        p.linearDamping = Real(12);
        p.angularDamping = Real(12);

        SolverContext ctx;
        const RelaxReport rep = relaxToEquilibrium(rod, p, world, ctx, 8000, Real(1e-6));

        // The lowest point of contact: for the sphere and box that is the rod's
        // centre particle, for plane and capsule the whole rod.
        Real lowest = rod.state.x[0].z;
        for (const Vec3& x : rod.state.x) lowest = std::min(lowest, x.z);

        Real penetration = 0;
        for (const Vec3& x : rod.state.x) {
            Vec3 nrm;
            const Real d = signedDistance(sc.prim, x, nrm);
            penetration = std::max(penetration, r - d);
        }

        // Compare the contact point, which for the sphere is the centre of the
        // rod and for the flat/cylindrical shapes is every particle.
        const double contactHeight =
            sc.prim.type == Primitive::kSphere ? rod.state.x[n / 2].z : lowest;
        const double err = std::abs(contactHeight - sc.supportZ) / sc.supportZ;
        worstHeight = std::max(worstHeight, err);
        worstPenetration = std::max(worstPenetration, double(penetration));

        csv.row(sc.name, contactHeight, sc.supportZ, err, penetration, rep.steps,
                rep.converged ? 1 : 0);
    }

    res.notes.push_back(fmt("  worst rest-height error %.3g, worst residual penetration %.3g m",
                            worstHeight, worstPenetration));
    res.checks.push_back(makeCheck("rest height on all four primitives", worstHeight, 0.0, 0.02,
                                   1.0));
    res.checks.push_back(makeCheck("residual penetration", worstPenetration, 0.0, 1e-5, 1.0));
    return res;
}

// ---------------------------------------------------------------- incline
//
// A rod on a slope of angle alpha with friction mu. Statics says it stays put
// while tan(alpha) <= mu; dynamics says that once it goes, it accelerates at
//
//     a = g (sin alpha - mu cos alpha)
//
// Measuring a(alpha) tests both at once, and far more sharply than the
// threshold alone: the slip angle is where the measured acceleration crosses
// zero, and the slope of that line is a direct check on the friction law.
CaseResult runIncline(const std::string& outDir) {
    CaseResult res;
    res.name = "block on incline: slip angle and friction law";

    const RodMaterial mat = referenceMaterial();
    const Real L = Real(0.3);
    const int n = 16;
    const Real g = Real(9.81);

    Csv csv(outDir, "incline.csv",
            "mu,alpha,tan_alpha,accel,accel_theory,mu_effective,displacement");

    const std::vector<double> mus = {0.1, 0.3, 0.5, 0.8};
    double worstAngle = 0, worstAccel = 0;

    for (double mu : mus) {
        const double alphaC = std::atan(mu);
        std::vector<double> alphas, accels;

        for (int k = 0; k <= 10; ++k) {
            // Sample from below the threshold to well above it.
            const double alpha = alphaC * (0.7 + 0.09 * k);
            const Vec3 normal(-std::sin(alpha), 0, std::cos(alpha));
            // In the plane, and descending: the direction gravity drives it.
            const Vec3 downSlope(-std::cos(alpha), 0, -std::sin(alpha));

            CollisionWorld world;
            world.primitives.push_back(
                Primitive::makePlane(Vec3(0, 0, 0), normal, Real(mu)));

            // Lay the rod along the slope, one radius off the surface so it
            // starts exactly in contact.
            Rod rod = makeStraightRod(n, L, mat, normal * mat.radius, downSlope);

            SolverParams p;
            p.dt = Real(1e-3);
            p.substeps = 8;
            p.iterations = 4;
            p.gravity = Vec3(0, 0, -g);

            SolverContext ctx;
            // Settle onto the surface first, so the measurement window contains
            // only sliding and not the initial contact transient.
            SolverParams settle = p;
            settle.linearDamping = Real(30);
            settle.angularDamping = Real(30);
            for (int i = 0; i < 300; ++i) step(rod, settle, world, ctx);

            // Sample the down-slope coordinate at three equally spaced times.
            // The second difference gives the acceleration while cancelling the
            // velocity the rod already carried out of the settling phase, which
            // would otherwise inflate it by a good 15%.
            const int half = 200;
            const double T = 2 * half * double(p.dt);
            const double s0 = dot(rod.state.x[n / 2], downSlope);
            for (int i = 0; i < half; ++i) step(rod, p, world, ctx);
            const double s1 = dot(rod.state.x[n / 2], downSlope);
            for (int i = 0; i < half; ++i) step(rod, p, world, ctx);
            const double s2 = dot(rod.state.x[n / 2], downSlope);

            const double disp = s2 - s0;
            const double accel = 4 * (s2 - 2 * s1 + s0) / (T * T);
            const double theory = std::max(0.0, g * (std::sin(alpha) - mu * std::cos(alpha)));

            // Invert the friction law instead of comparing accelerations
            // directly. Near the threshold a = g(sin a - mu cos a) is a small
            // difference of two large numbers, so a 2% error in the threshold
            // shows up as a 30% error in the acceleration while saying nothing
            // useful. The coefficient it implies is the well-conditioned form of
            // the same statement, and it is the quantity the model actually
            // claims to reproduce.
            const double muEff = (g * std::sin(alpha) - accel) / (g * std::cos(alpha));
            csv.row(mu, alpha, std::tan(alpha), accel, theory, muEff, disp);
            if (accel > 0.2) {  // clearly sliding: keep for the fit
                alphas.push_back(alpha);
                accels.push_back(accel);
                worstAccel = std::max(worstAccel, std::abs(muEff - mu) / mu);
            }
        }

        const double measured = zeroCrossing(alphas, accels);
        worstAngle = std::max(worstAngle, std::abs(measured - alphaC) / alphaC);
        res.notes.push_back(fmt("  mu = %.2f: slip angle %.5f rad measured vs atan(mu) = %.5f", mu,
                                measured, alphaC));
    }

    res.checks.push_back(makeCheck("slip angle vs atan(mu)", worstAngle, 0.0, 0.05, 1.0));
    res.notes.push_back(fmt("  friction coefficient recovered from the sliding acceleration is "
                            "within %.3g of the value asked for", worstAccel));
    res.checks.push_back(makeCheck("effective mu while sliding", worstAccel, 0.0, 0.08, 1.0));
    return res;
}

// ---------------------------------------------------------------- capstan
//
// The capstan equation: a rope wrapped by angle theta around a cylinder with
// friction mu holds against a tension ratio of
//
//     T2 / T1 = exp(mu theta)
//
// before it slips. This is the sharpest test in the suite of friction and
// contact working together, because the answer is exponential in both the
// friction coefficient and the wrap angle -- a contact model that is merely
// plausible will not reproduce a factor that grows like e^(mu theta) across
// several wrap angles.
//
// Measurement: above the critical ratio the rope slides at a terminal speed
// proportional to the excess tension, so the critical ratio is where that speed
// extrapolates to zero.
CaseResult runCapstan(const std::string& outDir) {
    CaseResult res;
    res.name = "capstan equation (friction and contact together)";

    const RodMaterial mat = ropeMaterial();
    const Real R = Real(0.05);       // cylinder radius
    const Real mu = Real(0.25);
    const Real T1 = Real(1);         // held tension
    const Real lead = Real(0.03);    // straight lead-in / lead-out
    const Real segLen = Real(1e-3);

    Csv csv(outDir, "capstan.csv", "wrap_turns,theta,ratio,terminal_speed,substeps,"
                                   "capstan_ratio");

    const std::vector<double> wraps = {0.25, 0.5, 0.75, 1.0};  // in turns
    double worst = 0;

    for (double turns : wraps) {
        const double theta = 2 * kPi * turns;
        const double capstanRatio = std::exp(mu * theta);

        // Centerline: straight lead-in, the wrap at radius R + r, straight
        // lead-out, all in the z = 0 plane around a cylinder along z.
        const Real Rc = R + mat.radius;
        std::vector<Vec3> pts;
        auto arcPoint = [&](double phi) {
            return Vec3(Rc * std::cos(phi), Rc * std::sin(phi), 0);
        };
        auto tangent = [&](double phi) {
            return Vec3(-std::sin(phi), std::cos(phi), 0);
        };

        const int leadN = std::max(2, int(lead / segLen));
        for (int i = leadN; i >= 1; --i)
            pts.push_back(arcPoint(0) - tangent(0) * (segLen * i));
        const int arcN = std::max(4, int(Rc * theta / segLen));
        for (int i = 0; i <= arcN; ++i) pts.push_back(arcPoint(theta * i / arcN));
        for (int i = 1; i <= leadN; ++i)
            pts.push_back(arcPoint(theta) + tangent(theta) * (segLen * i));

        const Vec3 pullIn = -tangent(0);           // direction the held end is pulled
        const Vec3 pullOut = tangent(theta);       // direction the loaded end is pulled

        CollisionWorld world;
        world.primitives.push_back(
            Primitive::makeCapsule(Vec3(0, 0, -1), Vec3(0, 0, 1), R, mu));

        std::vector<double> allRatios, allSpeeds;
        // Bracket the threshold from both sides; below it the rope holds and
        // those points are dropped from the fit.
        for (int k = 0; k <= 9; ++k) {
            const double ratio = capstanRatio * (0.70 + 0.12 * k);

            Rod rod = makeRodFromCenterline(pts, mat, Vec3(0, 0, 1));
            // The rope is built already wrapped, but it is a STRAIGHT rope: its
            // unstressed shape has no curvature, so clear the rest curvature the
            // builder inferred from the initial pose.
            for (std::size_t i = 0; i < rod.bend.size(); ++i) rod.bend.restDarboux[i] = Vec3();

            rod.state.extForce.front() = pullIn * T1;
            rod.state.extForce.back() = pullOut * Real(T1 * ratio);

            SolverParams p;
            p.dt = Real(5e-4);
            p.substeps = std::min(64, std::max(8, stableSubsteps(rod, Vec3(), p.dt)));
            p.iterations = 6;
            p.gravity = Vec3();
            // Light damping so the elastic ringing of the lead-ins dies away and
            // sliding reaches a terminal speed proportional to the excess pull.
            p.linearDamping = Real(40);
            p.angularDamping = Real(40);

            SolverContext ctx;
            const int settleSteps = 400, measureSteps = 400;
            for (int i = 0; i < settleSteps; ++i) step(rod, p, world, ctx);

            // Slip is motion along the rope, so measure it along the surface
            // tangent at the middle of the wrap, signed towards the pulled end.
            const Vec3 midTangent(-std::sin(theta / 2), std::cos(theta / 2), 0);
            const int mid = int(pts.size()) / 2;
            const Vec3 before = rod.state.x[mid];
            for (int i = 0; i < measureSteps; ++i) step(rod, p, world, ctx);
            const double speed =
                dot(rod.state.x[mid] - before, midTangent) / (measureSteps * double(p.dt));

            csv.row(turns, theta, ratio, speed, p.substeps, capstanRatio);
            allRatios.push_back(ratio);
            allSpeeds.push_back(speed);
        }

        // Keep only the points that are genuinely sliding. Below the threshold
        // the rope is not motionless: it creeps at ~1e-4 m/s while its lead-ins
        // settle elastically, and an absolute cutoff at that level let three
        // creeping points into a one-turn fit -- which dragged the intercept
        // from 4.80 to 4.15 and made correct friction look 14% weak. Sliding
        // speeds are tens of times larger, so a cutoff relative to the fastest
        // point separates the two regimes cleanly at every wrap angle.
        const double fastest = *std::max_element(allSpeeds.begin(), allSpeeds.end());
        std::vector<double> ratios, speeds;
        for (std::size_t i = 0; i < allSpeeds.size(); ++i) {
            if (allSpeeds[i] > 0.05 * fastest) {
                ratios.push_back(allRatios[i]);
                speeds.push_back(allSpeeds[i]);
            }
        }

        const double measured = zeroCrossing(ratios, speeds);
        const double err = std::abs(measured - capstanRatio) / capstanRatio;
        worst = std::max(worst, err);
        res.notes.push_back(fmt("  %.2f turns: critical ratio %.4f measured vs exp(mu theta) = "
                                "%.4f", turns, measured, capstanRatio));
    }

    res.notes.push_back(fmt("  mu = %.2f, worst relative error %.3g", double(mu), worst));
    res.checks.push_back(makeCheck("capstan ratio across wrap angles", worst, 0.0, 0.06, 1.0));
    return res;
}

// ---------------------------------------------------------------- self-contact
//
// A rope dropped end-first onto the floor buckles and coils into a pile, and a
// pile is nothing but sustained self-contact: strands resting on strands, under
// gravity, for seconds. Two claims are checked. Nothing passes through anything
// -- the minimum distance between strands never falls below the diameter by more
// than the solver's tolerance -- and, just as important, self-contact actually
// happened. A self-collision test that passes because the strands never met
// proves nothing, and the first version of this case did exactly that.
CaseResult runSelfCollision(const std::string& outDir) {
    CaseResult res;
    res.name = "self-collision (rope coiling into a pile)";

    RodMaterial mat = referenceMaterial();
    mat.radius = Real(5e-3);
    mat.youngs = Real(2e6);
    const int n = 150;
    const Real L = Real(1.2);
    const Real diameter = 2 * mat.radius;

    // Nearly vertical, with a slight lean so it coils rather than stacking.
    Rod rod = makeStraightRod(n, L, mat, Vec3(0, 0, Real(0.02)),
                              normalize(Vec3(Real(0.08), Real(0.03), 1)));
    // Flip it so the free end falls first onto the floor: the bottom particle
    // starts just above the plane and the rope lands on itself as it descends.
    for (Vec3& x : rod.state.x) x.z = L + Real(0.05) - x.z;
    rod.state.xPrev = rod.state.x;
    setRestFromCurrent(rod);

    CollisionWorld world;
    world.primitives.push_back(Primitive::makePlane(Vec3(0, 0, 0), Vec3(0, 0, 1), Real(0.6)));
    world.selfCollision = true;
    world.selfFriction = Real(0.3);

    SolverParams p;
    p.dt = Real(1e-3);
    p.substeps = 16;
    p.iterations = 4;
    p.gravity = Vec3(0, 0, Real(-9.81));
    p.linearDamping = Real(1);
    p.angularDamping = Real(1);

    SolverContext ctx;
    Csv csv(outDir, "self_collision.csv", "step,time,min_distance,diameter,self_contacts");

    const int gap = selfCollisionIndexGap(rod, world);
    double worstOverlap = 0;
    int peakSelfContacts = 0, framesWithContact = 0, samples = 0;
    const int steps = 2500;
    for (int i = 0; i <= steps; ++i) {
        if (i % 25 == 0) {
            // Brute force over every pair the solver is responsible for: an
            // independent check, never the hash the solver itself used.
            double minDist = 1e30;
            const int nSeg = int(rod.stretch.size());
            for (int a = 0; a < nSeg; ++a)
                for (int b = a + gap; b < nSeg; ++b) {
                    Real u, v;
                    const Vec3 a0 = rod.state.x[a], a1 = rod.state.x[a + 1];
                    const Vec3 b0 = rod.state.x[b], b1 = rod.state.x[b + 1];
                    closestPointsBetweenSegments(a0, a1, b0, b1, u, v);
                    minDist = std::min(minDist,
                                       double(norm((a0 + (a1 - a0) * u) - (b0 + (b1 - b0) * v))));
                }
            int self = 0;
            for (const Contact& c : ctx.contacts.contacts) self += c.count == 4 ? 1 : 0;
            peakSelfContacts = std::max(peakSelfContacts, self);
            framesWithContact += self > 0 ? 1 : 0;
            ++samples;
            worstOverlap = std::max(worstOverlap, (double(diameter) - minDist) / double(diameter));
            csv.row(i, i * double(p.dt), minDist, diameter, self);
        }
        if (i < steps) step(rod, p, world, ctx);
    }

    res.notes.push_back(fmt("  peak %.0f simultaneous self-contacts; self-contact in %.0f%% of "
                            "sampled frames", double(peakSelfContacts),
                            100.0 * framesWithContact / samples));
    res.notes.push_back(fmt("  worst interpenetration %.3g of the rope diameter over %.0f steps",
                            worstOverlap, double(steps)));
    res.checks.push_back(makeCheck("self-contact actually occurred", peakSelfContacts >= 10 ? 1 : 0,
                                   1.0, 0.0));
    res.checks.push_back(makeCheck("no self-interpenetration", std::max(0.0, worstOverlap), 0.0,
                                   0.05, 1.0));
    return res;
}

}  // namespace crs
