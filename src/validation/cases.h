// Validation cases. Each one compares the solver against an independent
// analytic or high-accuracy-numerical reference, reports a relative error, and
// passes or fails against a fixed tolerance. They are meant to run on every
// commit, so none of them may take more than a few seconds.
#pragma once

#include <string>
#include <vector>

namespace crs {

struct Check {
    std::string name;
    double value = 0;
    double reference = 0;
    double relError = 0;
    double tol = 0;
    bool pass = false;
};

// Helper: record `value` against `reference`, pass if the relative error (or,
// for a near-zero reference, the absolute error against `scale`) is under tol.
Check makeCheck(const std::string& name, double value, double reference, double tol,
                double scale = 0);

// Pass if lo <= value <= hi. Used for convergence orders, where the claim is
// "at least first order", not "exactly this number".
Check makeRangeCheck(const std::string& name, double value, double lo, double hi);

struct CaseResult {
    std::string name;
    std::vector<Check> checks;
    std::vector<std::string> notes;  // printed under the case, e.g. fitted slopes
    // Set when the case could not run at all for an environmental reason (no
    // GPU, say) rather than because anything is wrong. A skipped case reports
    // its reason and does not count against the exit status -- a suite that
    // silently passes when it did not run is worse than one that fails.
    bool skipped = false;
    std::string skipReason;

    bool pass() const;
};

// outDir receives the CSV data each case emits; pass "" to skip writing.
CaseResult runSolverConvergence(const std::string& outDir);  // substeps/iterations budget
CaseResult runCantilever(const std::string& outDir);      // Euler-Bernoulli, convergence
CaseResult runElasticaMoment(const std::string& outDir);  // pure end moment -> circular arc
CaseResult runElasticaTipLoad(const std::string& outDir); // large-deflection tip load
CaseResult runHelix(const std::string& outDir);           // intrinsic curvature + twist
CaseResult runTwistBuckling(const std::string& outDir);   // Michell/Greenhill threshold
CaseResult runEnergyDrift(const std::string& outDir);     // free flight conservation

// Phase 2: contact, friction, self-collision.
CaseResult runContactPrimitives(const std::string& outDir);  // rest height on 4 primitives
CaseResult runIncline(const std::string& outDir);            // slip angle and friction law
CaseResult runCapstan(const std::string& outDir);            // exp(mu theta) tension ratio
CaseResult runCapstanFrictionSweep(const std::string& outDir);  // capstan across mu
CaseResult runSelfCollision(const std::string& outDir);      // no interpenetration

// Phase 3: constraint coloring and the CUDA port.
CaseResult runColoring(const std::string& outDir);         // conflict-free, 2 colours
CaseResult runGpuParity(const std::string& outDir);        // GPU reproduces the CPU
CaseResult runGpuDeterminism(const std::string& outDir);   // bitwise, run to run
CaseResult runGpuThroughput(const std::string& outDir);    // scaling curves

// Phase 4: scale and characterization.
CaseResult runTimestepEnvelope(const std::string& outDir);  // largest accurate substep
CaseResult runTimestepConvergence(const std::string& outDir);  // refinement in time
CaseResult runTimestepMotions(const std::string& outDir);  // envelope beyond the swing
CaseResult runThroughput(const std::string& outDir);        // batched CPU segment-substeps/s

// Applications: a real task swept on the GPU, checked against theory and the CPU.
CaseResult runCableHanging(const std::string& outDir);  // will a cable stay on a hook?
CaseResult runHarnessLearning(const std::string& outDir);  // a robot learns to route a harness

// Dumps a short, fully deterministic trajectory for tools/reference_prototype.py
// to reproduce. Cross-implementation agreement on the solver map is a stronger
// statement than either implementation agreeing with itself.
CaseResult runCrossCheck(const std::string& outDir);

}  // namespace crs
