// rodsim -- validation driver for the CPU Cosserat rod reference.
//
//   rodsim all [--out DIR]     run every case, write CSVs, exit nonzero on fail
//   rodsim <case> [--out DIR]  run one case
//   rodsim scene <drape|grid> [--out DIR]
//                              simulate a demo scene for the offline renderer
//
// Exit status is the number of failed checks, so this drops straight into CI.
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "validation/cases.h"
#include "validation/scenes.h"

namespace {

using Runner = crs::CaseResult (*)(const std::string&);

struct Entry {
    const char* name;
    Runner run;
};

const Entry kCases[] = {
    {"solver-convergence", crs::runSolverConvergence},
    {"cantilever", crs::runCantilever},
    {"elastica-moment", crs::runElasticaMoment},
    {"elastica-tip", crs::runElasticaTipLoad},
    {"helix", crs::runHelix},
    {"twist-buckling", crs::runTwistBuckling},
    {"energy", crs::runEnergyDrift},
    {"contact", crs::runContactPrimitives},
    {"incline", crs::runIncline},
    {"capstan", crs::runCapstan},
    {"self-collision", crs::runSelfCollision},
    {"coloring", crs::runColoring},
    {"gpu-parity", crs::runGpuParity},
    {"gpu-determinism", crs::runGpuDeterminism},
    {"gpu-throughput", crs::runGpuThroughput},
    {"timestep-envelope", crs::runTimestepEnvelope},
    {"throughput", crs::runThroughput},
    {"crosscheck", crs::runCrossCheck},
};

int report(const crs::CaseResult& r) {
    std::printf("\n%s\n", r.name.c_str());
    for (const std::string& n : r.notes) std::printf("%s\n", n.c_str());
    if (r.skipped) {
        // A skipped case is loudly skipped, never quietly passed: a suite that
        // reports success for something it did not run is worse than one that
        // fails.
        std::printf("  [SKIP] %s\n", r.skipReason.c_str());
        return 0;
    }
    int failed = 0;
    for (const crs::Check& c : r.checks) {
        std::printf("  [%s] %-44s value %-12.6g ref %-12.6g err %-10.3g tol %.3g\n",
                    c.pass ? "PASS" : "FAIL", c.name.c_str(), c.value, c.reference, c.relError,
                    c.tol);
        if (!c.pass) ++failed;
    }
    return failed;
}

void usage() {
    std::printf("usage: rodsim <all");
    for (const Entry& e : kCases) std::printf("|%s", e.name);
    std::printf("> [--out DIR]\n");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        usage();
        return 2;
    }
    const std::string what = argv[1];
    if (what == "scene") {
        if (argc < 3) {
            usage();
            return 2;
        }
        std::string sceneDir = "out/scenes";
        for (int i = 3; i < argc; ++i)
            if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc) sceneDir = argv[++i];
        return crs::runScene(argv[2], sceneDir);
    }
    std::string outDir = "docs/data";
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc) outDir = argv[++i];
        else if (std::strcmp(argv[i], "--no-out") == 0) outDir.clear();
    }

    std::vector<Runner> toRun;
    if (what == "all") {
        for (const Entry& e : kCases) toRun.push_back(e.run);
    } else {
        for (const Entry& e : kCases)
            if (what == e.name) toRun.push_back(e.run);
        if (toRun.empty()) {
            usage();
            return 2;
        }
    }

    int failed = 0;
    for (Runner r : toRun) failed += report(r(outDir));

    std::printf("\n%s: %d failed check(s)\n", failed == 0 ? "PASS" : "FAIL", failed);
    return failed;
}
