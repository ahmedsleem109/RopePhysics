// Demo scenes for the Phase 5 video. See scenes.cpp for the trajectory format.
#pragma once

#include <string>

namespace crs {

// Simulate the named scene and write <name>.rodtraj and <name>.json into
// outDir. Returns 0 on success.
int runScene(const std::string& name, const std::string& outDir);

}  // namespace crs
