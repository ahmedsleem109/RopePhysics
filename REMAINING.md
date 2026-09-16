# What remains to build

State at the end of the first build session: every case in `rodsim all` passes
(exit 0), with the three GPU cases **skipped**, not passed. See `README.md` for
results and `docs/writeup.md` for the full account.

Items are in priority order. Each says what "done" means.

---

## 1. Get the GPU running (blocks all of Phase 3 and half of Phase 4)

**Blocker:** driver 576.52 supports CUDA ≤ 12.9, but the only installed toolkit
with a compiler is CUDA 13.1. Its runtime fails with error 801. The kernels
compile; they have never executed.

- [ ] Update the NVIDIA driver to r580+ (needs admin, may need a reboot), **or**
      install a CUDA 12.x toolkit with `nvcc` and point `build.cmd` at it.
      (`pip install nvidia-cuda-nvcc-cu12` does not work on Windows: it ships
      only `ptxas.exe`.) If using 12.x, check nvcc accepts MSVC 14.44.
- [ ] `build.cmd`, then `build\rodsim.exe gpu-parity` — expect this to find bugs
      first; the kernels are untested.
      **Done when:** GPU matches the colour-ordered CPU trajectory to float
      precision (case tolerance 1e-4 m over 200 steps) for both the multi-kernel
      and fused strategies, at 16 / 48 / 128 segments.
- [ ] `build\rodsim.exe gpu-determinism`.
      **Done when:** repeated runs are bitwise identical for both strategies.
- [ ] `build\rodsim.exe gpu-throughput`.
      **Done when:** scaling curves exist vs batch size, rod length, substeps,
      for both strategies, and the knee in each is explained.

## 2. Phase 3 measurement the plan requires

- [ ] Profile the kernels with Nsight Compute (installed: 2025.4.0). Record
      occupancy, achieved bandwidth, and which kernels are
      bandwidth- vs latency-bound. Save screenshots into `docs/`.
- [ ] Measure launch overhead: multi-kernel issues 64 launches/step at
      8 substeps × 1 iteration; fused issues 1. Quantify the difference.
- [ ] Confirm the fused shared-memory limit on the device
      (`maxFusedSegments()`, footprint `104 n + 24` bytes per rod).
- [ ] Run the **entire** Phase 1–2 suite on the GPU path (plan's Phase 3 gate),
      not just the parity trajectory. Contacts and self-collision currently
      exist only on the CPU — see item 3.

## 3. GPU features not yet ported

- [ ] Contacts and friction on the GPU (the `Contact` struct is already a flat
      POD, one projection routine, designed for this).
- [ ] Self-collision broadphase on the GPU: key kernel → radix sort → cell-start
      scan (CPU version mirrors these stages in `collision.cpp`).
- [ ] Contact constraints need colouring too: re-colour per step, or use a
      Jacobi-style pass for contacts. Decide and measure.
- [ ] External forces/torques on the GPU (GPU path supports gravity only).
- [ ] Ghost frames with driven orientation (used by twist cases) on the GPU.

## 4. Phase 4 GPU deliverables

- [ ] Headline number measured on the GPU: "X million rod-segment-steps/sec on a
      laptop 3060", stated with its configuration.
- [ ] CPU vs GPU throughput plot (CPU data already in
      `docs/data/throughput_cpu.csv`).
- [ ] Re-render the demo video with the GPU grid scene and the GPU headline
      (`tools/make_video.py`; scene timing captions must then say GPU).

## 5. Validation gaps to close

- [ ] **Faster static solver.** The suite takes ~30 min because relaxed static
      solves need a Gauss–Seidel budget quadratic in segment count. A direct
      block-tridiagonal solve for the reference would make it run in seconds —
      needed for "validation runs on every commit".
- [ ] Wire `rodsim all` into CI (CMake `add_test` exists; no CI config yet).
- [ ] Timestep-refinement convergence study (only mesh refinement is done).
- [ ] Extend mesh sweeps: cantilever beyond n = 64; moment, tip-load and twist
      buckling beyond n = 32. Twist buckling converges at slope 0.82 (first
      order) — explain or improve.
- [ ] Stability rule measured on one scenario only (gravity cantilever), and it
      does not hold once elements are shorter than the rod diameter. Test other
      scenarios (contact-driven motion, whipping) and the short-element regime.
- [ ] Capstan tested at a single μ = 0.25; incline at one geometry. Add a μ
      sweep for the capstan.
- [ ] Contacts apply no torque to material frames (no rolling/torsional
      friction). Add if the demo or a validation needs it.
- [ ] Rod–primitive contact is per particle; add segment-based contact if
      segments must be longer than their diameter.

## 6. Optional (plan: "if time allows")

- [ ] Minimal Python / Warp-style binding.
- [ ] Trivial policy-learning task (cable reaching a target).

## 7. Launch and distribution (not engineering)

- [ ] Choose a repo host and put the link into the video end card
      (`python tools/make_video.py --repo "<link>"`) — currently a placeholder.
- [ ] Commit or host `out/video/demo.mp4` (42 MB, git-ignored).
- [ ] Publish the writeup; post the long version a few days after the video.
- [ ] Direct outreach, upstream contributions, peer venues (see plan).

---

## Environment notes for the next session

- Build: `build.cmd` (MSVC 14.44 pinned — nvcc rejects 14.50; Ninja; CMake at
  `C:\Program Files\CMake\bin`). Needs `cmd //c "D:\ropephysics\build.cmd"`
  from Git Bash.
- `rodsim.exe` cannot be rebuilt while an instance is running (link lock).
- Timing-sensitive cases (`throughput`, scenes) must run on an idle CPU.
- Video tooling: `imageio-ffmpeg` (pip, user space) supplies ffmpeg.
- Regenerate outputs: `rodsim scene drape`, `rodsim scene grid`,
  `python tools/plot_validation.py [--theme dark]`, `python tools/make_video.py`.
