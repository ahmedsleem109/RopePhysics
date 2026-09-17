# What remains to build

## ▶ START HERE — next session (handoff written 2026-09-17)

Everything below "Done" is merged to `main` and pushed. Work the queue **in
this order**, one branch per item, merge to main with `--no-ff`, push each
branch on its own (see Working notes), and update this file as items close.

### Queue, in order

1. **Full Phase 1–2 suite on the GPU path.** Every CPU validation case that has
   a GPU-expressible scenario (cantilever/moment dynamics, contact, incline,
   capstan, self-collision, energy) run through `gpu::Batch` and compared with
   the CPU result / the analytic reference. New case(s) in `src/gpu/gpu_cases.cpp`,
   skipping without CUDA. Judge against a float CPU build (`CRS_REAL_FLOAT`,
   g++ in WSL), not bitwise double.
2. **Colour self contacts per rod on the GPU** (self-collision costs ~11x
   throughput because projection within a rod is sequential). Must keep
   `gpu-parity` self-collision checks passing within float rounding; report the
   throughput gain in `gpu-throughput` / writeup §5.
3. **Re-measure the GPU headline** (1.17 B segment-substeps/s, thermal drift seen
   1.16 B -> 0.94 B) on a cool, idle GPU; update README "How fast" and writeup.
4. **Nsight Compute profile** (installed 2025.4.0) of the fused kernel:
   occupancy, bandwidth, which kernels are latency- vs bandwidth-bound. Use the
   CLI (`ncu`/`nsys`) to capture; GUI screenshots need the user — ask them.
5. **Friction accuracy near the capstan boundary.** At 8 substeps x 2 (the
   float-safe cable-hanging setting) a 1.6:1 drape at mu 0.20 slides even in
   double; 32 substeps hold it (`build/rodhang.exe 1.6 0.20 0.25 32 2 cpu`).
   Soft-cable boundary median is 0.059 (safe side). Options: static-friction
   anchor per contact; or run the GPU scene in coordinates small enough that
   float resolves 32 substeps (float spacing ~ |coord|; see
   HangingCable::floatMotionMargin). Goal: median well under 0.03 without
   creep coming back (`build/rodcreep.exe 0.25` must stay ~0 mm/s).
6. **Short-element regime of the timestep rule** (elements shorter than the rod
   diameter), extending `timestep-motions` / `timestep-envelope`.
7. **Contact torque** (rolling / torsional friction) and **segment-based
   rod–primitive contact** (for segments longer than the diameter).
8. **Media:** re-render the long demo (`tools/make_video.py --repo
   "https://github.com/ahmedsleem109/RopePhysics"`) with current solver and GPU
   numbers. **Ask the user before creating a GitHub release** (public) to host
   demo.mp4 / harness.mp4.
9. **Python binding** (minimal, Warp-style): design choice (pybind11 vs nanobind
   vs ctypes) — propose one and ask if unclear.

### Done 2026-09-17 (for context)
- Harness demo: `rodsim harness-learning` (CEM, 10 x 1024 GPU attempts,
  2% -> 85% in 129 s; learned 1024/1024 vs hand-written 0/1024; CPU/GPU agree),
  `tools/make_harness_video.py`, README lead, writeup §7. Videos in out/video/
  (git-ignored): harness.mp4 (50 s), harness_learned_policy.mp4 (19 s,
  tools/make_learned_policy_video.py). LinkedIn post drafted for the user.
- CI: actions on Node 24; one branch per push triggers CI.
- `capstan-mu` (0.71%), `timestep-convergence` (first order), tip-load sweep to
  n = 256 (slope 1.96 vs Reissner), `timestep-motions` (swing/drop/whip, 1-6%
  of an element per substep), twist buckling by growth rate (0.18% at n = 48,
  second order, 3 min instead of 9).
- Friction creep fixed: `CollisionWorld::contactMarginRadii = 0.25` (CPU + GPU).
- Probes in tools/experiments: rodtwist (twist window/rate), rodcreep (creep),
  rodhang (CPU vs GPU hang placement).

### Working notes
- Build from Git Bash: `cmd //c "D:\ropephysics\build.cmd"`; rodsim.exe cannot
  relink while running.
- Python patch scripts via heredoc mangle backslash escapes (newlines in C
  string literals, Windows paths): write those with the Edit/Write tools.
- WSL Ubuntu has g++ (use for float-vs-double builds with `-DCRS_REAL_FLOAT`).
- Pushes: SSH key is not on GitHub; push with
  `git -c credential.helper= -c 'credential.helper=!gh auth git-credential' push https://github.com/ahmedsleem109/RopePhysics.git <branches>`.
  Push ONE branch per `git push`: pushes naming several refs at once created
  no workflow runs (observed 2026-09-17); single-ref pushes trigger CI.
- Workflow the user asked for: each feature on its own branch, merged to main
  with --no-ff, pushed.

---

## Earlier backlog (still open)

- **GPU:** Nsight profiling (occupancy, bandwidth, latency-bound kernels,
  screenshots into docs/); run the full Phase 1–2 suite on the GPU path;
  colour self contacts per rod (self-collision costs ~11x throughput);
  re-measure GPU headline numbers on a cool GPU (thermal drift seen: 1.16 B →
  0.94 B within a session).
- **Friction near the capstan boundary** is limited by the substep: at 8 x 2
  (the float-safe cable-hanging setting) a 1.6:1 drape at mu 0.20 slides even in
  double; 32 substeps hold it. Options: run the GPU scene near the origin or in
  local coordinates so float resolves smaller substeps, or a static-friction
  anchor per contact. Creep itself is fixed (contact margin, 2026-09-17).
- **Validation gaps:** the short-element regime of the timestep rule; contact
  torque (rolling / torsional friction); segment-based rod–primitive contact.
  Done 2026-09-17: tip-load sweep to n = 256 (slope 1.96 vs Reissner),
  `timestep-motions` (swing/drop/whip), twist buckling by growth rate (0.18% at
  n = 48, second order; the old slope 0.82 was the detection window).
  Done 2026-09-17: `capstan-mu` (μ 0.1–0.75, worst 0.71%) and
  `timestep-convergence` (first order in the substep, observed 1.00).
- **CI:** consider a shorter per-commit suite (twist buckling 9 min, capstan
  3 min). Done 2026-09-17: actions on Node 24 releases; pushes trigger CI when
  each branch is pushed on its own.
- **Media:** re-render the long demo video (tools/make_video.py) with the fixed
  solver, GPU numbers and repo link
  (`--repo "https://github.com/ahmedsleem109/RopePhysics"`); host demo.mp4 as a
  GitHub release asset.
- **Optional:** Python binding.

Detailed history of what was done (GPU bring-up, direct statics, frame bug,
timestep envelope, GPU loads/contacts/self-collision, cable hanging) is in
git log, README and docs/writeup.md.

---

## Completed items log (from earlier sessions)

## 1. Get the GPU running (blocks all of Phase 3 and half of Phase 4)

**Blocker:** driver 576.52 supports CUDA ≤ 12.9, but the only installed toolkit
with a compiler is CUDA 13.1. Its runtime fails with error 801. The kernels
compile; they have never executed.

- [x] Driver updated to 616.92 (CUDA 13.4) on 2026-09-16; kernels run.
- [x] `gpu-parity` passes. The original 1e-4 m @ 200 steps tolerance was below
      float rounding (float-vs-double CPU alone: 2.5e-4 m). Now checked at 1 step
      (2e-8 m), at 200 steps against the float envelope, and multi-kernel ==
      fused bitwise. No kernel bugs found.
- [x] `gpu-determinism` passes (bitwise, both strategies).
- [x] `gpu-throughput`: curves measured and knees explained in writeup §5;
      plot `docs/figs/throughput_gpu.png`.

## 2. Phase 3 measurement the plan requires

- [ ] Profile the kernels with Nsight Compute (installed: 2025.4.0). Record
      occupancy, achieved bandwidth, and which kernels are
      bandwidth- vs latency-bound. Save screenshots into `docs/`.
- [x] Launch overhead: ~15 µs/launch; 1 ms/step multi-kernel vs 88 µs fused
      at one rod (11×); 0.6× fused at saturation.
- [x] Fused shared-memory limit on the device: 383 segments (`128 n + 36` bytes).
- [ ] Run the **entire** Phase 1–2 suite on the GPU path (plan's Phase 3 gate),
      not just the parity trajectory. Contacts and self-collision currently
      exist only on the CPU — see item 3.

## 3. GPU features not yet ported

- [x] Contacts and friction against primitives on the GPU. Geometry shared via
      `src/core/geometry.h`; parity scenario "contact" within the float envelope
      (1.6e-7 m at 200 steps); 20% throughput cost for two primitives.
- [x] Self-collision on the GPU: per-rod hash, parallel count/fill into a pool,
      projection in CPU order. Restart parity 6e-6 m (float envelope 1.8e-5 m);
      throughput 0.09x plain.
- [x] Decided: no colouring. World contacts are per particle and conflict-free;
      self contacts are projected sequentially per rod to keep the CPU's order.
      Measured cost above; colouring self contacts is the next speed-up.
- [x] External forces/torques on the GPU (`Batch::setLoads`); parity scenario
      "loaded" matches the CPU to 4.9e-8 m after one step.
- [x] Driven fixed frames on the GPU (same API; root twist in the parity case).

## 4. Phase 4 GPU deliverables

- [x] Headline: 1.17 B segment-substeps/s on a laptop RTX 3060 (16 384 rods ×
      64 segments × 8 substeps, fused); 1.86 B at 2048 × 256.
- [x] CPU vs GPU throughput plot: `docs/figs/throughput_gpu.png`.
- [ ] Re-render the demo video with the GPU grid scene and the GPU headline
      (`tools/make_video.py`; scene timing captions must then say GPU).

## 5. Validation gaps to close

- [x] **Faster static solver.** Done: `src/core/statics.cpp`, Newton on the
      banded equilibrium system with load continuation. Cantilever, moment and
      tip-load cases went from ~11 min to ~1 s, and the XPBD relaxation is
      checked against it in `solver-convergence`. Suite is now ~17 min, mostly
      twist buckling (9 min) and capstan (3 min).
- [x] **Stretch/shear frame bug.** Found by the n = 256 cantilever: compliance
      was applied in world axes. Fixed on CPU, GPU kernel and NumPy mirror;
      oblique-axis regression check added. GPU kernel change is compiled but,
      like the rest, never run.
- [x] **Stability study redesigned** as `timestep-envelope`: largest substep
      keeping worst strain under 1% (monotone in dt, unlike the old energy
      bound). Limit at 1.1–3.1% of an element moved per substep across
      E = 1e7..1e9 and 16–64 segments. Replaces stability-envelope and
      failure-study.
- [ ] Re-run demo scenes and re-render the video (simulated before the fix).
- [~] CI: `.github/workflows/validation.yml` written (CPU, Ubuntu); the code
      builds warning-clean under g++ 13. Not yet run, as there is no remote. It
      will go red on the two stability cases until they are redesigned.
      Consider shortening twist buckling / capstan for per-commit runs.
- [x] Timestep-refinement convergence study (`timestep-convergence`: first order).
- [x] Extend mesh sweeps: cantilever (n = 256), moment (n = 128), tip load
      (n = 256, slope 1.96 vs Reissner's extensible elastica), twist buckling
      (n = 48, second order by growth rate).
- [x] Stability rule across motions (`timestep-motions`: swing, drop, whip; 1-6%
      of an element per substep; contact 3-5x tighter by landing speed).
- [ ] The short-element regime (elements shorter than the rod diameter).
- [x] Capstan μ sweep (`capstan-mu`, μ = 0.1–0.75 at half a turn, worst 0.71%).
      The incline is still one geometry.
- [ ] Contacts apply no torque to material frames (no rolling/torsional
      friction). Add if the demo or a validation needs it.
- [ ] Rod–primitive contact is per particle; add segment-based contact if
      segments must be longer than their diameter.

## 5b. Applications

- [x] Cable hanging: GPU sweep of friction × placement × stiffness, capstan
      boundary on the safe side, CPU/GPU agreement, demo scene and video.
- [x] Kinematic (moving) pinned particles on CPU and GPU, for grippers.
- [x] Friction creep: contacts generated within 0.25 r of the surface
      (`contactMarginRadii`), CPU and GPU; 1.6 mm/s -> 0 at mu = 0.5.
- [x] Per-rod friction (`Batch::setMaterialScales`).
- [x] A manipulation demo that uses the moving gripper (harness routing, learned on the GPU).

## 6. Optional (plan: "if time allows")

- [ ] Minimal Python / Warp-style binding.
- [x] Policy-learning task (harness routing, `rodsim harness-learning`).

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
