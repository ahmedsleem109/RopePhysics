# What remains to build

## Wire-harness demo (branch `feature/harness-routing`, 2026-09-17)

Goal: a demo a robotics-simulation company (target: Vsim) gets in seconds, a
robot arm routing a wire harness, learned on the GPU across randomized cables.

Done:
- Task `apps::HarnessTask`: pegs offset into an S-route, score 3 = routed and
  laid against the pegs, clip = between post centres, 10 ms gripper control.
- `rodsim harness-learning`: CEM, 10 x 1024 attempts, 2% -> 86% in 127 s; the
  learned motion routes 1024/1024 fresh cables, hand-written 0/1024; CPU/GPU
  agree 4/4. Deterministic (bitwise-identical rerun).
- `rodsim scene harness --out <dir>` replays `<dir>/harness.policy` and
  `<dir>/harness.cable` ("youngsScale frictionScale").
- Renderer: board, pegs, clip, connector, IK arm on the gripper path.
- `tools/make_harness_video.py` -> out/video/harness.mp4 + docs/media/harness.gif.
- README lead section and writeup §7.

Left: merge to main with --no-ff, push, dispatch CI
(`gh workflow run validation.yml -R ahmedsleem109/RopePhysics --ref main`).
Ideas: closed-loop policy (observe cable), randomize the cable's start pose,
Nsight-profile the learning batch.

### Working notes
- Build from Git Bash: `cmd //c "D:\ropephysics\build.cmd"`; rodsim.exe cannot
  relink while running.
- Python patch scripts via heredoc mangle backslash escapes (newlines in C
  string literals, Windows paths): write those with the Edit/Write tools.
- WSL Ubuntu has g++ (use for float-vs-double builds with `-DCRS_REAL_FLOAT`).
- Pushes: SSH key is not on GitHub; push with
  `git -c credential.helper= -c 'credential.helper=!gh auth git-credential' push https://github.com/ahmedsleem109/RopePhysics.git <branches>`.
  Pushes do not trigger CI (unknown why); dispatch manually.
- Workflow the user asked for: each feature on its own branch, merged to main
  with --no-ff, pushed.

---

## Earlier backlog (still open)

- **GPU:** Nsight profiling (occupancy, bandwidth, latency-bound kernels,
  screenshots into docs/); run the full Phase 1–2 suite on the GPU path;
  colour self contacts per rod (self-collision costs ~11x throughput);
  re-measure GPU headline numbers on a cool GPU (thermal drift seen: 1.16 B →
  0.94 B within a session).
- **Friction creep:** a cable resting on a bar creeps ~2 mm/s even at 2x the
  needed friction (per-particle contact ratcheting). Try segment-based contact
  or a static-friction anchor per contact; it biases hold/slip answers.
- **Validation gaps:** tip-load and twist-buckling mesh sweeps beyond n=32
  (twist buckling converges at slope 0.82 — explain or improve); timestep
  envelope beyond one scenario (contact-driven, whipping); contact torque
  (rolling / torsional friction); segment-based rod–primitive contact.
  Done 2026-09-17: `capstan-mu` (μ 0.1–0.75, worst 0.71%) and
  `timestep-convergence` (first order in the substep, observed 1.00).
- **CI:** find why pushes don't trigger workflows; bump actions to Node 24
  versions (checkout@v5, setup-python@v6, upload-artifact@v5); consider a
  shorter per-commit suite (twist buckling 9 min, capstan 3 min).
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
- [ ] Extend mesh sweeps: cantilever (now n = 256) and moment (now n = 128)
      done; tip-load and twist buckling still at n = 32. Twist buckling converges at slope 0.82 (first
      order) — explain or improve.
- [ ] Stability rule measured on one scenario only (gravity cantilever), and it
      does not hold once elements are shorter than the rod diameter. Test other
      scenarios (contact-driven motion, whipping) and the short-element regime.
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
- [ ] Friction creep (~2 mm/s on a draped cable even at 2× the needed friction).
      Investigate segment-based contact, or a static-friction anchor per contact.
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
