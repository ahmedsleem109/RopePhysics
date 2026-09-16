# What remains to build

## ▶ START HERE — next session (written 2026-09-17)

**Goal:** a demo video that a robotics-simulation company (target: Vsim,
Manchester; founded by ex-NVIDIA PhysX engineers; pitch is "AI agents learn
complex tasks in accurately simulated worlds, orders of magnitude faster")
understands in seconds: **a robot arm routing a wire harness, learned in
simulation across thousands of randomized cables on the GPU.** Abstract physics
demos (ropes on a field, cables sliding off bars) did not communicate value; the
video must show a robot doing a real factory job.

### State of the branch `feature/harness-routing` (pushed, NOT merged)

Committed and working:
- `Batch::setMaterialScales(youngsScale, frictionScale)` — per-rod stiffness and
  friction randomization on the GPU, verified against CPU rods built with those
  materials (`gpu-parity`, error 0.7% / 0.1% of the effect).

Committed as work in progress (compiles, last run OK, not validated):
- `src/apps/harness_routing.{h,cpp}` — the task definition `apps::HarnessTask`:
  - board: plane z=0 (mu 0.5), peg A (0.25,0), peg B (0.55,0) capsules r=12 mm
    h=6 cm; clip = two posts r=6 mm at (0.85, ±0.018), h=4 cm;
  - cable 1.2 m, 120 segments, r=4 mm, E_ref=5 MPa; particle 0 pinned at the
    connector (0,0); last particle pinned = gripper, driven by
    `kinematicVelocity`; cable starts straight along -y;
  - params: dt 1 ms, 4 substeps x 4 iterations (float margin ~5x; do not go to
    8+ substeps, see HangingCable::params for the float false-sticking bug);
  - policy = 5 gripper waypoints (x,y) on a fixed schedule `legSeconds`
    {2.4,0.7,0.7,0.8,0.7} + 1 s hold, smoothstep per leg;
  - `evaluate()`: under A / over B by where the cable crosses x=peg.x, within
    pegRadius + r + 1.5 cm ("laid against the peg"); through clip by crossing
    x=clip.x with |y| < inner gap; smooth `score` for learning.
- `rodsim scene harness` (src/validation/scenes.cpp): runs one policy on the CPU,
  writes out/scenes/harness.rodtraj + harness.gripper.csv (gripper path) +
  json, prints the outcome. Replays `out/scenes/harness.policy` ("x y" per
  line) if present, else `handWrittenPolicy()`.
- tools/render_scene.py: camera presets `harness` (oblique) and `harness-top`.

Findings so far:
- The hand-written policy FAILS (good for the story): the cable passes the
  right sides but with a slack loop near the connector, never laid against peg
  A, and 8 mm off the clip centre.
- A tuned policy was written to out/scenes/harness.policy but NOT yet run:
  `0.20 -0.06 / 0.40 0.0 / 0.55 0.07 / 0.75 -0.004 / 1.20 -0.008`
  (pull 35 cm past the clip to take up slack; aim slightly -y through the clip).
  Last build of scenes.cpp had an escaped-newline fix applied just before the
  interruption: rebuild first.

### Build next, in order

1. **Prove a successful routing exists.** Rebuild (`cmd //c "D:\ropephysics\build.cmd"`),
   `build\rodsim.exe scene harness` with the tuned policy, render the last frame
   top-down (`render_scene.default_camera('harness-top', ...)`). Iterate
   waypoints until ROUTED. If pegs are too easy to slip over or the route can't
   be taut, adjust geometry (peg positions, cable length) — keep it plausible
   for a harness board.
2. **GPU evaluation of many (policy, cable) pairs.** In
   `src/validation/application_cases.cpp` (or a new `harness_cases.cpp`):
   one fused batch of N rods (start 1024, then 4096) with the harness world;
   each rod gets a policy + `setMaterialScales` (E x 0.4..4, mu x 0.6..1.4);
   upload gripper velocities with `setKinematicVelocities` only at
   `legStarts()` — BUT gripperVelocity uses smoothstep, which changes every
   step: switch legs to constant velocity (linear) or accept one upload every
   ~20 ms. Evaluate with `downloadPositions` at the end. CPU cross-check a few
   rods (same verdict).
3. **Learning loop (cross-entropy method).** Mean/std over the 10 waypoint
   numbers; each iteration samples N policies, each on a random cable; score =
   `Outcome::score` (success = 3); elites = top 10%; refit; 8-10 iterations.
   Report success rate per iteration and wall time. Then evaluate the final
   mean policy on 1024 fresh random cables → robust success rate. Output
   `docs/data/harness_learning.csv` (iteration, success_rate, mean_score,
   seconds) and final shapes of a 16x16 subset per iteration for the grid view.
   New case `rodsim harness-learning` (skips without CUDA).
4. **Robot arm rendering.** In tools/render_scene.py, draw an arm from the
   gripper path (harness.gripper.csv): base column beside the board, 2-link
   IK in the vertical plane through base and gripper, wrist down to a
   two-finger gripper, as capsules in the painter's sort with the rod. Board as
   a visible slab, pegs/clip coloured, connector box at (0,0). Caption says the
   arm follows the simulated gripper path.
5. **Video** (tools/make_harness_video.py, ~40 s, 1280x720 MP4 + README GIF):
   title "Teaching a robot to route a wire harness" → naive policy close-up
   (fails, red label) → top-down grid of 256 boards per iteration, green/red,
   "iteration k: X% of N attempts routed, T s on a laptop GPU" → learned
   policy on a soft and a stiff cable close-up (succeeds) → end card with
   numbers and "physics validated against beam theory, capstan equation, CPU
   reference". Send the MP4 to the user.
6. **README:** replace the cable-hanging lead section with this story (keep
   cable-hanging lower down as the validation of friction). Writeup section.
7. Merge `feature/harness-routing` to main, push, dispatch CI
   (`gh workflow run validation.yml -R ahmedsleem109/RopePhysics --ref main`).

### Working notes that cost time this session
- Build from Git Bash: `cmd //c "D:\ropephysics\build.cmd"`; rodsim.exe cannot
  relink while running.
- Python patch scripts via heredoc mangle `\n` inside C string literals — write
  C++ edits with the Edit/Write tools, not heredoc'd Python.
- WSL Ubuntu has g++ (use for float-vs-double builds with `-DCRS_REAL_FLOAT`);
  run WSL scripts via PowerShell `wsl -d Ubuntu -- bash /mnt/c/...script.sh`.
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
- **Validation gaps:** timestep-refinement convergence study; tip-load and
  twist-buckling mesh sweeps beyond n=32 (twist buckling converges at slope
  0.82 — explain or improve); timestep envelope beyond one scenario
  (contact-driven, whipping); capstan mu sweep; contact torque (rolling /
  torsional friction); segment-based rod–primitive contact.
- **CI:** find why pushes don't trigger workflows; bump actions to Node 24
  versions (checkout@v5, setup-python@v6, upload-artifact@v5); consider a
  shorter per-commit suite (twist buckling 9 min, capstan 3 min).
- **Media:** re-render the long demo video (tools/make_video.py) with the fixed
  solver, GPU numbers and repo link
  (`--repo "https://github.com/ahmedsleem109/RopePhysics"`); host demo.mp4 as a
  GitHub release asset.
- **Optional:** Python binding; policy-learning task (the harness learning loop
  above covers this).

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
- [ ] Timestep-refinement convergence study (only mesh refinement is done).
- [ ] Extend mesh sweeps: cantilever (now n = 256) and moment (now n = 128)
      done; tip-load and twist buckling still at n = 32. Twist buckling converges at slope 0.82 (first
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

## 5b. Applications

- [x] Cable hanging: GPU sweep of friction × placement × stiffness, capstan
      boundary on the safe side, CPU/GPU agreement, demo scene and video.
- [x] Kinematic (moving) pinned particles on CPU and GPU, for grippers.
- [ ] Friction creep (~2 mm/s on a draped cable even at 2× the needed friction).
      Investigate segment-based contact, or a static-friction anchor per contact.
- [ ] Per-rod friction, so a friction sweep is one batch instead of one per value.
- [ ] A manipulation demo that uses the moving gripper (cable routing around pegs).

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
