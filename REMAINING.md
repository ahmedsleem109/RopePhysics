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
- [x] Fused shared-memory limit on the device: 472 segments.
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
