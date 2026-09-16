# GPU Cosserat Rod Simulator

A validated discrete Cosserat rod solver (XPBD, quaternion material frames) with
contact, Coulomb friction and self-collision, a CUDA port designed for batched
environments, and a validation suite that checks every claim against an
independent reference.

Plan and phase gates: [`gpu-cosserat-rod-simulator-plan.md`](gpu-cosserat-rod-simulator-plan.md).
Long-form writeup: [`docs/writeup.md`](docs/writeup.md).

## Status

| phase | state |
|---|---|
| 1 — CPU reference + validation | **done.** Every case passes; convergence is second order where the discretization allows it |
| 2 — contact, friction, self-collision | **done.** Capstan equation reproduced within 1.6% across four wrap angles |
| 3 — CUDA port | **built, not run.** Kernels compile and the coloring is verified, but this machine's driver (576.52, CUDA ≤ 12.9) cannot run the only installed CUDA compiler's runtime (13.1). Parity, determinism and GPU throughput cases are written and report the diagnostic instead of passing |
| 4 — scale and characterization | **done on CPU.** Stability envelope, failure study and batched throughput measured; GPU numbers blocked as above |
| 5 — demo, writeup | **done.** 60 s video rendered offline from CPU simulations with honest timing captions; writeup in `docs/` |

The GPU gap is the one thing standing between this and the plan as written. The
fix is a driver update to r580+ followed by `build\rodsim.exe gpu-parity
gpu-determinism gpu-throughput` — the cases are already there.

---

## Headline results

| claim | measured |
|---|---|
| cantilever tip deflection, mesh convergence | slope **2.23**, finest mesh **0.012%** off Euler–Bernoulli |
| pure end moment, circular arc | slope **2.02** |
| large-deflection elastica, tip position | worst **9.7e-4 L** over `PL²/EI` in [0.5, 5] |
| helix from intrinsic `(κ, τ)` | radius **2.2e-10**, pitch **5.2e-4** relative |
| Michell twist-buckling threshold | **2.3%** at the finest mesh, converging (slope 0.82) |
| rest height on plane / sphere / capsule / box | worst **5.1e-13** relative |
| slip angle on an incline | within **2.0%** of `atan μ`; sliding friction coefficient within **2.2%** |
| capstan `T₂/T₁ = e^{μθ}`, 0.25 to 1 turn | worst **1.6%**, 0.3% at one full turn |
| rope coiling into a pile | up to 13 simultaneous self-contacts, worst overlap **0.056%** of diameter |
| constraint coloring | **2 colours** per constraint family at every resolution, verified conflict-free |
| C++ vs independent NumPy implementation | **4.8e-13 m** after 200 steps |
| stability vs stiffness | stable dt per sweep varies **1.39×** across four decades of Young's modulus |
| what breaks it | motion of **0.66 element lengths per sweep**, 1.11× spread across meshes |
| batched CPU throughput | **23–24 M** segment-substeps/s, 16 threads (11.6× over one thread) |

Full data in `docs/data/*.csv`, figures in `docs/figs/`.

---

## The model

| | count | state |
|---|---|---|
| particles | `N` | centerline position `x_i`, lumped mass |
| segments | `N-1` | material frame `q_j` (unit quaternion), body-frame inertia |

```
stretch / shear   C_s = (x_{i+1} - x_i) / l  -  R(q_j) e3
bend / twist      C_b = (2 / lbar) Im(conj(q_a) q_b)  -  Omega_rest

alpha_s = diag( 1/(ks G A), 1/(ks G A), 1/(E A) ) / l
alpha_b = diag( 1/(E I),    1/(E I),    1/(G J) ) / lbar
alpha~  = alpha / h^2          (h = substep)
```

Both constraints are solved as 3×3 blocks. Jacobians are taken with respect to
body-frame rotation increments, so body-frame inverse inertia is used directly.
With this compliance scaling the static equilibrium contains neither `h` nor
`l`: stiffness is the material's, independent of timestep and resolution.

Contacts are one uniform constraint type (up to four particles with weights,
one normal), unilateral, with position-level Coulomb friction whose cone bounds
the *total* tangential correction per substep. Self-collision uses segment–
segment closest points behind a uniform spatial hash built as key → counting
sort → cell offsets, the same three steps a GPU broadphase runs.

The derivation, the compliance argument and the coloring scheme are written up
in [`docs/writeup.md`](docs/writeup.md).

---

## Layout

```
src/core/math3.h            vec3/mat3/quat, templated: double on the host, float on the device
src/core/rod.{h,cpp}        SoA state, material, constraints, builders, ghost-frame clamps
src/core/solver.{h,cpp}     XPBD substepping, projections, contacts, static relaxation
src/core/collision.{h,cpp}  primitives, contacts, spatial hash, self-collision
src/core/coloring.{h,cpp}   greedy constraint graph coloring, with a verifier
src/gpu/gpu_solver.cu       CUDA kernels: multi-kernel and fused shared-memory strategies
src/gpu/gpu_batch.cpp       host orchestration: layout, upload/download, launch
src/validation/             every validation and characterization case
src/validation/scenes.cpp   demo scenes for the video
tools/reference_prototype.py  independent NumPy implementation (cross-check)
tools/experiments/          exploratory probes (stability_probe -> rodexp)
tools/plot_validation.py    CSV -> figures, light (writeup) or dark (video)
tools/render_scene.py       offline renderer for scene trajectories
tools/make_video.py         assembles the demo video
```

---

## Build and run

Windows, CMake + Ninja + MSVC. `build.cmd` pins the toolchain: MSVC 14.44
(the newer 14.50 in the same install is rejected by nvcc) and CUDA 13.1 when
present. Without nvcc, the CPU reference and its full suite still build.

```
build.cmd
build\rodsim.exe all                    # every case; exit status = failed checks
build\rodsim.exe capstan                # one case
build\rodsim.exe scene drape            # demo scene -> out/scenes
python tools\plot_validation.py         # docs/data -> docs/figs
python tools\plot_validation.py --theme dark
python tools\make_video.py              # out/scenes + figures -> out/video/demo.mp4
python tools\reference_prototype.py     # cross-check against the NumPy mirror
build\rodexp.exe gravity                # exploratory stability probe
```

The full suite takes a long time (tens of minutes). Nearly all of it is the
quadratic Gauss–Seidel budget the static cases need to be genuinely converged —
see the writeup — plus the capstan and twist-buckling sweeps.

---

## What was hard, and what went wrong on the way

These are the findings a reviewer should care about, because each one is a way
to produce a plausible-looking wrong answer. The writeup has the detail.

1. **The clamp must sit where the continuum clamp sits.** Freezing the first
   segment's frame clamps at `s = l/2`, silently turning second-order
   convergence into first order. A ghost frame at `s = 0` fixes it.
2. **Static equilibrium is `h`-independent only once the system is solved.**
   An under-swept rod settles into a state that is genuinely too soft *and
   reports itself converged*. Separately, a too-large substep leaves a
   converged 0.77% bias that no number of sweeps removes.
3. **Buckling is a question for dynamics.** A heavily-iterated XPBD step is
   nearly backward Euler and damps genuinely unstable modes. An early version
   reported a threshold 18% low that turned out to be numerical.
4. **Friction's cap bounds the total, not each sweep's share.** Applying a fresh
   Coulomb cap per sweep gives `iterations` times the friction; a block sat
   motionless on a slope far steeper than `atan μ`.
5. **Self-collision must exclude pairs by rest length, not index.** With
   segments shorter than the diameter, segments two apart are closer than `2r`
   at rest; the solver pushed apart a rope that was not touching itself.
6. **What limits the timestep is motion, not stiffness.** The first theory in
   this repository (predictor displacement `h²a/l`) was inferred, not measured,
   and was wrong. The measured limit is ~0.66 element lengths of motion per
   Gauss–Seidel sweep; iterations tolerate ~2.8× more than substeps.
7. **Several tests passed without testing anything,** and were rewritten: a
   stability sweep whose bisection never left its upper bound, a self-collision
   test with zero contacts, and a capstan fit dragged by creeping points. Each
   case now also asserts that the thing it measures actually happened.

---

## Limitations

- **No GPU numbers.** The CUDA path is written, compiles, and its coloring is
  verified, but it has not executed. GPU/CPU parity, bitwise determinism and
  GPU throughput are unmeasured. There is no Nsight profile.
- **Static results are relaxed, not directly solved,** and converging them costs
  a sweep budget quadratic in the segment count.
- **Mesh sweeps are modest:** n ≤ 64 for the cantilever, ≤ 32 for the moment and
  twist-buckling cases.
- **Energy is characterized, not conserved.** The integrator dissipates; the
  dissipation and any spurious gain only shrink with substepping.
- **Contact against primitives is per particle** (the rod is a chain of spheres
  to the world), accurate while segments are no longer than about their
  diameter. Contacts apply no torque to material frames.
- **The stability rule is measured on one scenario** (gravity cantilever) and
  holds only while elements are longer than the rod's diameter.
- **The capstan rope uses an unphysical density** to keep the dynamics used to
  detect slip stable; the static threshold it measures does not depend on mass.
- **Not built:** the optional Python binding and policy-learning demo.
