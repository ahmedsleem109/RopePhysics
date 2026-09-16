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
| 1 — CPU reference + validation | **done.** Every physics case passes; convergence is second order where the discretization allows it. Static cases are solved directly and run in about a second |
| 2 — contact, friction, self-collision | **done.** Capstan equation reproduced within 1.6% across four wrap angles |
| 3 — CUDA port | **running.** On an RTX 3060 Laptop GPU (driver 616.92) both strategies match the colour-ordered CPU to float rounding, agree with each other bit for bit, and are bitwise deterministic across runs. Nsight profiling and GPU contacts are still open |
| 4 — scale and characterization | **reopened.** Batched CPU throughput measured. The stability envelope and failure study were invalidated by the stretch-constraint frame fix and fail until redesigned; GPU numbers blocked as above |
| 5 — demo, writeup | **done.** 60 s video rendered offline from CPU simulations with honest timing captions; writeup in `docs/` |

Still open against the plan: Nsight profiling, contacts and self-collision on
the GPU, the redesigned stability study, and a re-render of the demo video.

---

## Headline results

| claim | measured |
|---|---|
| cantilever tip deflection, mesh convergence | slope **2.00** to n = 256, **7.6e-6** off Timoshenko; identical on an oblique axis to 2e-12 |
| pure end moment, circular arc | slope **2.01** to n = 128, radius **1.0e-4** off |
| large-deflection elastica, tip position | worst **6.3e-4 L** over `PL²/EI` in [0.5, 5] |
| helix from intrinsic `(κ, τ)` | radius **2.2e-10**, pitch **5.2e-4** relative |
| Michell twist-buckling threshold | **2.3%** at the finest mesh, converging (slope 0.82) |
| rest height on plane / sphere / capsule / box | worst **2.2e-16** relative |
| slip angle on an incline | within **1.5%** of `atan μ`; sliding friction coefficient within **1.7%** |
| capstan `T₂/T₁ = e^{μθ}`, 0.25 to 1 turn | worst **1.6%**, 0.3% at one full turn |
| rope coiling into a pile | up to 13 simultaneous self-contacts, worst overlap **0.18%** of diameter |
| constraint coloring | **2 colours** per constraint family at every resolution, verified conflict-free |
| C++ vs independent NumPy implementation | **4.7e-13 m** after 200 steps |
| batched CPU throughput | **23–25 M** segment-substeps/s, 16 threads (11.6× over one thread) |
| **GPU throughput, laptop RTX 3060** | **1.17 B** segment-substeps/s (16 384 rods × 64 segments × 8 substeps, fused); **1.86 B** at 2048 × 256 segments; **47×** the 16-thread CPU on the same workload |
| GPU vs CPU parity | **2e-8 m** after one step, growing exactly as a float-vs-double CPU build does (2.5e-4 m at 200 steps); multi-kernel and fused **bitwise identical** |

Full data in `docs/data/*.csv`, figures in `docs/figs/`.

> **Stability claims withdrawn (2026-09-16).** The stability-envelope and
> failure-study results below were measured with a stretch/shear constraint that
> applied its compliance in world instead of material axes. After the fix, stable
> timesteps rose 5–20× and blow-up is no longer monotone in `dt`; both old and new
> runs near the limit carry constraint strains of 50–250%, so an energy-bounded
> criterion was never measuring usable accuracy. Those two cases currently fail
> and are being redesigned. See `REMAINING.md`.

---

## The model

| | count | state |
|---|---|---|
| particles | `N` | centerline position `x_i`, lumped mass |
| segments | `N-1` | material frame `q_j` (unit quaternion), body-frame inertia |

```
stretch / shear   C_s = R(q_j)^T (x_{i+1} - x_i) / l  -  e3      (material frame)
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
src/core/statics.{h,cpp}    direct static equilibrium: banded Newton with load continuation
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
.github/workflows/          CI: builds and runs the validation suite on every push
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

The full suite takes about 17 minutes, almost all of it twist buckling (9 min,
honest dynamics at 256 substeps) and the capstan sweep (3 min). The static
cases are solved directly (`src/core/statics.cpp`) and take about a second
together; under the old XPBD relaxation they took over 11 minutes.
`.github/workflows/validation.yml` runs the suite on every push.

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
6. **A constraint's compliance must live in the frame it is written in.** The
   stretch/shear constraint measured strain in world axes, so a rod along `x`
   carried `EA` in shear. Every case passed. Refining the cantilever to
   n = 256, affordable only once statics were solved directly, showed it
   converging to a third of the Timoshenko shear deflection. The same fix
   invalidated this repository's stability findings (see the note above).
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
