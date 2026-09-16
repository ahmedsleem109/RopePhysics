# GPU Cosserat Rod Simulator — Project Plan

**Goal:** Build a validated, GPU-accelerated Cosserat rod simulator from scratch on an RTX 3060 laptop, and produce a demo video + technical writeup credible to simulation engineers and research scientists.

**Timeline:** ~16 weeks, 5 phases.

**Positioning note:** The project's primary value is the skill it builds — correct solver design, GPU performance engineering, rigorous validation. Those transfer to any physics-sim, robotics-sim, or graphics-research role. Targeting a specific company is a bonus outcome, not the plan's foundation.

---

## Phase 1 — CPU reference + validation harness (weeks 1–3)

Build it correctly before building it fast. A CPU implementation you trust is what lets you prove the GPU version isn't quietly wrong.

### Implementation
- Discrete Cosserat rod: centerline positions + per-segment orientations as quaternions.
- Material frame via parallel transport for the rest configuration.
- Constraints:
  - Stretch/shear (position–orientation coupling)
  - Bend/twist (Darboux vector from quaternion difference)
- XPBD with substepping.
- **Get the compliance parameterization right** so stiffness is resolution- and timestep-independent. This is where most hobby implementations fail, and it's the first thing an experienced PhysX engineer will check.

### Validation suite (automated, runs on every commit)
- Cantilever tip deflection vs. Euler–Bernoulli small-deflection theory
- Kirchhoff elastica large-deflection curve
- Helical equilibrium under prescribed twist
- Michell buckling threshold (twist instability onset)
- Energy / momentum drift over 10^5 steps, free-flight case

### Output
Error vs. segment count, log-log plots showing convergence order.

### Gate
Convergence plots have the right slope. **Do not proceed until they do.**

---

## Phase 2 — Contact, friction, self-collision (weeks 4–6)

### Implementation
- Rod–rigid contact against analytic primitives (plane, sphere, capsule, box). Skip meshes — they buy nothing for the demo.
- Self-collision: segment–segment closest-point distance, broadphase via uniform spatial hash.
  - Keep the broadphase GPU-friendly from day one (flat arrays, sorted cell keys) so Phase 3 isn't a rewrite.
- Coulomb friction in the XPBD position-level formulation: static and dynamic.

### Validation
- Block-on-incline slip angle vs. μ
- **Capstan equation** for a rod wrapped around a cylinder — exponential tension ratio. A genuine quantitative test of friction *and* contact together.

### Gate
Capstan test matches theory within a few percent across several wrap angles.

---

## Phase 3 — CUDA port and performance engineering (weeks 7–10)

This is the phase that separates you from the field. Budget the most time here.

### Implementation
- **Constraint graph coloring** so independent constraints solve in parallel without races. Visualize the coloring — it makes an excellent diagram.
- SoA memory layout, coalesced access, quaternions in `float4`.
- Fuse the substep loop into as few kernels as possible; measure launch overhead.
- Spatial hash on GPU: key generation → radix sort → cell-start indices.

### Measurement
- Profile with Nsight Compute.
- Record occupancy, achieved bandwidth, and which kernels are bandwidth- vs. latency-bound.
- **Put the profiler screenshots in the writeup.**
- Bitwise-determinism check: same input → same output, every run. Non-deterministic solvers are unshippable and everyone in this field knows it.

### Gate
GPU matches CPU trajectories to solver tolerance across the entire Phase 1–2 validation suite.

---

## Phase 4 — Scale: batched environments (weeks 11–13)

The reframe that makes a 3060 an asset instead of an apology.

### Implementation
- Batch N independent rod environments in one kernel launch, RL-style.
- Report **total rod-steps/second**, not FPS of one pretty scene.

### Characterization
- Scaling curves: throughput vs. batch size, vs. segments per rod, vs. substep count. Find and explain the knee in each curve.
- **Stability envelope:** largest stable timestep vs. stiffness vs. substeps, as a heatmap. This is the plot practitioners actually want.
- One deliberate failure study: push it until it breaks, explain the mechanism, show what fixes it and what it costs.

### Optional (if time allows)
Expose a minimal Python / Warp-style binding and run a trivial policy-learning task (cable reaching a target). Even a weak result proves the framework is usable — that's the robotics-training-adjacent signal.

### Gate
You can state one headline number honestly. E.g. *"X million rod-segment-steps/sec on a laptop 3060."*

---

## Phase 5 — Demo, writeup, launch (weeks 14–16)

Treat this as real work, not an afterthought. Half the project's value lives here.

### Video — 45–60 seconds, no talking head

| Time | Content |
|---|---|
| 0–8s | Cable falls, wraps a post, self-contacts, twists taut. Clean lighting, single accent color, dark background, no UI clutter. |
| 8–20s | Same sim in wireframe with contact points highlighted — shows it's physics, not animation. |
| 20–35s | Grid of hundreds of rods stepping simultaneously. The moment that reads as "GPU." |
| 35–50s | Hard cut to plots: CPU vs. GPU, convergence, stability envelope, capstan validation. ~3s each with a one-line caption. |
| 50–60s | Headline number, repo URL, end. |

Render offline at 60fps, real-time-accurate playback speed, and state the wall-clock timing on screen. **Faking speed is the one unrecoverable credibility error.**

### Writeup (blog post or long README)
- Derivation
- Compliance parameterization
- Constraint coloring scheme
- The profiling journey
- Failure analysis
- **Limitations section** — load-bearing. It's the paragraph that makes a research scientist believe the rest.

### Launch
- Number in the first line of the post. Video autoplays. Repo link. One sentence on what you'd build next.
- Don't tag companies. Don't say you want a job.
- Post the long writeup separately a few days later.

---

## Distribution (runs parallel to Phases 3–5)

A strong project with no distribution reaches nobody. This is a separate job from building.

- **Direct email** to people whose work this is adjacent to. Short: one number, one link. A cold email with a working GPU Cosserat solver attached is a different object entirely.
- **Publish where peers read** — a writeup picked up by graphics/sim people, or a poster at SCA / SIGGRAPH. Internal forwarding beats social-media reach.
- **Adjacent audiences:** NVIDIA Warp / PhysX community, Genesis and MuJoCo contributors, deformable-sim researchers.
- **Contribute upstream:** a good issue or PR against an open-source sim project puts your name in front of practitioners before the project even ships.

---

## Scope discipline

**Cut:**
- Mesh collision
- Rod–rod attachment joints
- Fluid coupling
- Any GUI

Four months of depth on one problem beats eight months of breadth. Depth is the only thing legible to people who already build this for a living.
