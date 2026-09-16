"""Turn the CSVs written by `rodsim` into the validation figures.

    python tools/plot_validation.py [--data docs/data] [--figs docs/figs] [--theme light|dark]

Light figures go in the writeup; dark ones (default output out/figs_dark) are
the slides in the demo video, drawn on the video's own background.

Every figure names what it is compared against, because a convergence plot
without a named reference is decoration. Colours come from a palette validated
for colour-vision deficiency in both themes (worst adjacent CVD dE 9.1 light /
8.4 dark); where a series falls under 3:1 contrast it is direct-labelled rather
than left to its colour.
"""

from __future__ import annotations

import argparse
import csv
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "text2": "#52514e",
        "grid": "#e4e3df",
        "guide": "#8a8985",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
        "seq": "Blues",
    },
    "dark": {
        "surface": "#0e1014",  # the demo video's background
        "text": "#ffffff",
        "text2": "#c3c2b7",
        "grid": "#2a2c30",
        "guide": "#8c8b85",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500"],
        "seq": "Blues_r",
    },
}
T = THEMES["light"]


def read(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for k in rows[0] if rows else []:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[k]))
            except ValueError:
                vals.append(r[k])
        out[k] = vals
    return out


def new_fig(w=5.6, h=3.7, ncols=1):
    fig, axes = plt.subplots(1, ncols, figsize=(w, h), dpi=170)
    fig.patch.set_facecolor(T["surface"])
    for ax in np.atleast_1d(axes):
        ax.set_facecolor(T["surface"])
    return fig, axes


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=10, loc="left", color=T["text"])
    ax.set_xlabel(xlabel, fontsize=9, color=T["text2"])
    ax.set_ylabel(ylabel, fontsize=9, color=T["text2"])
    ax.grid(True, which="major", color=T["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(which="both", labelsize=8, colors=T["text2"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(T["grid"])


def legend(ax):
    leg = ax.legend(fontsize=8, frameon=False)
    for text in leg.get_texts():
        text.set_color(T["text2"])


def guide(ax, xs, ys, order):
    """Reference slope through the finest point."""
    x0, y0 = xs[-1], ys[-1]
    gx = [min(xs), max(xs)]
    gy = [y0 * (x / x0) ** order for x in gx]
    ax.plot(gx, gy, "--", color=T["guide"], linewidth=1.0, zorder=1)
    ax.annotate(f"$O(h^{order})$", (gx[1], gy[1]), color=T["guide"], fontsize=8,
                xytext=(-28, 4), textcoords="offset points")


def save(fig, figs, name):
    fig.tight_layout()
    fig.savefig(os.path.join(figs, name), facecolor=T["surface"])
    plt.close(fig)


# ----------------------------------------------------------------- phase 1

def plot_solver(data, figs):
    d = read(os.path.join(data, "solver_convergence.csv"))
    fig, ax = new_fig()
    for i, sub in enumerate(sorted(set(d["substeps"]))):
        xs = [b for b, s in zip(d["iterations"], d["substeps"]) if s == sub]
        ys = [t for t, s in zip(d["tip_deflection"], d["substeps"]) if s == sub]
        ax.semilogx(xs, ys, "o-", color=T["series"][i], linewidth=2, markersize=6,
                    label=f"{int(sub)} substep{'s' if sub != 1 else ''}")
    if "direct_tip" in d:
        ax.axhline(d["direct_tip"][0], color=T["guide"], linewidth=1.5, linestyle="--",
                   label="direct static solve")
    style(ax, "Static equilibrium vs solver budget (n = 32)",
          "Gauss-Seidel sweeps per substep", "tip deflection [m]")
    legend(ax)
    save(fig, figs, "solver_convergence.png")


def plot_cantilever(data, figs):
    d = read(os.path.join(data, "cantilever_convergence.csv"))
    fig, ax = new_fig()
    ax.loglog(d["h"], d["rel_err_tim"], "o-", color=T["series"][0], linewidth=2, markersize=6,
              label="vs Timoshenko (shearable)")
    ax.loglog(d["h"], d["rel_err_eb"], "s-", color=T["series"][1], linewidth=2, markersize=5,
              label="vs Euler-Bernoulli")
    guide(ax, d["h"], d["rel_err_tim"], 2)
    style(ax, "Cantilever tip deflection vs mesh refinement",
          "element length $h$ [m]", "relative error")
    legend(ax)
    save(fig, figs, "cantilever_convergence.png")


def plot_moment(data, figs):
    d = read(os.path.join(data, "elastica_moment.csv"))
    fig, ax = new_fig()
    ax.loglog(d["h"], d["rel_err"], "o-", color=T["series"][0], linewidth=2, markersize=6)
    guide(ax, d["h"], d["rel_err"], 2)
    style(ax, "Pure end moment: arc radius vs $EI/M$", "element length $h$ [m]",
          "relative error in radius")
    save(fig, figs, "elastica_moment.png")


def plot_tipload(data, figs):
    d = read(os.path.join(data, "elastica_tipload.csv"))
    fig, (ax0, ax1) = new_fig(9.0, 3.7, 2)
    ax0.plot(d["ref_x"], d["ref_z"], "-", color=T["guide"], linewidth=2, label="exact elastica")
    ax0.plot(d["tip_x"], d["tip_z"], "o", color=T["series"][0], markersize=7, label="solver")
    ax0.set_aspect("equal")
    style(ax0, "Tip locus under increasing load", "$x/L$", "$z/L$")
    legend(ax0)
    ax1.semilogy(d["alpha"], d["err_dist"], "o-", color=T["series"][0], linewidth=2, markersize=6)
    style(ax1, "Tip position error", r"$\alpha = P L^2 / EI$", "error / $L$")
    save(fig, figs, "elastica_tipload.png")


def plot_energy(data, figs):
    d = read(os.path.join(data, "energy_drift.csv"))
    fig, (ax0, ax1) = new_fig(9.0, 3.7, 2)
    ax0.plot(d["time"], d["kinetic"], color=T["series"][0], linewidth=1, label="kinetic")
    ax0.plot(d["time"], d["elastic"], color=T["series"][1], linewidth=1, label="elastic")
    ax0.plot(d["time"], d["total"], color=T["text"], linewidth=1.5, label="total")
    style(ax0, "Free flight energy exchange (16 substeps)", "time [s]", "energy [J]")
    legend(ax0)
    ax1.plot(d["time"], d["rel_energy_drift"], color=T["series"][0], linewidth=1.5)
    style(ax1, "Energy lost, relative to $E_0$", "time [s]", r"$|E - E_0| / E_0$")
    save(fig, figs, "energy_drift.png")


def plot_dissipation(data, figs):
    d = read(os.path.join(data, "energy_dissipation.csv"))
    fig, ax = new_fig()
    ax.loglog(d["substeps"], d["dissipation"], "o-", color=T["series"][0], linewidth=2,
              markersize=6, label="dissipated in first 20 s")
    gains = [max(g, 1e-7) for g in d["max_energy_gain"]]
    ax.loglog(d["substeps"], gains, "s-", color=T["series"][1], linewidth=2, markersize=5,
              label="spuriously gained (floored at 1e-7)")
    style(ax, "Energy error vs substepping (free flight)", "substeps per 1 ms step",
          "fraction of initial energy")
    legend(ax)
    save(fig, figs, "energy_dissipation.png")


def plot_twist(data, figs):
    d = read(os.path.join(data, "twist_buckling.csv"))
    fig, (ax0, ax1) = new_fig(9.0, 3.7, 2)
    ref = d["phi_crit_ref"][0]
    ax0.plot(d["h"], d["phi_crit"], "o-", color=T["series"][0], linewidth=2, markersize=7,
             label="measured")
    ax0.axhline(ref, color=T["guide"], linestyle="--", linewidth=1)
    ax0.annotate(f"clamped-clamped theory {ref:.3f} rad", (max(d["h"]), ref), color=T["guide"],
                 fontsize=8, xytext=(-150, 6), textcoords="offset points")
    ax0.set_xlim(0, max(d["h"]) * 1.1)
    style(ax0, "Twist-buckling threshold vs mesh", "element length $h$ [m]",
          r"critical end rotation $\Phi$ [rad]")
    ax1.loglog(d["h"], d["rel_err"], "o-", color=T["series"][0], linewidth=2, markersize=6)
    guide(ax1, d["h"], d["rel_err"], 1)
    style(ax1, "Threshold error", "element length $h$ [m]", "relative error")
    save(fig, figs, "twist_buckling.png")


# ----------------------------------------------------------------- phase 2

def sliding_threshold(xs, speeds, frac=0.05):
    """x-intercept of the linear sliding regime, same rule as the C++ case."""
    xs, speeds = np.asarray(xs), np.asarray(speeds)
    keep = speeds > frac * speeds.max()
    a, b = np.polyfit(xs[keep], speeds[keep], 1)
    return -b / a


def plot_capstan(data, figs):
    d = read(os.path.join(data, "capstan.csv"))
    turns = sorted(set(d["wrap_turns"]))
    theta_ref = np.linspace(0, 2 * math.pi * 1.05, 200)
    mu = None
    fig, (ax0, ax1) = new_fig(9.0, 3.8, 2)

    measured = []
    for i, t in enumerate(turns):
        rows = [k for k, w in enumerate(d["wrap_turns"]) if w == t]
        ratio = np.array([d["ratio"][k] for k in rows])
        speed = np.array([d["terminal_speed"][k] for k in rows])
        theta = d["theta"][rows[0]]
        cap = d["capstan_ratio"][rows[0]]
        mu = math.log(cap) / theta
        crit = sliding_threshold(ratio, speed)
        measured.append((theta, crit))
        ax1.plot(ratio / cap, speed * 1e3, "o-", color=T["series"][i], linewidth=1.8,
                 markersize=5, label=f"{t:g} turn{'s' if t != 1 else ''}")

    ax0.plot(theta_ref, np.exp(mu * theta_ref), "-", color=T["guide"], linewidth=2,
             label=fr"$e^{{\mu\theta}}$, $\mu$ = {mu:.2f}")
    th, cr = zip(*measured)
    ax0.plot(th, cr, "o", color=T["series"][0], markersize=8, label="measured slip threshold")
    style(ax0, "Capstan: tension ratio at slip", r"wrap angle $\theta$ [rad]", r"$T_2 / T_1$")
    legend(ax0)

    ax1.axvline(1.0, color=T["guide"], linestyle="--", linewidth=1)
    style(ax1, "Slip speed, pull normalised by $e^{\\mu\\theta}$", r"$(T_2/T_1)\,/\,e^{\mu\theta}$",
          "terminal slip speed [mm/s]")
    legend(ax1)
    save(fig, figs, "capstan.png")


def plot_incline(data, figs):
    d = read(os.path.join(data, "incline.csv"))
    mus = sorted(set(d["mu"]))
    fig, (ax0, ax1) = new_fig(9.0, 3.8, 2)
    grid = np.linspace(0, 1.0, 100)
    ax0.plot(grid, np.degrees(np.arctan(grid)), "-", color=T["guide"], linewidth=2,
             label=r"$\arctan\mu$")
    found = []
    for i, mu in enumerate(mus):
        rows = [k for k, m in enumerate(d["mu"]) if m == mu]
        alpha = np.array([d["alpha"][k] for k in rows])
        accel = np.array([d["accel"][k] for k in rows])
        theory = np.array([d["accel_theory"][k] for k in rows])
        keep = accel > 0.2
        if keep.sum() >= 2:
            a, b = np.polyfit(alpha[keep], accel[keep], 1)
            found.append((mu, math.degrees(-b / a)))
        ax1.plot(np.degrees(alpha), accel, "o", color=T["series"][i], markersize=5,
                 label=fr"$\mu$ = {mu:g}")
        ax1.plot(np.degrees(alpha), theory, "-", color=T["series"][i], linewidth=1.2, alpha=0.7)
    ms, angs = zip(*found)
    ax0.plot(ms, angs, "o", color=T["series"][0], markersize=8, label="measured slip angle")
    style(ax0, "Slip angle vs friction coefficient", r"$\mu$", "slip angle [deg]")
    legend(ax0)
    style(ax1, "Sliding acceleration (lines: $g(\\sin\\alpha-\\mu\\cos\\alpha)$)",
          "incline angle [deg]", r"acceleration [m/s$^2$]")
    legend(ax1)
    save(fig, figs, "incline.png")


def plot_self_collision(data, figs):
    d = read(os.path.join(data, "self_collision.csv"))
    fig, (ax0, ax1) = new_fig(9.0, 3.6, 2)
    ratio = np.array(d["min_distance"]) / np.array(d["diameter"])
    ax0.plot(d["time"], ratio, "-", color=T["series"][0], linewidth=1.8)
    ax0.axhline(1.0, color=T["guide"], linestyle="--", linewidth=1)
    style(ax0, "Closest approach between strands", "time [s]", "min distance / rope diameter")
    ax1.plot(d["time"], d["self_contacts"], "-", color=T["series"][0], linewidth=1.5)
    style(ax1, "Active self-contacts", "time [s]", "contacts")
    save(fig, figs, "self_collision.png")


# ----------------------------------------------------------------- phase 4

def plot_throughput(data, figs):
    d = read(os.path.join(data, "throughput_cpu.csv"))
    rows = list(zip(d["threads"], d["rods"], d["segments"], d["substeps"],
                    d["segment_steps_per_sec"]))
    maxthreads = max(d["threads"])
    fig, axes = new_fig(11.0, 3.2, 4)
    panels = [
        ("threads", lambda r: r[1] == 256 and r[2] == 64 and r[3] == 8, 0, "threads"),
        ("rods", lambda r: r[0] == maxthreads and r[2] == 64 and r[3] == 8, 1, "rods in batch"),
        ("segments", lambda r: r[0] == maxthreads and r[1] == 256 and r[3] == 8, 2,
         "segments per rod"),
        ("substeps", lambda r: r[0] == maxthreads and r[1] == 256 and r[2] == 64, 3, "substeps"),
    ]
    for ax, (title, pick, col, xlabel) in zip(axes, panels):
        # The baseline configuration appears in every sweep and was measured
        # each time; take the median of repeats rather than plotting a stack.
        by_x = {}
        for r in rows:
            if pick(r):
                by_x.setdefault(r[col], []).append(r[4])
        if by_x:
            xs = sorted(by_x)
            ys = [float(np.median(by_x[x])) / 1e6 for x in xs]
            ax.semilogx(xs, ys, "o-", color=T["series"][0], linewidth=2, markersize=6, base=2)
        # Zero-based: flatness is the finding in two of these panels, and a
        # zoomed axis would dress ~7% measurement noise up as structure.
        ax.set_ylim(0, max(d["segment_steps_per_sec"]) / 1e6 * 1.12)
        style(ax, f"vs {title}", xlabel, "M segment-substeps / s")
    save(fig, figs, "throughput_cpu.png")


def plot_throughput_gpu(data, figs):
    # CPU vs GPU on the same workload: 64-segment rods, 8 substeps, one sweep,
    # batch size swept. CPU at its full thread count, median of repeats.
    g = read(os.path.join(data, "gpu_throughput.csv"))
    if "primitives" in g:  # contact-workload rows are reported, not plotted here
        keep = [i for i, p in enumerate(g["primitives"]) if p == 0]
        g = {k: [v[i] for i in keep] for k, v in g.items()}
    fig, (ax0, ax1) = new_fig(9.0, 3.7, 2)
    series = [("fused", "GPU fused (1 launch/step)", 0), ("multikernel", "GPU multi-kernel", 1)]
    for strat, label, k in series:
        pts = sorted((r, v / 1e6) for s, r, n, sub, v in
                     zip(g["strategy"], g["rods"], g["segments"], g["substeps"],
                         g["segment_steps_per_sec"])
                     if s == strat and n == 64 and sub == 8 and r != 2048)
        ax0.loglog(*zip(*pts), "o-", color=T["series"][k], linewidth=2, markersize=6, base=2,
                   label=label)
    cpu_path = os.path.join(data, "throughput_cpu.csv")
    if os.path.exists(cpu_path):
        c = read(cpu_path)
        top = max(c["threads"])
        by_rods = {}
        for t, r, n, sub, v in zip(c["threads"], c["rods"], c["segments"], c["substeps"],
                                   c["segment_steps_per_sec"]):
            if t == top and n == 64 and sub == 8:
                by_rods.setdefault(r, []).append(v / 1e6)
        xs = sorted(by_rods)
        ax0.loglog(xs, [float(np.median(by_rods[x])) for x in xs], "s-", color=T["series"][2],
                   linewidth=2, markersize=5, base=2, label=f"CPU, {int(top)} threads")
    ax0.set_yscale("log", base=10)
    style(ax0, "Throughput vs batch size (64 segments, 8 substeps)", "rods in batch",
          "M segment-substeps / s")
    legend(ax0)

    for strat, label, k in series:
        pts = sorted((n, v / 1e6) for s, r, n, sub, v in
                     zip(g["strategy"], g["rods"], g["segments"], g["substeps"],
                         g["segment_steps_per_sec"])
                     if s == strat and r == 2048 and sub == 8)
        ax1.semilogx(*zip(*pts), "o-", color=T["series"][k], linewidth=2, markersize=6, base=2,
                     label=label)
    ax1.set_ylim(bottom=0)
    style(ax1, "GPU vs rod length (2048 rods)", "segments per rod", "M segment-substeps / s")
    save(fig, figs, "throughput_gpu.png")


def annotate_heatmap(ax, grid):
    """Write each cell's value, with ink that flips on cell lightness so every
    value stays readable against the sequential ramp in either theme."""
    logs = np.log10(grid)
    lo, hi = np.nanmin(logs), np.nanmax(logs)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if not np.isfinite(grid[i, j]):
                continue
            dark_cell = (logs[i, j] - lo) / (hi - lo + 1e-9) > 0.55
            if T is THEMES["dark"]:
                dark_cell = not dark_cell
            ax.text(j, i, f"{grid[i, j]:.2g}", ha="center", va="center", fontsize=7,
                    color="#ffffff" if dark_cell else "#0b0b0b")


def plot_timestep_envelope(data, figs):
    d = read(os.path.join(data, "timestep_envelope.csv"))
    rows = list(zip(d["youngs"], d["segments"], d["iterations"], d["max_substep"],
                    d["max_dt_per_sweep"], d["elements_per_substep"]))
    stiffness = sorted(set(d["youngs"]))
    fig, (ax0, ax1) = new_fig(9.0, 3.7, 2)
    for k, E in enumerate(stiffness):
        pts = sorted((n, h * 1e3) for e, n, it, h, _, _ in rows if e == E and it == 1)
        ax0.loglog(*zip(*pts), "o-", color=T["series"][k], linewidth=2, markersize=6, base=2,
                   label=f"E = {E:.0e} Pa")
    ax0.set_yscale("log", base=10)
    style(ax0, "Largest substep keeping strain under 1%", "segments", "substep h [ms]")
    legend(ax0)
    for k, E in enumerate(stiffness):
        pts = sorted((n, m) for e, n, it, _, _, m in rows if e == E and it == 1)
        ax1.semilogx(*zip(*pts), "o-", color=T["series"][k], linewidth=2, markersize=6, base=2,
                     label=f"E = {E:.0e} Pa")
    ax1.set_ylim(0, 0.05)
    style(ax1, "At the limit: motion per substep", "segments", "element lengths moved per substep")
    save(fig, figs, "timestep_envelope.png")


# ----------------------------------------------------------------- applications

def plot_cable_hanging(data, figs):
    d = read(os.path.join(data, "cable_hanging.csv"))
    stiffness = sorted(set(d["youngs"]))
    fig, axes = new_fig(10.0, 4.0, len(stiffness))
    for ax, E in zip(np.atleast_1d(axes), stiffness):
        mus = sorted(set(m for m, e in zip(d["friction"], d["youngs"]) if e == E))
        ratios = sorted(set(r for r, e in zip(d["leg_ratio"], d["youngs"]) if e == E))
        grid = np.zeros((len(mus), len(ratios)))
        for e, m, r, h in zip(d["youngs"], d["friction"], d["leg_ratio"], d["held"]):
            if e == E:
                grid[mus.index(m), ratios.index(r)] = h
        ax.pcolormesh(ratios, mus, grid, cmap=matplotlib.colors.ListedColormap(
            [T["series"][1], T["series"][2]]), shading="nearest", vmin=0, vmax=1)
        rr = np.linspace(min(ratios), max(ratios), 200)
        ax.plot(rr, np.log(rr) / math.pi, "--", color=T["text"], linewidth=1.8,
                label=r"ideal rope: $\mu = \ln(\mathrm{long/short})/\pi$")
        ax.set_ylim(min(mus), max(mus))
        style(ax, f"E = {E / 1e6:.0f} MPa: green holds, orange slides off",
              "placement: long leg / short leg", "friction coefficient $\mu$")
        legend(ax)
    save(fig, figs, "cable_hanging.png")


PLOTS = {
    "solver_convergence.csv": plot_solver,
    "cantilever_convergence.csv": plot_cantilever,
    "elastica_moment.csv": plot_moment,
    "elastica_tipload.csv": plot_tipload,
    "energy_drift.csv": plot_energy,
    "energy_dissipation.csv": plot_dissipation,
    "twist_buckling.csv": plot_twist,
    "capstan.csv": plot_capstan,
    "incline.csv": plot_incline,
    "self_collision.csv": plot_self_collision,
    "throughput_cpu.csv": plot_throughput,
    "cable_hanging.csv": plot_cable_hanging,
    "gpu_throughput.csv": plot_throughput_gpu,
    "timestep_envelope.csv": plot_timestep_envelope,
}


def main():
    global T
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="docs/data")
    ap.add_argument("--figs", default=None)
    ap.add_argument("--theme", choices=("light", "dark"), default="light")
    args = ap.parse_args()
    T = THEMES[args.theme]
    figs = args.figs or ("docs/figs" if args.theme == "light" else "out/figs_dark")
    os.makedirs(figs, exist_ok=True)

    for name, fn in PLOTS.items():
        path = os.path.join(args.data, name)
        if not os.path.exists(path):
            print(f"skip {name} (not found)")
            continue
        try:
            fn(args.data, figs)
            print(f"wrote {name.replace('.csv', '.png')}")
        except Exception as exc:  # one bad CSV must not sink the rest
            print(f"FAILED {name}: {exc}")


if __name__ == "__main__":
    main()
