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


def plot_stability(data, figs):
    d = read(os.path.join(data, "stability_envelope.csv"))
    if "sweeps" not in d:
        return
    rows = list(zip(d["youngs"], d["substeps"], d["iterations"], d["max_dt"]))
    es = sorted({r[0] for r in rows if r[2] == 1})
    subs = sorted({r[1] for r in rows if r[2] == 1})
    grid = np.full((len(es), len(subs)), np.nan)
    for e, s, it, dt in rows:
        if it == 1:
            grid[es.index(e), subs.index(s)] = dt

    fig, (ax0, ax1) = new_fig(10.0, 3.9, 2)
    ax0.imshow(np.log10(grid), cmap=T["seq"], aspect="auto", origin="lower")
    ax0.set_xticks(range(len(subs)), [str(int(s)) for s in subs])
    ax0.set_yticks(range(len(es)), [f"1e{int(round(math.log10(e)))}" for e in es])
    annotate_heatmap(ax0, grid)
    style(ax0, "Largest bounded dt [s]: flat in stiffness", "substeps", "Young's modulus [Pa]")
    ax0.grid(False)

    base = sorted((r[1], r[3]) for r in rows if r[0] == 1e7 and r[2] == 1)
    iters = sorted((r[2], r[3]) for r in rows if r[0] == 1e7 and r[1] == 1)
    ax1.loglog(*zip(*base), "o-", color=T["series"][0], linewidth=2, markersize=6,
               label="sweeps spent as substeps")
    ax1.loglog(*zip(*iters), "s-", color=T["series"][1], linewidth=2, markersize=5,
               label="sweeps spent as iterations")
    style(ax1, "The limit tracks total sweeps per step", "Gauss-Seidel sweeps per step",
          "largest bounded dt [s]")
    legend(ax1)
    save(fig, figs, "stability_envelope.png")


def plot_failure(data, figs):
    d = read(os.path.join(data, "failure_study.csv"))
    fig, ax = new_fig()
    splits = sorted({(s, i) for s, i in zip(d["substeps"], d["iterations"])})
    for k, (s, i) in enumerate(splits):
        pts = sorted((n, e) for n, e, ss, ii in zip(d["segments"], d["elements_per_sweep"],
                                                    d["substeps"], d["iterations"])
                     if ss == s and ii == i)
        if pts:
            ax.semilogx(*zip(*pts), "o-", color=T["series"][k], linewidth=2, markersize=6,
                        base=2, label=f"{int(s)} substep{'s' if s != 1 else ''}, "
                                      f"{int(i)} iteration{'s' if i != 1 else ''}")
    ax.set_ylim(bottom=0)
    style(ax, "At breakdown: motion per sweep", "segments", "element lengths moved per sweep")
    legend(ax)
    save(fig, figs, "failure_study.png")


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
    "stability_envelope.csv": plot_stability,
    "failure_study.csv": plot_failure,
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
