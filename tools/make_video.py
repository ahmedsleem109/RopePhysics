"""Assemble the Phase 5 demo video from scene trajectories and validation figures.

    python tools/make_video.py [--scenes out/scenes] [--figs out/figs_dark]
                               [--out out/video/demo.mp4] [--repo "<link>"]

Structure (60 fps, 1280x720):

    0-10 s   drape, shaded       cable drapes over a post, one end wound until it supercoils
   10-20 s   drape, wireframe    same simulation, contact points shown
   20-35 s   grid                256 independent rods stepping together
   35-50 s   figures             ~3 s each, one caption line
   50-60 s   headline            the measured number, stated with its hardware

Every simulated clip is played back at real-time-accurate speed, and its
caption states what the simulation cost in wall-clock time. The simulations
were run on the CPU: the CUDA kernels in src/gpu compile on this machine but
cannot execute under its installed driver, and the video says so instead of
implying otherwise.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(__file__))
import render_scene as rs  # noqa: E402

FPS = 60


def card(lines, sub=None, big=None):
    img = Image.new("RGB", (rs.WIDTH, rs.HEIGHT), rs.BACKGROUND)
    d = ImageDraw.Draw(img)
    y = rs.HEIGHT // 2 - 120
    if big:
        f = rs.font(84, bold=True)
        w = d.textlength(big, font=f)
        d.text(((rs.WIDTH - w) / 2, y), big, font=f, fill=rs.ACCENT)
        y += 120
    for i, line in enumerate(lines):
        f = rs.font(34 if i == 0 else 24, bold=(i == 0))
        w = d.textlength(line, font=f)
        d.text(((rs.WIDTH - w) / 2, y), line, font=f, fill=rs.TEXT if i == 0 else rs.TEXT_DIM)
        y += 52 if i == 0 else 38
    if sub:
        f = rs.font(20)
        w = d.textlength(sub, font=f)
        d.text(((rs.WIDTH - w) / 2, rs.HEIGHT - 70), sub, font=f, fill=rs.TEXT_DIM)
    return img


def figure_slide(path, caption):
    img = Image.new("RGB", (rs.WIDTH, rs.HEIGHT), rs.BACKGROUND)
    fig = Image.open(path).convert("RGB")
    max_w, max_h = rs.WIDTH - 120, rs.HEIGHT - 150
    scale = min(max_w / fig.width, max_h / fig.height)
    fig = fig.resize((int(fig.width * scale), int(fig.height * scale)), Image.LANCZOS)
    # Centre in the space above the caption, so wide two-panel figures do not
    # sit high with a gap beneath them.
    img.paste(fig, ((rs.WIDTH - fig.width) // 2, max(30, (rs.HEIGHT - 110 - fig.height) // 2)))
    d = ImageDraw.Draw(img)
    f = rs.font(26)
    w = d.textlength(caption, font=f)
    d.text(((rs.WIDTH - w) / 2, rs.HEIGHT - 80), caption, font=f, fill=rs.TEXT)
    return img


def hold(writer, img, seconds):
    frame = np.asarray(img)
    for _ in range(int(seconds * FPS)):
        writer.append_data(frame)


def clip(writer, traj, name, mode, title, frames, camera_frames=None):
    caption = rs.timing_caption(traj)
    total = len(traj)
    for k, f in enumerate(frames):
        cam = rs.default_camera(name, f if camera_frames is None else camera_frames[k], total)
        writer.append_data(np.asarray(rs.render_frame(traj, f, cam, mode, caption, title)))
        if k % 60 == 0:
            print(f"  {name}/{mode}: frame {k}/{len(frames)}", flush=True)


def peak_cpu_throughput(data_dir):
    path = os.path.join(data_dir, "throughput_cpu.csv")
    if not os.path.exists(path):
        return None
    best = None
    with open(path) as f:
        for row in csv.DictReader(f):
            rate = float(row["segment_steps_per_sec"])
            if best is None or rate > best[0]:
                best = (rate, row)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="out/scenes")
    ap.add_argument("--figs", default="out/figs_dark")
    ap.add_argument("--data", default="docs/data")
    ap.add_argument("--out", default="out/video/demo.mp4")
    ap.add_argument("--repo", default="source: this repository")
    ap.add_argument("--preview", action="store_true", help="every 6th frame, for a quick look")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    drape = rs.Trajectory(args.scenes, "drape")
    grid = rs.Trajectory(args.scenes, "grid")
    stride = 6 if args.preview else 1

    writer = imageio.get_writer(args.out, fps=FPS // stride, codec="libx264", quality=8,
                                macro_block_size=1, pixelformat="yuv420p")

    # 0-10 s: the shaded hero shot.
    clip(writer, drape, "drape", "shaded",
         "A cable drapes over a post; one end is wound ten turns and supercoils",
         range(0, min(600, len(drape)), stride))

    # 10-20 s: the same simulation as wireframe, contacts shown.
    clip(writer, drape, "drape", "wire",
         "Same simulation: centerline, particles, and every contact point",
         range(0, min(600, len(drape)), stride))

    # 20-35 s: the batch.
    clip(writer, grid, "grid", "shaded", "256 independent rods, stepped as one batch",
         range(0, min(900, len(grid)), stride))

    # 35-50 s: the evidence.
    slides = [
        ("cantilever_convergence.png", "Second-order convergence against Euler-Bernoulli"),
        ("capstan.png", "Capstan equation T2/T1 = exp(mu theta), reproduced across wrap angles"),
        ("twist_buckling.png", "Michell twist-buckling threshold, converging under refinement"),
        ("stability_envelope.png", "Stable timestep: flat in stiffness, set by sweeps per step"),
        ("failure_study.png", "Breakdown at ~2/3 of an element of motion per sweep"),
    ]
    for name, caption in slides:
        path = os.path.join(args.figs, name)
        if os.path.exists(path):
            hold(writer, figure_slide(path, caption), 3.0 / stride)
        else:
            print("  skipping missing figure", path)

    # 50-60 s: the headline, stated with its hardware and its limits.
    best = peak_cpu_throughput(args.data)
    if best:
        rate, row = best
        big = f"{rate / 1e6:.1f} M"
        lines = [
            "rod-segment-substeps per second",
            f"laptop CPU, {row['threads']} threads, {row['rods']} rods x {row['segments']} segments",
            "CUDA kernels built; GPU run pending a driver that supports CUDA 13",
        ]
    else:
        big, lines = None, ["Cosserat rods, validated"]
    hold(writer, card(lines, sub=args.repo, big=big), 10.0 / stride)

    writer.close()
    print("wrote", args.out)


if __name__ == "__main__":
    main()
