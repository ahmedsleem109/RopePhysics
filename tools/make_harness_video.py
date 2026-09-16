"""The wire-harness demo video: a robot learns to route a cable, in simulation.

    python tools/make_harness_video.py [--preview]

Inputs (make them first):

    rodsim harness-learning          # docs/data/harness_*.csv, harness_learned.policy,
                                     # out/scenes/harness_grid.f32
    # One CPU attempt per close-up, each in its own directory holding the
    # policy (and cable) to replay:
    #   out/harness/hand   no harness.policy        -> the hand-written motion
    #   out/harness/soft   learned policy, harness.cable "0.4 1.4"
    #   out/harness/stiff  learned policy, harness.cable "4.0 0.6"
    rodsim scene harness --out out/harness/<name>

Writes out/video/harness.mp4 (1280x720, 30 fps) and docs/media/harness.gif (the
learning grid and the learned motion).

Structure (~45 s):
    title
    the task: under peg A, over peg B, through the clip
    the hand-written motion, one cable (fails)
    learning: 256 of each iteration's 1024 attempts, top-down, green = routed
    the learned motion on the softest and the stiffest cable (both route)
    end card with the measured numbers
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_scene as rs  # noqa: E402

FPS = 30
GREEN = (96, 200, 120)
RED = (232, 84, 84)

# The board, as in src/apps/harness_routing.h.
PEG_A, PEG_B, CLIP_X, CLIP_HALF_GAP = (0.25, -0.06), (0.55, 0.06), 0.85, 0.018
PARTICLES = 121


def smoothstep(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def moving_camera(t):
    """Wide while the gripper picks the cable up, then in close on the board."""
    wide, close = rs.default_camera("harness-wide"), rs.default_camera("harness")
    u = smoothstep((t - 1.0) / 2.0)
    eye = wide.eye + (close.eye - wide.eye) * u
    target = (wide.eye + wide.forward) + ((close.eye + close.forward) - (wide.eye + wide.forward)) * u
    return rs.Camera(eye, target, fov_deg=42.0)


class Out:
    def __init__(self, path, stride):
        self.writer = imageio.get_writer(path, fps=FPS // stride, codec="libx264", quality=8,
                                         macro_block_size=1, pixelformat="yuv420p")
        self.stride = stride
        self.frames = 0

    @property
    def seconds(self):
        return self.frames * self.stride / FPS

    def add(self, img):
        self.writer.append_data(np.asarray(img))
        self.frames += 1

    def hold(self, img, seconds):
        for _ in range(max(1, int(seconds * FPS / self.stride))):
            self.add(img)


def text_center(draw, y, text, size, color, bold=False, width=rs.WIDTH, x0=0):
    f = rs.font(size, bold)
    w = draw.textlength(text, font=f)
    draw.text((x0 + (width - w) / 2, y), text, font=f, fill=color)


def card(lines):
    img = Image.new("RGB", (rs.WIDTH, rs.HEIGHT), rs.BACKGROUND)
    d = ImageDraw.Draw(img)
    y = rs.HEIGHT // 2 - 30 * len(lines)
    for i, (text, size, color, bold) in enumerate(lines):
        text_center(d, y, text, size, color, bold)
        y += size + 22
    return img


def banner(img, text, color):
    d = ImageDraw.Draw(img)
    f = rs.font(30, bold=True)
    w = d.textlength(text, font=f)
    x, y = (rs.WIDTH - w) / 2, 96
    d.rounded_rectangle([x - 18, y - 8, x + w + 18, y + 44], radius=10, fill=(20, 22, 28),
                        outline=color, width=3)
    d.text((x, y), text, font=f, fill=color)
    return img


# ------------------------------------------------------------------ segments

def attempt_clip(out, directory, title, caption, verdict=None, verdict_color=None, hold=2.0):
    traj = rs.Trajectory(directory, "harness")
    decor = rs.HarnessDecor(directory, "harness", traj)
    step = 2 * out.stride  # trajectories are recorded at 60 fps
    last = None
    for f in range(0, len(traj), step):
        t = f / traj.fps
        last = rs.render_frame(traj, f, moving_camera(t), title=title, caption=caption, decor=decor)
        out.add(last)
    if verdict:
        banner(last, verdict, verdict_color)
    out.hold(last, hold)


def task_slide(out, directory):
    traj = rs.Trajectory(directory, "harness")
    decor = rs.HarnessDecor(directory, "harness", traj)
    cam = rs.default_camera("harness")
    # What done looks like: the last frame of a routed attempt, labelled.
    traj.meta["labels"] = [
        {"at": [PEG_A[0], PEG_A[1] - 0.05, 0.11], "text": "under peg A"},
        {"at": [PEG_B[0], PEG_B[1] + 0.02, 0.12], "text": "over peg B"},
        {"at": [CLIP_X, 0.0, 0.10], "text": "through the clip"},
        {"at": [-0.03, 0.0, 0.07], "text": "connector"},
    ]
    img = rs.render_frame(traj, len(traj) - 1, cam, title="The job: route the cable along the board",
                          caption="Car and aircraft wiring looms are still largely routed by hand",
                          decor=decor)
    out.hold(img, 4.0)


def load_grid(data_dir, scenes_dir):
    rows = list(csv.DictReader(open(os.path.join(data_dir, "harness_grid.csv"))))
    iterations = max(int(r["iteration"]) for r in rows) + 1
    routed = np.zeros((iterations, 256), bool)
    for r in rows:
        routed[int(r["iteration"]), int(r["rod"])] = r["routed"] == "1"
    shapes = np.fromfile(os.path.join(scenes_dir, "harness_grid.f32"), np.float32)
    shapes = shapes.reshape(-1, 256, PARTICLES, 2)[:iterations]
    learning = list(csv.DictReader(open(os.path.join(data_dir, "harness_learning.csv"))))
    return shapes, routed, learning


def grid_frame(shapes, routed, it, rate, seconds_so_far, attempts):
    img = Image.new("RGB", (rs.WIDTH * 2, rs.HEIGHT * 2), rs.BACKGROUND)
    d = ImageDraw.Draw(img)
    # Board window shown per tile, metres.
    xmin, xmax, ymin, ymax = -0.06, 1.24, -0.42, 0.30
    tiles, tile, gap = 16, 100, 5
    tile_h = int(tile * (ymax - ymin) / (xmax - xmin))
    x0, y0 = 60, (rs.HEIGHT * 2 - tiles * (tile_h + gap)) // 2 + 40
    sx = tile / (xmax - xmin)

    def to_px(px, py, cx, cy):
        return cx + (px - xmin) * sx, cy + (ymax - py) * sx

    for k in range(256):
        cx = x0 + (k % tiles) * (tile + gap)
        cy = y0 + (k // tiles) * (tile_h + gap)
        ok = routed[it, k]
        d.rectangle([cx, cy, cx + tile, cy + tile_h],
                    fill=(22, 40, 30) if ok else (44, 24, 26))
        for (px, py), r in ((PEG_A, 3), (PEG_B, 3), ((CLIP_X, CLIP_HALF_GAP), 2),
                            ((CLIP_X, -CLIP_HALF_GAP), 2)):
            qx, qy = to_px(px, py, cx, cy)
            d.ellipse([qx - r, qy - r, qx + r, qy + r], fill=(130, 136, 150))
        pts = shapes[it, k]
        keep = (pts[:, 0] > xmin) & (pts[:, 0] < xmax) & (pts[:, 1] > ymin) & (pts[:, 1] < ymax)
        line = [to_px(float(px), float(py), cx, cy) for (px, py), kp in zip(pts, keep) if kp]
        if len(line) > 1:
            d.line(line, fill=GREEN if ok else RED, width=3)

    # Right-hand panel.
    px = x0 + tiles * (tile + gap) + 70
    f_big, f_mid, f_small = rs.font(128, True), rs.font(44, True), rs.font(34)
    d.text((px, 150), "Learning, in simulation", font=f_mid, fill=rs.TEXT)
    d.text((px, 230), f"iteration {it + 1}", font=f_small, fill=rs.TEXT_DIM)
    d.text((px, 300), f"{100 * rate:.0f}%", font=f_big, fill=GREEN if rate > 0.5 else rs.ACCENT)
    d.text((px, 440), f"of {attempts} attempts routed", font=f_small, fill=rs.TEXT)
    lines = [
        "every attempt gets a different cable:",
        "stiffness 0.4-4x, friction 0.6-1.4x",
        "",
        f"{seconds_so_far:.0f} s of GPU time so far",
        "(one laptop RTX 3060)",
        "",
        f"shown: 256 of the {attempts} final cable shapes",
        "green = routed, red = not",
    ]
    y = 560
    for line in lines:
        d.text((px, y), line, font=f_small, fill=rs.TEXT_DIM)
        y += 48
    return img.resize((rs.WIDTH, rs.HEIGHT), Image.LANCZOS)


def learning_segment(out, shapes, routed, learning, attempts):
    elapsed = 0.0
    for it in range(len(learning)):
        elapsed += float(learning[it]["seconds"])
        img = grid_frame(shapes, routed, it, float(learning[it]["success_rate"]), elapsed, attempts)
        out.hold(img, 1.5 if 0 < it < len(learning) - 1 else 2.5)


def side_by_side(out, left_dir, right_dir, left_label, right_label, title, caption):
    trajs = [rs.Trajectory(p, "harness") for p in (left_dir, right_dir)]
    decors = [rs.HarnessDecor(p, "harness", t) for p, t in zip((left_dir, right_dir), trajs)]
    n = min(len(t) for t in trajs)
    step = 2 * out.stride
    last = None
    for f in list(range(0, n, step)):
        img = Image.new("RGB", (rs.WIDTH, rs.HEIGHT), rs.BACKGROUND)
        t = f / trajs[0].fps
        for side, (traj, decor, label) in enumerate(zip(trajs, decors, (left_label, right_label))):
            panel = rs.render_frame(traj, f, moving_camera(t), decor=decor)
            panel = panel.resize((rs.WIDTH // 2 - 6, rs.HEIGHT // 2 - 4), Image.LANCZOS)
            img.paste(panel, (side * (rs.WIDTH // 2 + 6), rs.HEIGHT // 4 + 10))
            d = ImageDraw.Draw(img)
            text_center(d, rs.HEIGHT // 4 - 30, label, 24, rs.TEXT, True, rs.WIDTH // 2,
                        side * (rs.WIDTH // 2))
        rs.overlay(img, title=title, caption=caption)
        last = img
        out.add(img)
    d = ImageDraw.Draw(last)
    for side in (0, 1):
        text_center(d, rs.HEIGHT * 3 // 4 + 30, "routed", 30, GREEN, True, rs.WIDTH // 2,
                    side * (rs.WIDTH // 2))
    out.hold(last, 2.5)


def make_gif(video, start, end, path):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    vf = "fps=12,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=160:stats_mode=diff[p];" \
         "[b][p]paletteuse=dither=bayer"
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-ss", f"{start:.2f}", "-to", f"{end:.2f}",
                    "-i", video, "-vf", vf, "-loop", "0", path], check=True)
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempts", default="out/harness")
    ap.add_argument("--scenes", default="out/scenes")
    ap.add_argument("--data", default="docs/data")
    ap.add_argument("--out", default="out/video/harness.mp4")
    ap.add_argument("--gif", default="docs/media/harness.gif")
    ap.add_argument("--preview", action="store_true", help="every 3rd frame, for a quick look")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    evaluation = {r["policy"]: r for r in csv.DictReader(open(os.path.join(args.data,
                                                                          "harness_evaluation.csv")))}
    learned = evaluation["learned"]
    attempts = int(learned["attempts"])
    learned_rate = float(learned["success_rate"])
    hand_rate = float(evaluation["hand_written"]["success_rate"])
    learn_seconds = float(learned["learning_seconds"])
    shapes, routed, learning = load_grid(args.data, args.scenes)
    out = Out(args.out, 3 if args.preview else 1)
    d = lambda sub: os.path.join(args.attempts, sub)  # noqa: E731
    cpu_caption = "one attempt, simulated on the CPU, played back in real time"

    out.hold(card([("Teaching a robot to route a wire harness", 46, rs.TEXT, True),
                   ("learned in simulation, across thousands of different cables", 28, rs.TEXT_DIM,
                    False)]), 3.0)
    task_slide(out, d("soft"))
    attempt_clip(out, d("hand"), "A motion programmed by hand", cpu_caption,
                 "Not routed: a slack loop, never laid against peg A", RED)
    out.hold(card([("A motion that works for one cable fails for the next.", 34, rs.TEXT, True),
                   ("So the robot learns it: try 1024 motions, each on a different cable,", 26,
                    rs.TEXT_DIM, False),
                   ("keep the best 10%, repeat. The physics runs on the GPU.", 26, rs.TEXT_DIM,
                    False)]), 4.0)
    gif_start = out.seconds
    learning_segment(out, shapes, routed, learning, attempts)
    side_by_side(out, d("soft"), d("stiff"), "soft cable, high friction (E x0.4, μ x1.4)",
                 "stiff cable, low friction (E x4, μ x0.6)",
                 "The learned motion, on the two most different cables", cpu_caption)
    gif_end = out.seconds
    out.hold(card([(f"Learned motion: {100 * learned_rate:.0f}% of {attempts} new random cables "
                    f"routed", 36, GREEN, True),
                   (f"hand-written motion: {100 * hand_rate:.0f}%", 30, rs.TEXT, False),
                   (f"{len(learning)} iterations x {attempts} simulated attempts: "
                    f"{learn_seconds:.0f} s on one laptop GPU", 26, rs.TEXT_DIM, False),
                   ("Physics validated against beam theory, the capstan equation", 22, rs.TEXT_DIM,
                    False),
                   ("and a double-precision CPU reference", 22, rs.TEXT_DIM, False)]), 5.0)
    out.writer.close()
    print(f"wrote {args.out} ({out.seconds:.1f} s)")
    if not args.preview and args.gif:
        os.makedirs(os.path.dirname(args.gif), exist_ok=True)
        make_gif(args.out, gif_start, gif_end, args.gif)


if __name__ == "__main__":
    main()
