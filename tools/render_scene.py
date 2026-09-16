"""Offline renderer for the trajectories written by `rodsim scene <name>`.

    python tools/render_scene.py drape --frame 300 --mode shaded   # one PNG to check
    python tools/render_scene.py grid  --frame 200 --mode wire

No GPU, no 3D engine: a pinhole camera, painter's-order thick lines with round
caps, a cylinder shading term and a ground shadow. That is enough to read the
motion clearly, and it keeps the video pipeline to numpy + Pillow + ffmpeg.

Frames are drawn at 2x and downsampled, since Pillow's lines are not
antialiased.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct

import numpy as np
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1280, 720
SUPERSAMPLE = 2

BACKGROUND = (14, 16, 20)
FLOOR_LINE = (34, 38, 46)
ACCENT = (242, 142, 64)        # the rod, shaded mode
WIRE = (96, 200, 255)          # the rod, wireframe mode
CONTACT = (255, 72, 120)       # contact points, wireframe mode
PRIMITIVE = (88, 94, 106)
TEXT = (200, 204, 212)
TEXT_DIM = (130, 136, 148)

FONT_DIR = "C:/Windows/Fonts"


def font(size, bold=False):
    name = "segoeuib.ttf" if bold else "segoeui.ttf"
    try:
        return ImageFont.truetype(os.path.join(FONT_DIR, name), size)
    except OSError:
        return ImageFont.load_default()


# ----------------------------------------------------------------- data

class Trajectory:
    def __init__(self, directory, name):
        with open(os.path.join(directory, name + ".json")) as f:
            self.meta = json.load(f)
        path = os.path.join(directory, name + ".rodtraj")
        raw = open(path, "rb").read()
        if raw[:8] != b"RODTRAJ1":
            raise ValueError(f"{path}: not a rod trajectory")
        num_frames, num_rods, num_particles = struct.unpack_from("<3i", raw, 8)
        self.radius, self.fps = struct.unpack_from("<2f", raw, 20)
        self.num_rods, self.num_particles = num_rods, num_particles

        offset = 28
        per_frame = num_rods * num_particles * 3
        self.wall = np.empty(num_frames, np.float32)
        self.positions = np.empty((num_frames, num_rods, num_particles, 3), np.float32)
        self.contacts = []
        for f in range(num_frames):
            self.wall[f] = struct.unpack_from("<f", raw, offset)[0]
            offset += 4
            self.positions[f] = np.frombuffer(raw, np.float32, per_frame, offset).reshape(
                num_rods, num_particles, 3)
            offset += per_frame * 4
            (count,) = struct.unpack_from("<i", raw, offset)
            offset += 4
            self.contacts.append(np.frombuffer(raw, np.float32, count * 3, offset).reshape(count, 3))
            offset += count * 12

    def __len__(self):
        return len(self.wall)


# ----------------------------------------------------------------- camera

class Camera:
    def __init__(self, eye, target, fov_deg=40.0, up=(0, 0, 1)):
        self.eye = np.asarray(eye, np.float64)
        target = np.asarray(target, np.float64)
        f = target - self.eye
        self.forward = f / np.linalg.norm(f)
        r = np.cross(self.forward, up)
        self.right = r / np.linalg.norm(r)
        self.up = np.cross(self.right, self.forward)
        self.focal = (HEIGHT * SUPERSAMPLE / 2) / math.tan(math.radians(fov_deg) / 2)

    def project(self, points):
        """World points (..., 3) -> screen x, screen y, depth."""
        d = np.asarray(points, np.float64) - self.eye
        x = d @ self.right
        y = d @ self.up
        z = np.maximum(d @ self.forward, 1e-4)
        sx = WIDTH * SUPERSAMPLE / 2 + self.focal * x / z
        sy = HEIGHT * SUPERSAMPLE / 2 - self.focal * y / z
        return sx, sy, z


def orbit_camera(center, radius, height, angle, fov=40.0):
    eye = (center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle), height)
    return Camera(eye, center, fov)


# ----------------------------------------------------------------- drawing

LIGHT = np.array([-0.35, -0.55, 0.76])
LIGHT /= np.linalg.norm(LIGHT)


def mix(color, target, t):
    t = float(np.clip(t, 0.0, 1.0))
    return tuple(int(round(c + (g - c) * t)) for c, g in zip(color, target))


def draw_floor(draw, cam, extent=2.4, step=0.2):
    ticks = np.arange(-extent, extent + 1e-9, step)
    for t in ticks:
        for a, b in (((t, -extent, 0), (t, extent, 0)), ((-extent, t, 0), (extent, t, 0))):
            pts = np.linspace(a, b, 24)
            sx, sy, z = cam.project(pts)
            for i in range(len(pts) - 1):
                fade = np.clip((z[i] - 1.5) / 6.0, 0, 1)
                draw.line([(sx[i], sy[i]), (sx[i + 1], sy[i + 1])],
                          fill=mix(FLOOR_LINE, BACKGROUND, fade), width=SUPERSAMPLE)


def capsule_segments(prim, pieces=24):
    a, b = np.asarray(prim["a"]), np.asarray(prim["b"])
    pts = np.linspace(a, b, pieces + 1)
    return [(pts[i], pts[i + 1], prim["radius"]) for i in range(pieces)]


def render_frame(traj, frame, cam, mode="shaded", caption=None, title=None):
    img = Image.new("RGB", (WIDTH * SUPERSAMPLE, HEIGHT * SUPERSAMPLE), BACKGROUND)
    draw = ImageDraw.Draw(img)
    draw_floor(draw, cam)

    pos = traj.positions[frame]
    radius = traj.radius

    # Ground shadow first: it sits under everything and reads as depth.
    if mode == "shaded":
        for rod in pos:
            shadow = rod.copy()
            shadow[:, 2] = 0.0
            sx, sy, z = cam.project(shadow)
            for i in range(len(rod) - 1):
                w = max(1, int(2 * radius * cam.focal / z[i] * 1.6))
                draw.line([(sx[i], sy[i]), (sx[i + 1], sy[i + 1])], fill=(6, 7, 9), width=w)

    # Gather every drawable piece -- rod segments and primitive pieces -- and
    # draw far to near, so a cable wrapping over a post occludes correctly.
    items = []
    for prim in traj.meta.get("primitives", []):
        if prim["type"] == "capsule":
            for p0, p1, r in capsule_segments(prim):
                items.append((p0, p1, r, "prim"))

    for rod in pos:
        for i in range(len(rod) - 1):
            items.append((rod[i], rod[i + 1], radius, "rod"))

    p0s = np.array([it[0] for it in items])
    p1s = np.array([it[1] for it in items])
    sx0, sy0, z0 = cam.project(p0s)
    sx1, sy1, z1 = cam.project(p1s)
    depth = 0.5 * (z0 + z1)
    order = np.argsort(-depth)

    tangents = p1s - p0s
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / np.maximum(norms, 1e-9)
    # Brightest a cylinder gets under a directional light: its surface normal
    # can point anywhere perpendicular to the axis.
    diffuse = np.sqrt(np.clip(1.0 - (tangents @ LIGHT) ** 2, 0.0, 1.0))
    znear, zfar = float(depth.min()), float(depth.max()) + 1e-6

    for k in order:
        p0, p1, r, kind = items[k]
        fog = 0.45 * (depth[k] - znear) / (zfar - znear)
        if mode == "wire":
            if kind == "prim":
                color = mix(PRIMITIVE, BACKGROUND, 0.35 + fog)
                w = max(1, int(2 * r * cam.focal / depth[k]))
                draw.line([(sx0[k], sy0[k]), (sx1[k], sy1[k])], fill=color, width=w)
            else:
                draw.line([(sx0[k], sy0[k]), (sx1[k], sy1[k])],
                          fill=mix(WIRE, BACKGROUND, fog), width=2 * SUPERSAMPLE)
            continue

        base = PRIMITIVE if kind == "prim" else ACCENT
        shade = 0.30 + 0.70 * float(diffuse[k])
        color = tuple(int(c * shade) for c in base)
        color = mix(color, BACKGROUND, fog)
        w = max(2, int(round(2 * r * cam.focal / depth[k])))
        draw.line([(sx0[k], sy0[k]), (sx1[k], sy1[k])], fill=color, width=w)
        h = w / 2
        draw.ellipse([sx1[k] - h, sy1[k] - h, sx1[k] + h, sy1[k] + h], fill=color)

    if mode == "wire":
        for rod in pos:
            sx, sy, z = cam.project(rod)
            for x, y in zip(sx, sy):
                d = 1.5 * SUPERSAMPLE
                draw.ellipse([x - d, y - d, x + d, y + d], fill=WIRE)
        contacts = traj.contacts[frame]
        if len(contacts):
            sx, sy, z = cam.project(contacts)
            for x, y in zip(sx, sy):
                d = 4 * SUPERSAMPLE
                draw.ellipse([x - d, y - d, x + d, y + d], fill=CONTACT)

    img = img.resize((WIDTH, HEIGHT), Image.LANCZOS)
    draw_labels(img, cam, traj.meta.get("labels", []))
    overlay(img, title=title, caption=caption)
    return img


def draw_labels(img, cam, labels):
    """Text anchored at world points, centred above them."""
    draw = ImageDraw.Draw(img)
    f = font(22, bold=True)
    for label in labels:
        sx, sy, z = cam.project(np.array([label["at"]]))
        x, y = sx[0] / SUPERSAMPLE, sy[0] / SUPERSAMPLE
        text = label["text"].replace("mu", "\u03bc")
        w = draw.textlength(text, font=f)
        draw.text((x - w / 2, y - 14), text, font=f, fill=TEXT)


def overlay(img, title=None, caption=None):
    draw = ImageDraw.Draw(img)
    if title:
        title = title.replace("mu >=", "μ ≥")
        draw.text((36, 28), title, font=font(26, bold=True), fill=TEXT)
    if caption:
        draw.text((36, HEIGHT - 50), caption, font=font(19), fill=TEXT_DIM)


# ----------------------------------------------------------------- presets

def default_camera(name, frame=0, total=1):
    if name == "drape":
        return Camera((1.3, -1.8, 0.95), (0.08, 0.0, 0.28), fov_deg=40)
    if name == "grid":
        # A slow quarter orbit over the clip keeps a static field readable in 3D.
        angle = -2.2 + 0.5 * math.pi * frame / max(1, total)
        return orbit_camera((0.0, 0.0, 0.2), 5.0, 2.6, angle, fov=46)
    if name == "harness-top":
        return Camera((0.45, -0.3, 1.9), (0.45, -0.3, 0.0), fov_deg=45, up=(0, 1, 0))
    if name == "harness":
        return Camera((0.45, -1.25, 1.05), (0.45, -0.25, 0.0), fov_deg=45)
    if name == "cable-hanging":
        return Camera((0.0, -3.0, 1.0), (0.0, 0.0, 0.58), fov_deg=42)
    return Camera((2, -2, 1.5), (0, 0, 0.5))


def timing_caption(traj):
    t = traj.meta["timing"]
    cost = t["wall_seconds"] / t["simulated_seconds"]
    threads = t.get("threads", 1)
    if t.get("device", "CPU") == "GPU":
        where = "GPU"
    else:
        where = f"CPU, {threads} thread{'s' if threads != 1 else ''}"
    return (f"played back in real time  ·  simulating it cost {cost:.2f} s of wall clock per "
            f"simulated second ({where})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--dir", default="out/scenes")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--mode", choices=("shaded", "wire"), default="shaded")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    traj = Trajectory(args.dir, args.scene)
    cam = default_camera(args.scene, args.frame, len(traj))
    img = render_frame(traj, args.frame, cam, args.mode, caption=timing_caption(traj),
                       title=traj.meta["description"])
    out = args.out or os.path.join(args.dir, f"{args.scene}_{args.mode}_{args.frame:04d}.png")
    img.save(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
