"""Render the looping GIFs embedded in the README.

    python tools/make_readme_media.py            # every scene in out/scenes
    python tools/make_readme_media.py drape      # just one

Reads the trajectories written by `rodsim scene <name>` and renders them with
tools/render_scene.py, without titles or captions (the README carries those).
Frames go through ffmpeg's two-pass palette, which keeps a 640 px, 15 fps clip
of a dark scene to a few megabytes.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import imageio_ffmpeg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_scene as rs  # noqa: E402

# name -> (render mode, first frame, last frame, keep every Nth frame, output width).
# Scenes are recorded at 60 fps, so every 4th frame at 15 fps plays in real time.
SCENES = {
    "drape": ("shaded", 0, 600, 4, 640),
    "grid": ("shaded", 0, 900, 9, 560),
}

# Closer framings than the video's, since a README image is viewed small.
CAMERAS = {
    "drape": lambda frame, total: rs.Camera((0.85, -1.2, 0.72), (0.06, 0.0, 0.26), fov_deg=40),
}
GIF_FPS = 15


def render(name, scenes_dir, out_dir):
    mode, first, last, every, width = SCENES[name]
    traj = rs.Trajectory(scenes_dir, name)
    last = min(last, len(traj))
    frames = range(first, last, every)

    tmp = tempfile.mkdtemp(prefix=f"readme_{name}_")
    try:
        for k, f in enumerate(frames):
            cam = CAMERAS.get(name, lambda fr, tot: rs.default_camera(name, fr, tot))(f, len(traj))
            rs.render_frame(traj, f, cam, mode).save(os.path.join(tmp, f"{k:05d}.png"))
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        out = os.path.join(out_dir, f"{name}.gif")
        scale = f"fps={GIF_FPS},scale={width}:-1:flags=lanczos"
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-framerate", str(GIF_FPS),
             "-i", os.path.join(tmp, "%05d.png"),
             "-vf", f"{scale},split[a][b];[a]palettegen=max_colors=64[p];[b][p]paletteuse=dither=bayer",
             "-loop", "0", out],
            check=True)
        print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB, {len(frames)} frames)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*", default=list(SCENES))
    ap.add_argument("--dir", default="out/scenes")
    ap.add_argument("--out", default="docs/media")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for name in args.scenes:
        render(name, args.dir, args.out)


if __name__ == "__main__":
    main()
