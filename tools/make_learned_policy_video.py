"""The trained-policy clip: the learned harness motion on the softest and stiffest cable.

    python tools/make_learned_policy_video.py

Needs out/harness/soft and out/harness/stiff (see tools/make_harness_video.py).
Writes out/video/harness_learned_policy.mp4.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_harness_video as m
import render_scene as rs

out = m.Out("out/video/harness_learned_policy.mp4", 1)
out.hold(m.card([("The learned routing motion", 46, rs.TEXT, True),
                 ("learned on 10,240 simulated attempts; routes 1024 of 1024 new random cables", 26,
                  rs.TEXT_DIM, False)]), 2.5)
cap = "one attempt, simulated on the CPU, played back in real time"
m.attempt_clip(out, "out/harness/soft", "Learned motion: soft cable, high friction (E x0.4, μ x1.4)",
               cap, "Routed: under A, over B, through the clip", m.GREEN)
m.attempt_clip(out, "out/harness/stiff", "Learned motion: stiff cable, low friction (E x4, μ x0.6)",
               cap, "Routed: under A, over B, through the clip", m.GREEN)
out.writer.close()
print("wrote", out.seconds, "s")
