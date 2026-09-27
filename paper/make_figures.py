"""Build the paper-only figures in paper/figures/ (the result figures live in results/figures/).

    python paper/make_figures.py

teaser.png   frames captured from the deployed browser demo (eval/browser_bench/capture.mjs)
models.png   block diagrams of both simulators
loop.png     the interaction loop: seed -> model -> prediction -> human action -> model ...
"""

import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval import plotstyle as ps  # noqa: E402

OUT = ROOT / "paper" / "figures"
CAPTURES = ROOT / "results" / "eval" / "captures"


def capture_frames(capture, times):
    meta = json.loads((capture / "canvas.json").read_text())
    video = next(capture.glob("*.webm"))
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
                               capture_output=True, text=True, check=True).stdout)
    start = dur - meta["record_seconds"]
    b = meta["box"]
    # Crop below the HUD text (top ~18% of the canvas) and above the on-screen WASD buttons.
    top = int(b["y"] + 0.17 * b["height"])
    h = int(0.66 * b["height"])
    frames = []
    for t in times:
        tmp = OUT / "_frame.png"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start + t:.2f}", "-i", str(video),
                        "-frames:v", "1", "-vf", f"crop={int(b['width'])}:{h}:{int(b['x'])}:{top}", str(tmp)], check=True)
        frames.append(np.asarray(Image.open(tmp).convert("RGB")))
        tmp.unlink()
    return frames


def teaser():
    times = [0.5, 4, 8, 12, 16]
    rows = [("CARLA (driving)", capture_frames(CAPTURES / "driving_gentle_s2", times)),
            ("TinySkies (flying)", capture_frames(CAPTURES / "flying_a", times))]
    fig, axes = plt.subplots(2, len(times), figsize=(2.3 * len(times), 3.25))
    for r, (name, frames) in enumerate(rows):
        for c, img in enumerate(frames):
            ax = axes[r, c]
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(f"t = {times[c]:.0f} s", fontsize=9, color=ps.INK_2)
            if c == 0:
                ax.set_ylabel(name, fontsize=9.5, color=ps.INK)
    fig.subplots_adjust(wspace=0.04, hspace=0.06)
    fig.savefig(OUT / "teaser.png")
    plt.close(fig)


def box(ax, x, y, w, h, text, color, fs=8.2, text_color=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=color, ec="none"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=text_color or ps.INK)


def arrow(ax, a, b, color=None, rad=0.0, lw=1.2):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, lw=lw, color=color or ps.INK_2,
                                 connectionstyle=f"arc3,rad={rad}"))


def models():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    img_c, state_c, act_c, out_c, core_c = "#dbe8f8", "#dcefe6", "#fbe3d6", "#f1efe9", "#ebe9f7"

    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")
    ax.set_title("CARLA: one-step multimodal transition (1.58 M params)", fontsize=10.5, loc="left")
    box(ax, 0.1, 5.6, 2.0, 0.8, "RGB 96×96", img_c)
    box(ax, 0.1, 4.2, 2.0, 0.6, "LiDAR latent 64", state_c)
    box(ax, 0.1, 3.4, 2.0, 0.6, "telemetry 5", state_c)
    box(ax, 0.1, 2.6, 2.0, 0.6, "waypoints 25×3", state_c)
    box(ax, 0.1, 1.5, 2.0, 0.8, "throttle,\nbrake, steer", act_c)
    box(ax, 2.8, 5.6, 2.0, 0.8, "CNN encoder\n→ 6×6×192", core_c)
    box(ax, 2.8, 2.6, 2.0, 1.6, "per-input MLPs,\nfused into a\nstate-action\nvector (224)", core_c, fs=7.8)
    box(ax, 5.3, 4.3, 1.9, 0.9, "Dense → 6×6×48\nconditioning map", core_c)
    box(ax, 5.3, 5.6, 1.9, 0.8, "concat → CNN\ndecoder (↑×16)", core_c)
    box(ax, 5.3, 2.6, 1.9, 1.0, "MLP head\n192 → 192", core_c)
    box(ax, 7.8, 5.6, 2.1, 0.8, "next RGB", img_c)
    box(ax, 7.8, 3.6, 2.1, 0.6, "next LiDAR latent", state_c, fs=7.6)
    box(ax, 7.8, 2.8, 2.1, 0.6, "next telemetry", state_c)
    box(ax, 7.8, 2.0, 2.1, 0.6, "next waypoints", state_c)
    for y in (4.5, 3.7, 2.9):
        arrow(ax, (2.1, y), (2.8, 3.4))
    arrow(ax, (2.1, 1.9), (2.8, 2.9))
    arrow(ax, (2.1, 6.0), (2.8, 6.0))
    arrow(ax, (4.8, 6.0), (5.3, 6.0))
    arrow(ax, (4.8, 3.6), (5.3, 4.6))
    arrow(ax, (6.25, 5.2), (6.25, 5.6))
    arrow(ax, (4.8, 3.2), (5.3, 3.1))
    arrow(ax, (7.2, 6.0), (7.8, 6.0))
    for y in (3.9, 3.1, 2.3):
        arrow(ax, (7.2, 3.1), (7.8, y))
    ax.text(5.0, 0.95, "state → image only: the structured-state head never sees pixels", ha="center", fontsize=8.2, color=ps.INK_2, style="italic")
    ax.text(5.0, 0.45, "every output is fed back as the next input (no frame history)", ha="center", fontsize=8.2, color=ps.INK_2, style="italic")

    ax = axes[1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")
    ax.set_title("TinySkies: VAE + recurrent latent dynamics (3.79 M params)", fontsize=10.5, loc="left")
    box(ax, 0.1, 5.6, 1.9, 0.8, "seed frame\n96×96", img_c)
    box(ax, 2.5, 5.6, 1.9, 0.8, "encoder\n(once)", core_c)
    box(ax, 0.1, 3.7, 2.2, 0.9, "latents z[t−3..t]\n4 × 6×6×64", state_c)
    box(ax, 0.1, 2.5, 2.2, 0.8, "past actions\na[t−4..t−1]", act_c)
    box(ax, 0.1, 1.4, 2.2, 0.8, "action a[t]\nforward, brake, turn", act_c)
    box(ax, 2.9, 2.9, 2.2, 1.5, "Dense 128\nper step\n→ GRU 256 ×2", core_c)
    box(ax, 5.6, 2.2, 1.8, 1.4, "Dense 256 ×2\n→ Δz", core_c)
    box(ax, 7.9, 3.6, 2.0, 0.8, "z[t+1] =\nz[t] + Δz", state_c)
    box(ax, 7.9, 5.6, 2.0, 0.8, "decoder\n→ frame", core_c)
    arrow(ax, (2.0, 6.0), (2.5, 6.0))
    arrow(ax, (3.45, 5.6), (1.2, 4.6), rad=0.2)
    arrow(ax, (2.3, 4.1), (2.9, 3.8))
    arrow(ax, (2.3, 2.9), (2.9, 3.3))
    arrow(ax, (2.3, 1.8), (5.6, 2.6))
    arrow(ax, (5.1, 3.6), (5.6, 3.1))
    arrow(ax, (7.4, 2.9), (7.9, 3.9))
    arrow(ax, (8.9, 4.4), (8.9, 5.6))
    arrow(ax, (8.3, 3.6), (1.8, 3.7), rad=0.35, color=ps.ORANGE)
    ax.text(5.0, 0.95, "rolls forward entirely in latent space; the decoder is only a viewer", ha="center", fontsize=8.2, color=ps.INK_2, style="italic")
    ax.text(5.0, 0.45, "the new latent is appended to the 4-step history (orange)", ha="center", fontsize=8.2, color=ps.INK_2, style="italic")
    fig.savefig(OUT / "models.png")
    plt.close(fig)


def loop():
    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.set_xlim(0, 12); ax.set_ylim(-0.6, 3.2); ax.axis("off")
    box(ax, 0.1, 1.1, 1.9, 1.0, "real seed state\n(one frame)", "#f1efe9")
    box(ax, 2.7, 1.1, 2.4, 1.0, "learned transition\n$\\hat s_{t+1} = f_\\theta(\\hat s_t, a_t)$", "#ebe9f7")
    box(ax, 5.9, 1.1, 2.3, 1.0, "predicted frame\n+ predicted state", "#dbe8f8")
    box(ax, 9.0, 1.1, 2.8, 1.0, "human (keyboard)\nor controller", "#fbe3d6")
    arrow(ax, (2.0, 1.6), (2.7, 1.6))
    arrow(ax, (5.1, 1.6), (5.9, 1.6))
    arrow(ax, (8.2, 1.6), (9.0, 1.6))
    arrow(ax, (10.4, 1.1), (3.9, 1.1), rad=-0.35, color=ps.ORANGE)
    arrow(ax, (7.05, 2.1), (3.9, 2.1), rad=0.35, color=ps.BLUE)
    ax.text(5.5, 2.95, "prediction is fed back as the next input", ha="center", fontsize=8.5, color=ps.BLUE)
    ax.text(7.1, -0.5, "action $a_t$ chosen after seeing the prediction", ha="center", fontsize=8.5, color=ps.ORANGE)
    fig.savefig(OUT / "loop.png")
    plt.close(fig)


if __name__ == "__main__":
    ps.apply()
    OUT.mkdir(parents=True, exist_ok=True)
    teaser()
    models()
    loop()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
