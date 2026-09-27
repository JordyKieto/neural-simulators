"""Evaluate the released TinySkies neural simulator. Writes results/eval/tinyskies/.

1. vae         Reconstruction ceiling: D(E(o)) vs o. No dynamics model can beat this in pixels.
2. replay      Seed with 4 real frames and their controls, feed the recorded controls, and roll
               forward in latent space for H steps. Compare against the real frames (`rgb_*`)
               and against the VAE reconstruction of the real frames (`dyn_*`), which isolates
               dynamics error from reconstruction error. Reference: `copy` (hold the seed latent).
3. long        2,000-step free-running rollouts from the browser seed frames under scripted actions.
4. response    Turn / forward / brake sweeps: does the view rotate in the commanded direction?

Sessions: every 10th session in manifest order plus all human-piloted sessions (0070-0077).
The exact training hold-out cannot be reconstructed (see PROVENANCE.md, section 5), so these
results are reported as possibly seen during training.

Usage: python eval/tinyskies_eval.py --processed DIR [--horizon 64]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval import metrics  # noqa: E402
from tinyskies.data import load_sessions  # noqa: E402
from tinyskies.runtime import HISTORY, TinySkiesSimulator  # noqa: E402

SNAPSHOT_KEEP = [0, 1, 2, 4, 5, 10, 20]  # snapshot indices kept on disk (every 100 steps -> steps 0..2000)
OUT = ROOT / "results" / "eval" / "tinyskies"
MANUAL = [f"session_{i:04d}" for i in range(70, 78)]


def eval_sessions(processed):
    names = sorted(json.loads((Path(processed) / "manifest.json").read_text())["sessions"])
    every10 = [n for i, n in enumerate(names) if i % 10 == 9]
    return sorted(set(every10) | set(MANUAL))


def _batched(fn, x, size=256):
    return np.concatenate([fn(x[i : i + size]) for i in range(0, len(x), size)])


def vae_ceiling(sim, sessions):
    rows = []
    for name, (frames, _) in sessions.items():
        recon = _batched(lambda f: sim.decode(sim.encode(f)), frames)
        rows.append(
            pd.DataFrame(
                {
                    "session": name,
                    "frame": np.arange(len(frames)),
                    "mae": metrics.mae(recon, frames),
                    "psnr": metrics.psnr(recon, frames),
                    "ssim": metrics.ssim(recon, frames),
                    "sharpness_recon": metrics.sharpness(recon),
                    "sharpness_true": metrics.sharpness(frames),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def replay(sim, sessions, horizon, stride):
    rows = []
    for name, (frames, controls) in sessions.items():
        n = len(frames)
        starts = np.arange(0, n - HISTORY - horizon, stride)
        if not len(starts):
            continue
        latents = _batched(sim.encode, frames)
        recon = _batched(sim.decode, latents)
        hist_frames = np.stack([frames[s : s + HISTORY] for s in starts])
        # control_history[j] is the action that led into history frame j (zero for the first).
        hist_controls = np.stack(
            [np.concatenate([np.zeros((1, 3), np.float32), controls[s : s + HISTORY - 1]]) for s in starts]
        )
        sim.reset(hist_frames, hist_controls)
        seed_latent = latents[starts + HISTORY - 1]
        for k in range(horizon):
            t = starts + HISTORY - 1 + k  # index of the newest frame in the history
            z = sim.step(controls[t])
            pred = sim.decode(z)
            target, target_recon = frames[t + 1], recon[t + 1]
            rows.append(
                pd.DataFrame(
                    {
                        "session": name,
                        "human": name in MANUAL,
                        "start": starts,
                        "k": k + 1,
                        "latent_mse": np.square(z - latents[t + 1]).reshape(len(z), -1).mean(1),
                        "copy_latent_mse": np.square(seed_latent - latents[t + 1]).reshape(len(z), -1).mean(1),
                        "rgb_mae": metrics.mae(pred, target),
                        "rgb_ssim": metrics.ssim(pred, target),
                        "dyn_mae": metrics.mae(pred, target_recon),
                        "dyn_ssim": metrics.ssim(pred, target_recon),
                        "copy_mae": metrics.mae(recon[starts + HISTORY - 1], target_recon),
                        "sharpness_pred": metrics.sharpness(pred),
                        "sharpness_recon": metrics.sharpness(target_recon),
                        "turn": controls[t, 2],
                    }
                )
            )
        print(f"replay {name}: {len(starts)} starts x {horizon} steps")
    return pd.concat(rows, ignore_index=True)


def _browser_seed_frames():
    root = ROOT / "samples/tinyskies/browser_seeds/session_0005"
    rows = [json.loads(l) for l in (root / "samples.jsonl").read_text().splitlines() if l.strip()]
    frames = []
    for r in rows:
        img = Image.open(root / "frames" / Path(r["frame_path"]).name).convert("RGB")
        frames.append(np.asarray(img.resize((96, 96), Image.BILINEAR), np.float32) / 255.0)
    return np.stack(frames)


def _random_keys(rng, steps):
    """Same distribution as the recording bot, one decision per step."""
    out, k = np.zeros((steps, 3), np.float32), 0
    while k < steps:
        hold = rng.integers(2, 10)
        keys = rng.choice(list("wasd"), size=rng.integers(0, 3), replace=False)
        a = [float("w" in keys), float("s" in keys), float("d" in keys) - float("a" in keys)]
        out[k : k + hold] = a
        k += hold
    return out


SCRIPTS = {
    "none": lambda k: (0, 0, 0),
    "forward": lambda k: (1, 0, 0),
    "brake": lambda k: (0, 1, 0),
    "left": lambda k: (0, 0, -1),
    "right": lambda k: (0, 0, 1),
}


def long_rollouts(sim, seeds, steps, snapshot_every):
    rows, snaps = [], {}
    rng = np.random.default_rng(0)
    scripted = {**{k: (lambda fn: np.array([fn(i) for i in range(steps)], np.float32))(fn) for k, fn in SCRIPTS.items()},
                "random_bot": _random_keys(rng, steps)}
    for script, actions in scripted.items():
        sim.reset(seeds)
        prev = sim.decode(sim.latents[:, -1])
        seed_recon = prev
        frames = [prev]
        for k in range(steps):
            z = sim.step(np.tile(actions[k], (len(seeds), 1)))
            v = sim.decode(z)
            rows.append(
                pd.DataFrame(
                    {
                        "script": script,
                        "seed": np.arange(len(v)),
                        "k": k + 1,
                        "frame_change": metrics.frame_change(prev, v),
                        "shift_px": metrics.horizontal_shift(prev, v),
                        "sharpness": metrics.sharpness(v),
                        "mean_intensity": v.reshape(len(v), -1).mean(1),
                        "dist_from_seed": metrics.mae(v, seed_recon),
                        "latent_norm": np.linalg.norm(z.reshape(len(z), -1), axis=1),
                    }
                )
            )
            if (k + 1) % snapshot_every == 0:
                frames.append(v)
            prev = v
        snaps[script] = (np.stack(frames, 1)[:, SNAPSHOT_KEEP] * 255).astype(np.uint8)
        print(f"long {script}: {steps} steps")
    return pd.concat(rows, ignore_index=True), snaps


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--long-steps", type=int, default=2000)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sim = TinySkiesSimulator()
    names = eval_sessions(args.processed)
    sessions = load_sessions(args.processed, names)
    print("sessions:", names)

    vae_ceiling(sim, sessions).to_csv(OUT / "vae_ceiling.csv.gz", index=False)
    replay(sim, sessions, args.horizon, args.stride).to_csv(OUT / "replay.csv.gz", index=False)
    long_df, snaps = long_rollouts(sim, _browser_seed_frames(), args.long_steps, snapshot_every=100)
    long_df.to_csv(OUT / "long_rollouts.csv.gz", index=False)
    np.savez_compressed(OUT / "long_rollout_snapshots.npz", snapshot_steps=np.array(SNAPSHOT_KEEP) * 100, **snaps)
    (OUT / "config.json").write_text(
        json.dumps(
            {
                "sessions": names,
                "human_sessions": MANUAL,
                "horizon": args.horizon,
                "stride": args.stride,
                "long_steps": args.long_steps,
                "seeds": "samples/tinyskies/browser_seeds/session_0005 (9 frames)",
                "scripts": list(SCRIPTS) + ["random_bot"],
                "note": "exact training hold-out not reconstructible; sessions possibly seen in training",
            },
            indent=2,
        )
    )
    print("wrote", OUT)


if __name__ == "__main__":
    main()
