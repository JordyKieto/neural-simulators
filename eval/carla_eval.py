"""Evaluate the released CARLA neural simulator (world_v3 epoch 20). Writes results/eval/carla/.

1. replay      Start from a real recorded state, feed the recorded controls, and let the model
               roll forward on its own predictions for H steps. Compare with the real recording
               at every step. Two references: `copy` (hold the seed frame) and `one_step` (the
               model given the true state at every step, i.e. the training-style metric).
2. long        Free-running rollouts of 2,000 steps (100 s of simulated time) from the six
               browser seeds under scripted actions: does the loop stay coherent, and how does
               the image degrade?
3. response    Controlled action sweeps from the same seeds: throttle levels, a
               throttle-then-coast / brake test (momentum), and steering levels.

Usage: python eval/carla_eval.py --recordings DIR [DIR ...] [--horizon 100]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from carla_sim.data import load_recording  # noqa: E402
from carla_sim.model import speed_mps  # noqa: E402
from carla_sim.runtime import CarlaSimulator, load_browser_seeds  # noqa: E402
from eval import metrics  # noqa: E402

SNAPSHOT_KEEP = [0, 1, 2, 4, 5, 10, 20]  # snapshot indices kept on disk (every 100 steps -> steps 0..2000)
OUT = ROOT / "results" / "eval" / "carla"
ROLES = {
    "drive_20260509_110744": "train",
    "drive_20260509_231436": "val",
    "drive_20260517_052600": "unseen",
    "drive_20260517_063215": "unseen",
}


def replay(sim, paths, horizon, stride):
    rows = []
    for path in paths:
        name = Path(path).name.replace(".processed.tfrecord", "")
        rec = load_recording(path)
        n = len(rec["vision"])
        starts = np.arange(0, n - horizon - 1, stride)
        true_latent = sim.encode_lidar(rec["lidar"])
        sim.reset(*(rec[k][starts] for k in ("vision", "lidar", "telemetry", "waypoints")))
        seed = rec["vision"][starts]
        for k in range(horizon):
            t = starts + k
            # One-step reference: true state in, compare prediction for t+1.
            one = sim.transition.run(
                None,
                {
                    "vision": rec["vision"][t],
                    "lidar_latent": true_latent[t],
                    "telemetry": rec["telemetry"][t],
                    "waypoints": rec["waypoints"][t],
                    "controls": rec["controls"][t],
                },
            )[0]
            state = sim.step(rec["controls"][t])
            target = rec["vision"][t + 1]
            pred = state["vision"]
            speed_true = speed_mps(rec["telemetry"][t + 1])
            block = pd.DataFrame(
                {
                    "recording": name,
                    "role": ROLES.get(name, "other"),
                    "start": starts,
                    "k": k + 1,
                    "rgb_mae": metrics.mae(pred, target),
                    "rgb_psnr": metrics.psnr(pred, target),
                    "rgb_ssim": metrics.ssim(pred, target),
                    "one_step_mae": metrics.mae(one, target),
                    "one_step_ssim": metrics.ssim(one, target),
                    "copy_mae": metrics.mae(seed, target),
                    "copy_ssim": metrics.ssim(seed, target),
                    "sharpness_pred": metrics.sharpness(pred),
                    "sharpness_true": metrics.sharpness(target),
                    "speed_true_mps": speed_true,
                    "speed_pred_mps": speed_mps(state["telemetry"]),
                    "telemetry_mae": np.abs(state["telemetry"] - rec["telemetry"][t + 1]).mean(axis=1),
                    "waypoints_mae": np.abs(state["waypoints"] - rec["waypoints"][t + 1]).reshape(len(t), -1).mean(axis=1),
                    "lidar_latent_cos": _cos(state["lidar_latent"], true_latent[t + 1]),
                    "steer": rec["controls"][t, 2],
                    "throttle": rec["controls"][t, 0],
                }
            )
            rows.append(block)
        print(f"replay {name}: {len(starts)} starts x {horizon} steps")
    return pd.concat(rows, ignore_index=True)


def _cos(a, b):
    return (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-9)


SCRIPTS = {
    "cruise_0.5": lambda k: (0.5, 0.0, 0.0),
    "full_throttle": lambda k: (1.0, 0.0, 0.0),
    "idle": lambda k: (0.0, 0.0, 0.0),
    "left_0.5": lambda k: (0.5, 0.0, -0.5),
    "right_0.5": lambda k: (0.5, 0.0, 0.5),
    "weave": lambda k: (0.5, 0.0, 0.5 * np.sin(2 * np.pi * k / 100)),
}


def recorded_controls(paths, steps):
    """Real control sequences (autopilot) from the recordings, concatenated end to end: an
    in-distribution action stream that never feeds the real state back."""
    seq = np.concatenate([load_recording(p)["controls"] for p in paths])
    return np.resize(seq, (steps, 3)).astype(np.float32)


def long_rollouts(sim, seeds, steps, snapshot_every, extra=None):
    """Free-running rollouts. Scripts are functions of the step; `extra` maps a name to a
    (steps, 3) action array applied identically to every seed."""
    rows, snaps = [], {}
    scripts = {name: (lambda k, fn=fn: fn(k)) for name, fn in SCRIPTS.items()}
    for name, arr in (extra or {}).items():
        scripts[name] = lambda k, arr=arr: arr[k]
    for script, fn in scripts.items():
        sim.reset(seeds["vision"], seeds["lidar"], seeds["telemetry"], seeds["waypoints"])
        prev = seeds["vision"]
        frames = [prev]
        for k in range(steps):
            a = np.tile(np.asarray(fn(k), np.float32), (len(prev), 1))
            state = sim.step(a)
            v = state["vision"]
            rows.append(
                pd.DataFrame(
                    {
                        "script": script,
                        "seed": np.arange(len(v)),
                        "k": k + 1,
                        "speed_mps": speed_mps(state["telemetry"]),
                        "v_lateral": state["telemetry"][:, 4],
                        "frame_change": metrics.frame_change(prev, v),
                        "telemetry_0": state["telemetry"][:, 0],
                        "sharpness": metrics.sharpness(v),
                        "mean_intensity": v.reshape(len(v), -1).mean(1),
                        "dist_from_seed": metrics.mae(v, seeds["vision"]),
                        "waypoint_lateral_10": state["waypoints"][:, 10, 1],
                    }
                )
            )
            if (k + 1) % snapshot_every == 0:
                frames.append(v)
            prev = v
        snaps[script] = (np.stack(frames, 1)[:, SNAPSHOT_KEEP] * 255).astype(np.uint8)  # (seed, snapshot, H, W, 3)
        print(f"long {script}: {steps} steps")
    return pd.concat(rows, ignore_index=True), snaps


def closed_loop(sim, seeds, steps, targets=(3.0, 5.0, 7.0), snapshot_every=100):
    """A hand-written controller that sees only the simulator's own predicted structured state:
    proportional speed control on predicted telemetry, and steering toward the heading of a
    predicted route waypoint. No real CARLA state is used after the seed. This tests whether the
    predicted state is usable for control, and whether feedback keeps rollouts in distribution."""
    rows, snaps = [], {}
    for target in targets:
        state = sim.reset(seeds["vision"], seeds["lidar"], seeds["telemetry"], seeds["waypoints"])
        prev, frames = seeds["vision"], [seeds["vision"]]
        for k in range(steps):
            v = speed_mps(state["telemetry"])
            err = target - v
            throttle = np.clip(0.15 * err + 0.25, 0.0, 1.0) * (err > -0.5)
            brake = np.clip(-0.2 * err, 0.0, 1.0) * (err < -1.0)
            wp = state["waypoints"][:, 5]  # (x/d, y/d, d/50) in the ego frame, y to the right
            steer = np.clip(1.5 * np.arctan2(wp[:, 1], np.maximum(wp[:, 0], 1e-3)), -1.0, 1.0)
            controls = np.stack([throttle, brake, steer], 1).astype(np.float32)
            state = sim.step(controls)
            img = state["vision"]
            rows.append(
                pd.DataFrame(
                    {
                        "target_mps": target,
                        "seed": np.arange(len(img)),
                        "k": k + 1,
                        "speed_mps": speed_mps(state["telemetry"]),
                        "telemetry_0": state["telemetry"][:, 0],
                        "throttle": throttle,
                        "brake": brake,
                        "steer": steer,
                        "frame_change": metrics.frame_change(prev, img),
                        "sharpness": metrics.sharpness(img),
                        "dist_from_seed": metrics.mae(img, seeds["vision"]),
                    }
                )
            )
            if (k + 1) % snapshot_every == 0:
                frames.append(img)
            prev = img
        snaps[f"target_{target}"] = (np.stack(frames, 1)[:, SNAPSHOT_KEEP] * 255).astype(np.uint8)
        print(f"closed loop target {target} m/s: {steps} steps")
    return pd.concat(rows, ignore_index=True), snaps


def response(sim, seeds, steps=200):
    rows = []

    def run(name, fn):
        sim.reset(seeds["vision"], seeds["lidar"], seeds["telemetry"], seeds["waypoints"])
        prev = seeds["vision"]
        for k in range(steps):
            state = sim.step(np.tile(np.asarray(fn(k), np.float32), (len(prev), 1)))
            v = state["vision"]
            rows.append(
                pd.DataFrame(
                    {
                        "condition": name,
                        "seed": np.arange(len(v)),
                        "k": k + 1,
                        "speed_mps": speed_mps(state["telemetry"]),
                        "v_forward": state["telemetry"][:, 3],
                        "v_lateral": state["telemetry"][:, 4],
                        "frame_change": metrics.frame_change(prev, v),
                        "telemetry_0": state["telemetry"][:, 0],
                        "waypoint_lateral_10": state["waypoints"][:, 10, 1],
                    }
                )
            )
            prev = v

    for level in (0.0, 0.25, 0.5, 0.75, 1.0):
        run(f"throttle_{level}", lambda k, l=level: (l, 0.0, 0.0))
    run("throttle_then_coast", lambda k: (1.0, 0.0, 0.0) if k < 60 else (0.0, 0.0, 0.0))
    run("throttle_then_brake", lambda k: (1.0, 0.0, 0.0) if k < 60 else (0.0, 1.0, 0.0))
    for steer in (-1.0, -0.5, 0.0, 0.5, 1.0):
        run(f"steer_{steer}", lambda k, s=steer: (0.5, 0.0, s))
    return pd.concat(rows, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recordings", nargs="+", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--long-steps", type=int, default=2000)
    parser.add_argument("--only", choices=["recorded", "closed_loop"], help="run only one experiment")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sim = CarlaSimulator()
    seeds = load_browser_seeds()
    # Two real autopilot control streams replayed open-loop: all recordings in the given order
    # (starting with the slower training recording), and the same streams led by the recording
    # with the highest mean throttle (the faster held-out recording).
    by_throttle = sorted(args.recordings, key=lambda p: -float(load_recording(p)["controls"][:, 0].mean()))
    recorded = {
        "recorded_controls": recorded_controls(args.recordings, args.long_steps),
        "recorded_controls_fast_first": recorded_controls(by_throttle, args.long_steps),
    }
    if args.only == "closed_loop":
        df, snaps = closed_loop(sim, seeds, args.long_steps)
        df.to_csv(OUT / "closed_loop.csv.gz", index=False)
        np.savez_compressed(OUT / "closed_loop_snapshots.npz", snapshot_steps=np.array(SNAPSHOT_KEEP) * 100, **snaps)
        return
    if args.only == "recorded":
        SCRIPTS.clear()
        df, snaps = long_rollouts(sim, seeds, args.long_steps, 100, extra=recorded)
        long_df = pd.read_csv(OUT / "long_rollouts.csv.gz")
        long_df = pd.concat([long_df[~long_df.script.isin(recorded)], df], ignore_index=True)
        long_df.to_csv(OUT / "long_rollouts.csv.gz", index=False)
        old = dict(np.load(OUT / "long_rollout_snapshots.npz"))
        np.savez_compressed(OUT / "long_rollout_snapshots.npz", **{**old, **snaps})
        return

    rep = replay(sim, args.recordings, args.horizon, args.stride)
    rep.to_csv(OUT / "replay.csv.gz", index=False)
    long_df, snaps = long_rollouts(sim, seeds, args.long_steps, snapshot_every=100, extra=recorded)
    long_df.to_csv(OUT / "long_rollouts.csv.gz", index=False)
    np.savez_compressed(OUT / "long_rollout_snapshots.npz", snapshot_steps=np.array(SNAPSHOT_KEEP) * 100, **snaps)
    response(sim, seeds).to_csv(OUT / "action_response.csv.gz", index=False)
    cl, cl_snaps = closed_loop(sim, seeds, args.long_steps)
    cl.to_csv(OUT / "closed_loop.csv.gz", index=False)
    np.savez_compressed(OUT / "closed_loop_snapshots.npz", snapshot_steps=np.array(SNAPSHOT_KEEP) * 100, **cl_snaps)
    (OUT / "config.json").write_text(
        json.dumps(
            {
                "recordings": {Path(p).name: ROLES.get(Path(p).name.split(".")[0], "other") for p in args.recordings},
                "horizon": args.horizon,
                "stride": args.stride,
                "long_steps": args.long_steps,
                "hz": 20,
                "seeds": seeds["labels"],
                "scripts": list(SCRIPTS),
            },
            indent=2,
        )
    )
    print("wrote", OUT)


if __name__ == "__main__":
    main()
