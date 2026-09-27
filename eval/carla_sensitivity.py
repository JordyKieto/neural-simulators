"""Which inputs does the released CARLA transition model actually use? Writes
results/eval/carla/sensitivity.json.

For real states from each recording, replace one input at a time and measure how much each
output changes (mean |delta|):
    shuffle     the input is taken from a different real frame (information test)
    zero        the input is zeroed
    action      a fixed alternative action (full throttle / full brake / hard left / hard right)

Also reports how much the LiDAR latent varies across real scans relative to the latent of an
empty scan, which shows whether the LiDAR encoder output carries scan-specific information.

Usage: python eval/carla_sensitivity.py --recordings FILE [FILE ...]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from carla_sim.data import load_recording  # noqa: E402
from carla_sim.runtime import CarlaSimulator  # noqa: E402

OUT = ROOT / "results" / "eval" / "carla" / "sensitivity.json"
OUTPUTS = ("next_vision", "next_lidar_latent", "next_telemetry", "next_waypoints")
ACTIONS = {
    "full_throttle": (1.0, 0.0, 0.0),
    "full_brake": (0.0, 1.0, 0.0),
    "hard_left": (0.3, 0.0, -1.0),
    "hard_right": (0.3, 0.0, 1.0),
}


def run(sim, feeds):
    names = [o.name for o in sim.transition.get_outputs()]
    return dict(zip(names, sim.transition.run(None, feeds)))


def delta(a, b):
    return {k: float(np.abs(a[k] - b[k]).mean()) for k in OUTPUTS}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recordings", nargs="+", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    sim = CarlaSimulator()

    recs = [load_recording(p) for p in args.recordings]
    data = {k: np.concatenate([r[k] for r in recs]) for k in recs[0]}
    latent = sim.encode_lidar(data["lidar"])
    empty = sim.encode_lidar(np.zeros((1, 64, 64, 1), np.float32))[0]

    idx = rng.choice(len(latent), size=min(512, len(latent)), replace=False)
    perm = rng.permutation(len(idx))
    base_feeds = {
        "vision": data["vision"][idx],
        "lidar_latent": latent[idx],
        "telemetry": data["telemetry"][idx],
        "waypoints": data["waypoints"][idx],
        "controls": data["controls"][idx],
    }
    base = run(sim, base_feeds)

    result = {"frames": int(len(idx)), "shuffle": {}, "zero": {}, "action": {}}
    for name in ("vision", "lidar_latent", "telemetry", "waypoints", "controls"):
        result["shuffle"][name] = delta(run(sim, {**base_feeds, name: base_feeds[name][perm]}), base)
        result["zero"][name] = delta(run(sim, {**base_feeds, name: np.zeros_like(base_feeds[name])}), base)
    for name, a in ACTIONS.items():
        controls = np.tile(np.asarray(a, np.float32), (len(idx), 1))
        result["action"][name] = delta(run(sim, {**base_feeds, "controls": controls}), base)

    result["lidar"] = {
        "input_nonzero_fraction": float((data["lidar"] > 0).mean()),
        "input_mean": float(data["lidar"].mean()),
        "latent_std_across_scans_mean_over_dims": float(latent.std(0).mean()),
        "latent_mean_abs": float(np.abs(latent).mean()),
        "mean_abs_diff_real_vs_empty_scan_latent": float(np.abs(latent - empty).mean()),
        "mean_abs_diff_between_random_real_scans": float(np.abs(latent[idx] - latent[idx][perm]).mean()),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
