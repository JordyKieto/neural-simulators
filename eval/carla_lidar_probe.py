"""Does the CARLA simulator's LiDAR pathway carry information? Writes results/eval/carla/lidar_probe.json.

The released model encodes the seed LiDAR scan once (64-d latent) and then predicts the next
latent at every step. Training scored the prediction with an MSE against the encoder's latent of
the true future scan. This script separates three questions:

1. Encoding. Does the encoder's latent contain scan information? Ridge probes predict
   scan statistics (occupied cells, left/right asymmetry) from the standardized latent. The
   upper bound is a probe on the 8x8-pooled raw scan.
2. Scale. How much does the latent vary across real scans, and what MSE does a constant
   predictor get? That number is the floor of the training LiDAR loss without any learning.
3. Propagation. Does the predicted next latent track the true future latent during a rollout?
   Per-dimension correlation, measured on the latent's standardized scale.

Usage: python eval/carla_lidar_probe.py --recordings FILE [FILE ...]
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

OUT = ROOT / "results" / "eval" / "carla" / "lidar_probe.json"


def ridge_r2(x, y, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    n = int(0.7 * len(x))
    tr, te = idx[:n], idx[n:]
    xs = (x - x[tr].mean(0)) / (x[tr].std(0) + 1e-12)
    best = -np.inf
    for lam in (1e-2, 1e-1, 1, 10, 100, 1000):
        a = xs[tr]
        w = np.linalg.solve(a.T @ a + lam * np.eye(a.shape[1]), a.T @ (y[tr] - y[tr].mean()))
        p = xs[te] @ w + y[tr].mean()
        best = max(best, 1 - ((y[te] - p) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum())
    return float(best)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recordings", nargs="+", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=40)
    args = parser.parse_args()
    sim = CarlaSimulator()
    recs = [load_recording(p) for p in args.recordings]
    lidar = np.concatenate([r["lidar"] for r in recs])
    z = sim.encode_lidar(lidar)
    empty = sim.encode_lidar(np.zeros((1, 64, 64, 1), np.float32))

    occupied = (lidar > 0).reshape(len(lidar), -1).sum(1).astype(float)
    left, right = lidar[:, :, :32].sum((1, 2, 3)), lidar[:, :, 32:].sum((1, 2, 3))
    asymmetry = (left - right) / (left + right + 1e-6)
    pooled = lidar.reshape(len(lidar), 8, 8, 8, 8).mean((2, 4)).reshape(len(lidar), -1)

    rng = np.random.default_rng(0)
    i, j = rng.integers(0, len(z), (2, 4000))
    result = {
        "frames": int(len(z)),
        "encoding": {
            "ridge_r2_occupied_cells": {"latent": ridge_r2(z, occupied), "raw_bev_pooled": ridge_r2(pooled, occupied)},
            "ridge_r2_left_right_asymmetry": {"latent": ridge_r2(z, asymmetry), "raw_bev_pooled": ridge_r2(pooled, asymmetry)},
        },
        "scale": {
            "input_mean_abs_diff_between_scans_over_mean_abs_scan": float(np.abs(lidar[i] - lidar[j]).mean() / np.abs(lidar).mean()),
            "latent_mean_abs_diff_between_scans_over_mean_abs_latent": float(np.abs(z[i] - z[j]).mean() / np.abs(z).mean()),
            "constant_predictor_mse (floor of the training LiDAR loss)": float(z.var(0).mean()),
            "empty_scan_predictor_mse": float(np.square(z - empty).mean()),
        },
        "propagation": {},
    }

    mu, sd = z.mean(0), z.std(0) + 1e-9
    corr_by_k = {}
    for r in recs:
        zr = sim.encode_lidar(r["lidar"])
        starts = np.arange(0, len(zr) - args.horizon - 1, 10)
        sim.reset(r["vision"][starts], r["lidar"][starts], r["telemetry"][starts], r["waypoints"][starts])
        for k in range(args.horizon):
            state = sim.step(r["controls"][starts + k])
            if k + 1 in (1, 5, 20, args.horizon):
                p, t = (state["lidar_latent"] - mu) / sd, (zr[starts + k + 1] - mu) / sd
                c = [np.corrcoef(p[:, d], t[:, d])[0, 1] for d in range(p.shape[1]) if t[:, d].std() > 0 and p[:, d].std() > 0]
                corr_by_k.setdefault(k + 1, []).append(float(np.mean(c)))
                if k == 0:
                    result["propagation"].setdefault("one_step_error_in_latent_stds", []).append(float(np.abs(p - t).mean()))
    result["propagation"]["mean_per_dim_correlation_pred_vs_true_by_step"] = {k: float(np.mean(v)) for k, v in corr_by_k.items()}
    result["propagation"]["one_step_error_in_latent_stds"] = float(np.mean(result["propagation"]["one_step_error_in_latent_stds"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
