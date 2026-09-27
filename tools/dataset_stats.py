"""Compute dataset size statistics for both environments (maintainer tool).

The full datasets are not redistributed. This script reads the original metadata and
writes small, shippable summaries to results/dataset/:

    carla_summary.json       windows, recordings, frames, hours, direction / steer mix
    carla_recordings.csv     one row per recording (frames, windows, direction mix, tags)
    tinyskies_summary.json   sessions, frames, hours, effective frame rate, key mix
    tinyskies_sessions.csv   one row per session (frames, duration, fps, mode, source)

Usage:
    python tools/dataset_stats.py \
        --carla-manifest ../drift-world/.asset-cache/windows_manifest.parquet \
        --tinyskies-recordings ../drift-world/recordings \
        --tinyskies-manifest ../drift-world/tinyskies_session_tfrecords/manifest.json
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "results" / "dataset"
CARLA_HZ = 20  # CARLA synchronous mode, fixed_delta_seconds = 0.05


def _md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def _shares(series):
    counts = series.value_counts()
    return {k: {"count": int(v), "share": round(float(v) / len(series), 4)} for k, v in counts.items()}


def carla(manifest_path):
    df = pd.read_parquet(manifest_path)
    df["recording"] = df["processed_path"].map(lambda p: Path(p).name.replace(".processed.tfrecord", ""))
    per_rec = df.groupby("recording").agg(
        record_date=("record_date", "first"),
        windows=("target_index", "size"),
        frames=("end_index_exclusive", "max"),
        straight=("direction", lambda d: int((d == "straight").sum())),
        left=("direction", lambda d: int((d == "left").sum())),
        right=("direction", lambda d: int((d == "right").sum())),
        tags=("tag_key", "first"),
    )
    per_rec["seconds"] = per_rec["frames"] / CARLA_HZ
    frames = int(per_rec["frames"].sum())
    summary = {
        "source_manifest": str(manifest_path),
        "source_manifest_md5": _md5(manifest_path),
        "dataset_version": sorted(df["dataset_version"].unique().tolist()),
        "record_dates": sorted(df["record_date"].unique().tolist()),
        "recordings": int(len(per_rec)),
        "windows_stack5": int(len(df)),
        "frames": frames,
        "hours_at_20hz": round(frames / CARLA_HZ / 3600, 3),
        "frames_per_recording": {
            "min": int(per_rec["frames"].min()),
            "median": float(per_rec["frames"].median()),
            "max": int(per_rec["frames"].max()),
        },
        "quality_flags": _shares(df["quality_flags"].astype(str)),
        "direction": _shares(df["direction"]),
        "steer_bucket": _shares(df["steer_bucket"]),
        "speed_bucket": _shares(df["speed_bucket"]),
        "tag_key": _shares(df["tag_key"].fillna("")),
        "target_controls_mean": {
            c: round(float(df[f"target_{c}"].mean()), 4) for c in ("throttle", "brake", "steer", "speed")
        },
        "abs_target_steer": {
            "mean": round(float(df["target_steer"].abs().mean()), 4),
            "p90": round(float(df["target_steer"].abs().quantile(0.9)), 4),
            "p99": round(float(df["target_steer"].abs().quantile(0.99)), 4),
            "share_above_0.1": round(float((df["target_steer"].abs() > 0.1).mean()), 4),
        },
    }
    per_rec.sort_index().to_csv(OUT / "carla_recordings.csv")
    (OUT / "carla_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def tinyskies(recordings_dir, manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    rows, keys = [], Counter()
    for session in sorted(Path(recordings_dir).glob("session_*")):
        meta = json.loads((session / "metadata.json").read_text())
        samples = [json.loads(l) for l in (session / "samples.jsonl").read_text().splitlines() if l.strip()]
        if not samples:
            continue
        t = [s["timestamp_ms"] for s in samples]
        duration = (t[-1] - t[0]) / 1000
        gaps = pd.Series(t).diff().dropna() / 1000
        for s in samples:
            down = "".join(k for k in "wasd" if s.get(f"{k}_down"))
            keys[down or "none"] += 1
        rows.append(
            {
                "session": session.name,
                "frames": len(samples),
                "duration_s": round(duration, 2),
                "effective_fps": round((len(samples) - 1) / duration, 2) if duration else None,
                "median_gap_ms": round(float(gaps.median()) * 1000, 1),
                "p95_gap_ms": round(float(gaps.quantile(0.95)) * 1000, 1),
                "nominal_fps": meta.get("fps"),
                "mode": "manual" if meta.get("manual_mode") else "random",
                "source": "localhost" if "localhost" in meta.get("game_url", "") else "tinyskies.vercel.app",
                "has_game_state": any("game_state" in s for s in samples),
                "in_processed_manifest": session.name in manifest["sessions"],
            }
        )
    df = pd.DataFrame(rows).set_index("session")
    total_keys = sum(keys.values())
    summary = {
        "source_recordings": str(recordings_dir),
        "processed_dataset_version": manifest["dataset_version"],
        "processed_schema": manifest["schema"],
        "processed_sessions": len(manifest["sessions"]),
        "processed_frames": int(sum(s["frames"] for s in manifest["sessions"].values())),
        "sessions": int(len(df)),
        "frames": int(df["frames"].sum()),
        "hours": round(float(df["duration_s"].sum()) / 3600, 3),
        "raw_resolution": [1280, 720],
        "model_resolution": manifest["vision_resize"],
        "effective_fps": {
            "median_session": float(df["effective_fps"].median()),
            "min_session": float(df["effective_fps"].min()),
            "max_session": float(df["effective_fps"].max()),
            "overall": round(float((df["frames"] - 1).sum() / df["duration_s"].sum()), 2),
        },
        "nominal_fps": {str(k): int(v) for k, v in df["nominal_fps"].value_counts().items()},
        "mode": {k: int(v) for k, v in df["mode"].value_counts().items()},
        "source": {k: int(v) for k, v in df["source"].value_counts().items()},
        "sessions_with_game_state": int(df["has_game_state"].sum()),
        "keys_down_share": {k: round(v / total_keys, 4) for k, v in keys.most_common()},
        "controls": "[forward (W), brake (S), turn (D - A)], each in {0,1} / {-1,0,1}",
    }
    df.to_csv(OUT / "tinyskies_sessions.csv")
    (OUT / "tinyskies_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--carla-manifest", type=Path)
    parser.add_argument("--tinyskies-recordings", type=Path)
    parser.add_argument("--tinyskies-manifest", type=Path)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.carla_manifest:
        print(json.dumps(carla(args.carla_manifest), indent=2)[:3000])
    if args.tinyskies_recordings:
        print(json.dumps(tinyskies(args.tinyskies_recordings, args.tinyskies_manifest), indent=2))
