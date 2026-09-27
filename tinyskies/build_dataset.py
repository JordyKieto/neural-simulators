"""Build processed TinySkies TFRecords from raw recordings (output of tinyskies/record.py).

Each recordings/session_XXXX directory becomes one tf.train.Example holding the whole continuous
session, so trainers choose rollout length and batching themselves:

    frames           uint8  [N, 96, 96, 3]  raw bytes (PIL BOX resize, as in the original builder)
    controls         float32 [N, 3]          [W, S, D - A] from the key state at each screenshot
    num_frames, vision_height, vision_width, vision_channels, control_dim   int64
    dataset_version, session, metadata_json                                 bytes

Sessions are sorted by number and packed 5 per GZIP shard. manifest.json records, for every
session, its shard, record index, frame count and first/last frame index and timestamp; the
trainers and tinyskies/data.py read it.

Usage:
    python tinyskies/build_dataset.py --recordings recordings --out tinyskies_session_tfrecords
    python tinyskies/build_dataset.py --recordings recordings --out DIR --start 65 --end 140
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

DATASET_VERSION = "tinyskies_session_rollout_v3"
CONTROL_DIM = 3


def session_number(path):
    match = re.search(r"session_(\d+)$", Path(path).name)
    return int(match.group(1)) if match else None


def controls_from_sample(sample):
    """[forward (W), brake (S), turn (D - A)]. Accepts the recorder schema and the redistributed sample schema."""
    keys = sample.get("keys") or {k: sample.get(f"{k}_down", False) for k in "wasd"}
    return [float(bool(keys["w"])), float(bool(keys["s"])), float(bool(keys["d"])) - float(bool(keys["a"]))]


def resolve_frame(session_dir, sample):
    """Frame paths are stored relative to the recorder's working directory; fall back to frames/<name>."""
    stored = Path(sample.get("frame_path") or sample["frame"])
    for candidate in (session_dir / "frames" / stored.name, session_dir / stored, stored):
        if candidate.is_file():
            return candidate
    return None


def load_rows(session_dir):
    path = session_dir / "samples.jsonl"
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        sample = json.loads(line)
        frame = resolve_frame(session_dir, sample)
        if frame is None:
            print(f"[warn] {session_dir.name}: missing frame at line {line_number}")
            continue
        rows.append({
            "frame_path": frame,
            "frame_idx": int(sample.get("frame_idx", len(rows))),
            "timestamp_ms": float(sample.get("timestamp_ms", 0.0)),
            "controls": controls_from_sample(sample),
        })
    return sorted(rows, key=lambda r: r["frame_idx"])


def load_frame(path, size):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB").resize((size[1], size[0]), Image.Resampling.BOX), dtype=np.uint8)


def serialize_session(session_dir, rows, size):
    frames = np.stack([load_frame(r["frame_path"], size) for r in rows])
    controls = np.asarray([r["controls"] for r in rows], dtype=np.float32)
    metadata = session_dir / "metadata.json"

    def b(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def i(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))

    feature = {
        "dataset_version": b(DATASET_VERSION.encode()),
        "session": b(session_dir.name.encode()),
        "metadata_json": b(metadata.read_bytes() if metadata.exists() else b"{}"),
        "num_frames": i(frames.shape[0]),
        "vision_height": i(frames.shape[1]),
        "vision_width": i(frames.shape[2]),
        "vision_channels": i(frames.shape[3]),
        "control_dim": i(CONTROL_DIM),
        "frames": b(frames.tobytes()),
        "controls": b(controls.tobytes()),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()


def discover(recordings, start, end, names):
    sessions = [p for p in Path(recordings).glob("session_*") if p.is_dir()]
    sessions.sort(key=lambda p: (session_number(p) is None, session_number(p) or 0, p.name))
    if names:
        wanted = {n if n.startswith("session_") else f"session_{int(n):04d}" for n in names}
        sessions = [p for p in sessions if p.name in wanted]
    if start is not None:
        sessions = [p for p in sessions if (session_number(p) or -1) >= start]
    if end is not None:
        sessions = [p for p in sessions if session_number(p) is not None and session_number(p) <= end]
    return sessions


def shard_name(index, shard_sessions, compression):
    numbers = [n for n in map(session_number, shard_sessions) if n is not None]
    tag = f"-sessions_{min(numbers):04d}_{max(numbers):04d}" if numbers else ""
    return f"{DATASET_VERSION}{tag}-{index:05d}" + (".tfrecord.gz" if compression == "GZIP" else ".tfrecord")


def build(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sessions = discover(args.recordings, args.start, args.end, args.sessions)
    if not sessions:
        raise FileNotFoundError(f"No session_* directories in {args.recordings}")

    manifest = {
        "dataset_version": DATASET_VERSION,
        "created_at_unix": time.time(),
        "compression": args.compression,
        "vision_resize": list(args.resize),
        "control_shape": [CONTROL_DIM],
        "schema": "continuous_session_rgb_controls",
        "features": {"frames": "uint8[num_frames,height,width,3]", "controls": "float32[num_frames,3]"},
        "recording_semantics": "one TFRecord example is one continuous temporally ordered session",
        "shard_size_sessions": args.shard_size,
        "sessions": {},
        "shards": [],
    }
    options = tf.io.TFRecordOptions(compression_type=args.compression)
    writer, shard, in_shard = None, None, 0
    try:
        for idx, session_dir in enumerate(sessions):
            rows = load_rows(session_dir)
            if len(rows) < args.min_frames:
                print(f"[skip] {session_dir.name}: {len(rows)} frames")
                continue
            if writer is None or in_shard >= args.shard_size:
                if writer is not None:
                    writer.close()
                shard = shard_name(len(manifest["shards"]), sessions[idx: idx + args.shard_size], args.compression)
                writer = tf.io.TFRecordWriter(str(out / shard), options=options)
                manifest["shards"].append(shard)
                in_shard = 0
            writer.write(serialize_session(session_dir, rows, args.resize))
            manifest["sessions"][session_dir.name] = {
                "frames": len(rows),
                "shard": shard,
                "record_index_in_shard": in_shard,
                "first_frame_idx": rows[0]["frame_idx"],
                "last_frame_idx": rows[-1]["frame_idx"],
                "first_timestamp_ms": rows[0]["timestamp_ms"],
                "last_timestamp_ms": rows[-1]["timestamp_ms"],
            }
            in_shard += 1
            print(f"[{idx + 1}/{len(sessions)}] {session_dir.name}: {len(rows)} frames -> {shard}", flush=True)
    finally:
        if writer is not None:
            writer.close()

    manifest["total_sessions"] = len(manifest["sessions"])
    manifest["total_frames"] = sum(s["frames"] for s in manifest["sessions"].values())
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[done] {manifest['total_sessions']} sessions, {manifest['total_frames']} frames -> {out / 'manifest.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recordings", type=Path, default=Path("recordings"))
    parser.add_argument("--out", type=Path, default=Path("tinyskies_session_tfrecords"))
    parser.add_argument("--start", type=int, help="first session number (inclusive)")
    parser.add_argument("--end", type=int, help="last session number (inclusive)")
    parser.add_argument("--sessions", nargs="*", help="explicit session names or numbers")
    parser.add_argument("--resize", type=int, nargs=2, default=(96, 96), metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--min-frames", type=int, default=2)
    parser.add_argument("--shard-size", type=int, default=5, help="sessions per shard")
    parser.add_argument("--compression", default="GZIP", choices=("GZIP", "ZLIB", ""))
    build(parser.parse_args())


if __name__ == "__main__":
    main()
