"""Read processed TinySkies sessions (one tf.train.Example per continuous session, GZIP TFRecords)."""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

SESSION_FEATURES = {
    "frames": tf.io.FixedLenFeature([], tf.string),
    "controls": tf.io.FixedLenFeature([], tf.string),
    "num_frames": tf.io.FixedLenFeature([], tf.int64),
    "vision_height": tf.io.FixedLenFeature([], tf.int64),
    "vision_width": tf.io.FixedLenFeature([], tf.int64),
    "vision_channels": tf.io.FixedLenFeature([], tf.int64),
    "control_dim": tf.io.FixedLenFeature([], tf.int64),
}


def parse_session(proto):
    ex = tf.io.parse_single_example(proto, SESSION_FEATURES)
    n = int(ex["num_frames"])
    shape = (n, int(ex["vision_height"]), int(ex["vision_width"]), int(ex["vision_channels"]))
    frames = tf.reshape(tf.io.decode_raw(ex["frames"], tf.uint8), shape).numpy().astype(np.float32) / 255.0
    controls = tf.reshape(tf.io.decode_raw(ex["controls"], tf.float32), (n, int(ex["control_dim"]))).numpy()
    return frames, controls


def load_sessions(processed_dir, names=None):
    """{session_name: (frames (N,96,96,3) in [0,1], controls (N,3))} from a processed directory."""
    root = Path(processed_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    wanted = names or sorted(manifest["sessions"])
    by_shard = {}
    for name in wanted:
        info = manifest["sessions"][name]
        by_shard.setdefault(info["shard"], []).append((info["record_index_in_shard"], name))
    out = {}
    for shard, items in by_shard.items():
        records = list(tf.data.TFRecordDataset(str(root / shard), compression_type="GZIP"))
        for index, name in items:
            out[name] = parse_session(records[index])
    return {name: out[name] for name in wanted}
