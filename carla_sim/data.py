"""Read processed CARLA recordings (one tf.train.Example per 20 Hz frame, raw numpy bytes per feature)."""

import numpy as np
import tensorflow as tf

HZ = 20
FEATURES = {
    name: tf.io.FixedLenFeature([], tf.string)
    for name in ("vision", "lidar", "telemetry", "waypoints", "controls")
}


def parse_example(proto):
    ex = tf.io.parse_single_example(proto, FEATURES)
    return {
        "vision": tf.cast(tf.reshape(tf.io.decode_raw(ex["vision"], tf.uint8), (96, 96, 3)), tf.float32) / 255.0,
        "lidar": tf.reshape(tf.io.decode_raw(ex["lidar"], tf.float32), (64, 64, 1)),
        "telemetry": tf.reshape(tf.io.decode_raw(ex["telemetry"], tf.float32), (5,)),
        "waypoints": tf.reshape(tf.io.decode_raw(ex["waypoints"], tf.float32), (25, 3)),
        "controls": tf.reshape(tf.io.decode_raw(ex["controls"], tf.float32), (3,)),
    }


def load_recording(path):
    """Return a dict of stacked float32 arrays, one row per frame."""
    frames = [parse_example(p) for p in tf.data.TFRecordDataset(str(path))]
    return {k: np.stack([f[k].numpy() for f in frames]).astype(np.float32) for k in frames[0]}
