"""Recover Keras checkpoints for the released CARLA model from its ONNX export (maintainer tool).

The original world_v3 epoch-20 Keras checkpoint is not part of this release. The ONNX graph is,
and tf2onnx preserves Keras layer names. This script loads the ONNX weights into the clean
architecture in `carla_sim/model.py`, verifies that both produce the same outputs on real recorded
frames, and writes:

    artifacts/carla/world_transition.keras
    artifacts/carla/lidar_encoder.keras

A successful parity check also proves that `carla_sim/model.py` is the exact released architecture.
"""

import re
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from carla_sim.data import load_recording  # noqa: E402
from carla_sim.model import build_lidar_encoder, build_transition  # noqa: E402

SAMPLE = ROOT / "samples/carla/drive_20260509_110744_f240-359.processed.tfrecord"


def onnx_layers(path):
    """{keras_layer_name: [kernel, bias]} with kernels converted back to Keras layout."""
    out = {}
    for init in onnx.load(str(path)).graph.initializer:
        arr = numpy_helper.to_array(init)
        if arr.dtype != np.float32:
            continue  # shape / axis constants
        layer = re.search(r"/((?:conv2d|dense|next)_[a-z_]*?\d*)_1/", init.name).group(1)
        kind = 0 if arr.ndim > 1 else 1
        if arr.ndim == 4:
            arr = arr.transpose(2, 3, 1, 0)  # OIHW -> HWIO
        out.setdefault(layer, [None, None])[kind] = arr
    return out


def _order(name):
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else 10_000


def assign(model, weights):
    """Match layers by type in creation order (Keras numbers conv2d_* and dense_* separately)."""
    queues = {
        kind: sorted((n for n in weights if n.startswith(kind + "_")), key=_order)
        for kind in ("conv2d", "dense")
    }
    for layer in (l for l in model.layers if l.weights):
        if layer.name.startswith("next_"):
            layer.set_weights(weights[layer.name])
        else:
            kind = "conv2d" if len(layer.weights[0].shape) == 4 else "dense"
            layer.set_weights(weights[queues[kind].pop(0)])
    assert not any(queues.values()), queues


def main():
    data = load_recording(SAMPLE)
    lidar_enc, transition = build_lidar_encoder(), build_transition()
    assign(lidar_enc, onnx_layers(ROOT / "artifacts/carla/lidar_encoder.onnx"))
    assign(transition, onnx_layers(ROOT / "artifacts/carla/world_transition.onnx"))

    enc_sess = ort.InferenceSession(str(ROOT / "artifacts/carla/lidar_encoder.onnx"))
    sim_sess = ort.InferenceSession(str(ROOT / "artifacts/carla/world_transition.onnx"))

    latent_onnx = enc_sess.run(None, {"lidar": data["lidar"]})[0]
    latent_keras = lidar_enc.predict(data["lidar"], verbose=0)
    print(f"lidar_encoder   max |keras - onnx| = {np.abs(latent_onnx - latent_keras).max():.2e}")

    feeds = {
        "vision": data["vision"],
        "lidar_latent": latent_onnx,
        "telemetry": data["telemetry"],
        "waypoints": data["waypoints"],
        "controls": data["controls"],
    }
    names = [o.name for o in sim_sess.get_outputs()]
    onnx_out = dict(zip(names, sim_sess.run(None, feeds)))
    keras_out = transition.predict([feeds[k] for k in ("vision", "lidar_latent", "telemetry", "waypoints", "controls")], verbose=0)
    worst = 0.0
    for k in names:
        diff = np.abs(onnx_out[k] - keras_out[k]).max()
        worst = max(worst, diff)
        print(f"{k:18s} max |keras - onnx| = {diff:.2e}")
    assert worst < 1e-4, "architecture or weight mapping mismatch"

    lidar_enc.save(ROOT / "artifacts/carla/lidar_encoder.keras")
    transition.save(ROOT / "artifacts/carla/world_transition.keras")
    print("saved artifacts/carla/{world_transition,lidar_encoder}.keras")


if __name__ == "__main__":
    main()
