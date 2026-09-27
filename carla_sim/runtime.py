"""Run the released CARLA neural simulator with onnxruntime (mirrors the browser loop in drift-sim)."""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "carla"


def _session(path, threads):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


class CarlaSimulator:
    """Batched state container. The original CARLA simulator is never called.

    reset() encodes the seed LiDAR once; step() applies f(s_t, a_t) and feeds every predicted
    modality back as the next input, exactly as the browser does.
    """

    def __init__(self, artifacts=ARTIFACTS, threads=1):
        self.encoder = _session(Path(artifacts) / "lidar_encoder.onnx", threads)
        self.transition = _session(Path(artifacts) / "world_transition.onnx", threads)
        self.state = None

    def encode_lidar(self, lidar):
        return self.encoder.run(None, {"lidar": np.asarray(lidar, np.float32)})[0]

    def reset(self, vision, lidar, telemetry, waypoints):
        """All arguments batched: (B,96,96,3) in [0,1], (B,64,64,1), (B,5), (B,25,3)."""
        self.state = {
            "vision": np.asarray(vision, np.float32),
            "lidar_latent": self.encode_lidar(lidar),
            "telemetry": np.asarray(telemetry, np.float32),
            "waypoints": np.asarray(waypoints, np.float32),
        }
        return self.state

    def step(self, controls):
        """controls: (B,3) throttle [0,1], brake [0,1], steer [-1,1]."""
        feeds = dict(self.state, controls=np.asarray(controls, np.float32))
        outputs = self.transition.run(None, feeds)
        names = [o.name for o in self.transition.get_outputs()]
        out = dict(zip(names, outputs))
        self.state = {
            "vision": out["next_vision"],
            "lidar_latent": out["next_lidar_latent"],
            "telemetry": out["next_telemetry"],
            "waypoints": out["next_waypoints"],
        }
        return self.state


def load_browser_seeds(path=ROOT / "samples/carla/browser_seeds.json"):
    """The six seed states used by the deployed browser demo, as batched arrays."""
    data = json.loads(Path(path).read_text())
    samples = data["samples"]

    def stack(key, shape):
        return np.stack([np.asarray(s[key], np.float32).reshape(shape) for s in samples])

    vision = stack("vision", (96, 96, 3))
    if vision.max() > 1.5:
        vision = vision / 255.0
    return {
        "labels": [s.get("label") for s in samples],
        "vision": vision,
        "lidar": stack("lidar", (64, 64, 1)),
        "telemetry": stack("telemetry", (5,)),
        "waypoints": stack("waypoints", (25, 3)),
    }
