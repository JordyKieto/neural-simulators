"""Run the released TinySkies neural simulator with onnxruntime (mirrors the browser loop in drift-sim)."""

from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "tinyskies"
HISTORY = 4


def _session(path, threads):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


class TinySkiesSimulator:
    """Batched latent-space simulator. The game is never called after the seed frame(s)."""

    def __init__(self, artifacts=ARTIFACTS, threads=1):
        self.encoder = _session(Path(artifacts) / "encoder.onnx", threads)
        self.decoder = _session(Path(artifacts) / "decoder.onnx", threads)
        self.sim = _session(Path(artifacts) / "latent_sim.onnx", threads)
        self.latents = None
        self.controls = None

    def encode(self, rgb):
        return self.encoder.run(None, {"rgb": np.asarray(rgb, np.float32)})[0]

    def decode(self, z):
        return self.decoder.run(None, {"z": np.asarray(z, np.float32)})[0]

    def reset(self, frames, control_history=None):
        """frames: (B,96,96,3) seeds a history by repetition (as the browser does), or
        (B,T<=4,96,96,3) seeds it from real frames. control_history: (B,4,3), zeros by default."""
        frames = np.asarray(frames, np.float32)
        if frames.ndim == 4:
            z = self.encode(frames)
            self.latents = np.repeat(z[:, None], HISTORY, axis=1)
        else:
            b, t = frames.shape[:2]
            z = self.encode(frames.reshape((-1,) + frames.shape[2:])).reshape((b, t) + (6, 6, 64))
            pad = np.repeat(z[:, :1], HISTORY - t, axis=1)
            self.latents = np.concatenate([pad, z], axis=1)[:, -HISTORY:]
        b = self.latents.shape[0]
        self.controls = (
            np.zeros((b, HISTORY, 3), np.float32) if control_history is None else np.asarray(control_history, np.float32)
        )
        return self.latents[:, -1]

    def step(self, controls):
        """controls: (B,3) forward {0,1}, brake {0,1}, turn {-1,0,1}. Returns the next latent."""
        controls = np.asarray(controls, np.float32)
        out = dict(
            zip(
                [o.name for o in self.sim.get_outputs()],
                self.sim.run(
                    None,
                    {"latent_history": self.latents, "control_history": self.controls, "controls": controls},
                ),
            )
        )
        z = out["next_latent"]
        self.latents = np.concatenate([self.latents[:, 1:], z[:, None]], axis=1)
        self.controls = np.concatenate([self.controls[:, 1:], controls[:, None]], axis=1)
        return z
