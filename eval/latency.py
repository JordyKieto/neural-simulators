"""CPU inference latency of the released simulators with onnxruntime (Python). Writes
results/eval/latency_python.json. Browser latency is measured separately (eval/browser_bench/).

One simulator step, batch 1:
    CARLA      world_transition (the LiDAR encoder runs once per reset, timed separately)
    TinySkies  latent_sim + decoder (the encoder runs only on reset, timed separately)
"""

import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "eval" / "latency_python.json"


def session(path, threads):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    return ort.InferenceSession(str(ROOT / path), opts, providers=["CPUExecutionProvider"])


def bench(fn, warmup=20, iters=300):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000)
    t = np.asarray(times)
    return {"median_ms": round(float(np.median(t)), 3), "p90_ms": round(float(np.percentile(t, 90)), 3), "iters": iters}


def main():
    rng = np.random.default_rng(0)
    results = {}
    for threads in (1, 4):
        enc = session("artifacts/carla/lidar_encoder.onnx", threads)
        sim = session("artifacts/carla/world_transition.onnx", threads)
        carla_feeds = {
            "vision": rng.random((1, 96, 96, 3), np.float32),
            "lidar_latent": rng.random((1, 64), np.float32),
            "telemetry": rng.random((1, 5), np.float32),
            "waypoints": rng.random((1, 25, 3), np.float32),
            "controls": rng.random((1, 3), np.float32),
        }
        t_enc = session("artifacts/tinyskies/encoder.onnx", threads)
        t_dec = session("artifacts/tinyskies/decoder.onnx", threads)
        t_sim = session("artifacts/tinyskies/latent_sim.onnx", threads)
        z = rng.random((1, 6, 6, 64), np.float32)
        tiny_feeds = {
            "latent_history": rng.random((1, 4, 6, 6, 64), np.float32),
            "control_history": rng.random((1, 4, 3), np.float32),
            "controls": rng.random((1, 3), np.float32),
        }
        results[f"threads_{threads}"] = {
            "carla_step (transition)": bench(lambda: sim.run(None, carla_feeds)),
            "carla_reset (lidar_encoder)": bench(lambda: enc.run(None, {"lidar": rng.random((1, 64, 64, 1), np.float32)})),
            "tinyskies_step (latent_sim + decoder)": bench(lambda: (t_sim.run(None, tiny_feeds), t_dec.run(None, {"z": z}))),
            "tinyskies_latent_sim": bench(lambda: t_sim.run(None, tiny_feeds)),
            "tinyskies_decoder": bench(lambda: t_dec.run(None, {"z": z})),
            "tinyskies_reset (encoder)": bench(lambda: t_enc.run(None, {"rgb": rng.random((1, 96, 96, 3), np.float32)})),
        }
    cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    results["machine"] = {"cpu": cpu or platform.processor(), "onnxruntime": ort.__version__, "python": platform.python_version()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
