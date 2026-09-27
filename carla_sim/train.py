"""Train the CARLA world_v3 transition model (reproduction of the original training code).

The model (`carla_sim/model.py`) is trained with an 8-step open-loop rollout: from a real start state
the transition is applied to the recorded controls, and every prediction is fed back as the next
input. The LiDAR encoder is trained jointly through the initial latent.

Loss per rollout step, averaged over the horizon:
    1.0  * RGB          (0.7 L1 + 0.3 MSE) + 0.1 SSIM loss + 0.1 "perceptual"
                        (0.4 half-scale L1 + 0.3 quarter-scale L1 + 0.3 Sobel-edge L1)
    0.25 * LiDAR latent MSE vs stop_gradient(encoder(true future LiDAR))
    0.25 * telemetry MSE
    0.25 * waypoints MSE
Adam, lr 1e-4, batch 500. Split: manifest windows sorted by processed path, first 80% train.

The original config declared scheduled sampling (0.1) and feedback noise, but the training loop
never used them. This reproduction keeps that behaviour: --scheduled-sampling and --feedback-noise
exist but default to 0.

Inputs:
    --manifest   windows parquet (columns processed_path, start_index, quality_flags)
    --data-dir   directory of processed TFRecords (format in carla_sim/data.py); manifest paths
                 that do not exist are resolved to <data-dir>/<basename>

Outputs (--out): world_v3_epoch_XXX.keras every --checkpoint-every epochs, ONNX exports
(transition + LiDAR encoder, opset 17, same I/O names as artifacts/carla/) in <out>/onnx/,
config.json and history.csv. W&B logging is used only if WANDB_API_KEY is set.

Usage:
    python carla_sim/train.py --manifest windows_manifest.parquet --data-dir recordings_processed
    python carla_sim/train.py ... --resume checkpoints/carla/world_v3_epoch_010.keras
    python carla_sim/train.py --export-onnx-from checkpoints/carla/world_v3_epoch_020.keras
    python carla_sim/train.py --smoke --out /tmp/carla_smoke    # 2 steps on samples/carla/
"""

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import tensorflow as tf
import keras

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from carla_sim.data import FEATURES  # noqa: E402
from carla_sim.model import (  # noqa: E402
    CONTROLS_DIM,
    LIDAR_LATENT_DIM,
    LIDAR_SHAPE,
    TELEMETRY_DIM,
    VISION_SHAPE,
    WAYPOINTS_SHAPE,
    build_lidar_encoder,
    build_transition,
)

SAMPLE = ROOT / "samples/carla/drive_20260509_110744_f240-359.processed.tfrecord"

RGB_L1_MIX, RGB_MSE_MIX = 0.7, 0.3
RGB_SSIM_WEIGHT, RGB_PERCEPTUAL_WEIGHT = 0.1, 0.1
LOSS_WEIGHTS = {"rgb": 1.0, "lidar": 0.25, "telemetry": 0.25, "waypoints": 0.25}
METRICS = ("loss", "rgb_loss", "lidar_loss", "telemetry_loss", "waypoints_loss", "one_step_loss", "final_step_loss")
HISTORY_FIELDS = (
    ["progress/epoch", "progress/batch_step", "progress/frames_seen", "time/elapsed_sec", "time/epoch_sec"]
    + [f"train/{m}" for m in METRICS]
    + [f"val/{m}" for m in METRICS]
)


class RunLog:
    """stdout + history.csv, and W&B only when WANDB_API_KEY is set."""

    def __init__(self, out_dir, fields, config, wandb_project):
        self.path = Path(out_dir) / "history.csv"
        self.fields = ["_step"] + list(fields)
        self.wandb = None
        if os.environ.get("WANDB_API_KEY"):
            import wandb

            self.wandb = wandb.init(project=wandb_project, config=config)

    def log(self, metrics, step):
        print(f"step={step} " + " ".join(f"{k}={v:.6g}" for k, v in metrics.items()), flush=True)
        new = not self.path.exists()
        with self.path.open("a", newline="") as f:
            writer = csv.DictWriter(f, self.fields, extrasaction="ignore")
            if new:
                writer.writeheader()
            writer.writerow({"_step": step, **metrics})
        if self.wandb is not None:
            self.wandb.log(metrics, step=step)


# ----------------------------------------------------------------------------- data


def load_windows(manifest_path, data_dir):
    """Manifest rows with quality_flags == 'ok', with processed_path resolved to local files."""
    df = pd.read_parquet(manifest_path)
    if "quality_flags" in df.columns:
        df = df[df["quality_flags"] == "ok"]
    if df.empty:
        raise RuntimeError("Manifest filtering produced no rows.")

    def resolve(path):
        if Path(path).exists():
            return str(path)
        local = Path(data_dir) / Path(str(path).replace("\\", "/")).name
        if not local.exists():
            raise FileNotFoundError(f"Processed recording not found: {path} (looked in {data_dir})")
        return str(local)

    df = df.copy()
    df["processed_path"] = df["processed_path"].astype(str).map(resolve)
    return df


def smoke_windows(horizon):
    """In-memory manifest over the bundled sample recording."""
    n = sum(1 for _ in tf.data.TFRecordDataset(str(SAMPLE)))
    return pd.DataFrame({"processed_path": str(SAMPLE), "start_index": range(n - horizon), "quality_flags": "ok"})


def windows_to_ranges(df):
    """Merge consecutive windows of one recording into {path, start, num_windows} ranges."""
    ranges, current = [], None
    for row in df.sort_values(["processed_path", "start_index"]).itertuples(index=False):
        path, start = str(row.processed_path), int(row.start_index)
        if current is not None and current["path"] == path and current["start"] + current["num_windows"] == start:
            current["num_windows"] += 1
        else:
            current = {"path": path, "start": start, "num_windows": 1}
            ranges.append(current)
    return ranges


def count_windows(ranges):
    return int(sum(r["num_windows"] for r in ranges))


def split_ranges(ranges, train_split):
    """First `train_split` of the windows (sorted by path) are train; one range may be cut in two."""
    target = int(round(count_windows(ranges) * train_split))
    train, val, used = [], [], 0
    for r in ranges:
        n, remain = r["num_windows"], target - used
        if remain <= 0:
            val.append(r)
        elif remain >= n:
            train.append(r)
            used += n
        else:
            train.append({**r, "num_windows": remain})
            val.append({**r, "start": r["start"] + remain, "num_windows": n - remain})
            used += remain
    return train or list(ranges), val or list(train)


def parse_example(proto):
    ex = tf.io.parse_single_example(proto, FEATURES)
    return (
        tf.cast(tf.reshape(tf.io.decode_raw(ex["vision"], tf.uint8), VISION_SHAPE), tf.float32) / 255.0,
        tf.reshape(tf.io.decode_raw(ex["lidar"], tf.float32), LIDAR_SHAPE),
        tf.reshape(tf.io.decode_raw(ex["telemetry"], tf.float32), (TELEMETRY_DIM,)),
        tf.reshape(tf.io.decode_raw(ex["waypoints"], tf.float32), WAYPOINTS_SHAPE),
        tf.reshape(tf.io.decode_raw(ex["controls"], tf.float32), (CONTROLS_DIM,)),
    )


def make_dataset(ranges, horizon, batch_size, shuffle):
    """Yields (start_state, controls[B,H,3], targets[B,H,...]) from sliding windows of H+1 frames."""
    seq_len = horizon + 1

    def to_sequence(*field_datasets):
        seq = tf.data.Dataset.zip(tuple(d.batch(seq_len, drop_remainder=True) for d in field_datasets))
        return seq.map(
            lambda vision, lidar, telemetry, waypoints, controls: (
                {"vision": vision[0], "lidar": lidar[0], "telemetry": telemetry[0], "waypoints": waypoints[0]},
                controls[:-1],
                {"vision": vision[1:], "lidar": lidar[1:], "telemetry": telemetry[1:], "waypoints": waypoints[1:]},
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    def range_dataset(path, start, num_windows):
        ds = tf.data.TFRecordDataset(path).skip(start).take(num_windows + horizon)
        ds = ds.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.window(seq_len, shift=1, drop_remainder=True).flat_map(to_sequence)

    ds = tf.data.Dataset.from_tensor_slices(
        (
            tf.constant([r["path"] for r in ranges]),
            tf.constant([r["start"] for r in ranges], dtype=tf.int64),
            tf.constant([r["num_windows"] for r in ranges], dtype=tf.int64),
        )
    )
    if shuffle:
        ds = ds.shuffle(len(ranges))
    ds = ds.interleave(
        range_dataset, cycle_length=tf.data.AUTOTUNE, num_parallel_calls=tf.data.AUTOTUNE, deterministic=not shuffle
    )
    if shuffle:
        ds = ds.shuffle(2048)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ----------------------------------------------------------------------------- model and loss


def build_model():
    """Training graph: LiDAR encoder -> initial latent -> transition. Returns (model, encoder, transition)."""
    encoder, transition = build_lidar_encoder(), build_transition()
    vision = keras.Input(VISION_SHAPE, name="vision")
    lidar = keras.Input(LIDAR_SHAPE, name="lidar")
    telemetry = keras.Input((TELEMETRY_DIM,), name="telemetry")
    waypoints = keras.Input(WAYPOINTS_SHAPE, name="waypoints")
    controls = keras.Input((CONTROLS_DIM,), name="controls")
    out = transition([vision, encoder(lidar), telemetry, waypoints, controls])
    model = keras.Model([vision, lidar, telemetry, waypoints, controls], out, name="world_v3_rgb_action_state")
    return model, encoder, transition


def load_model(path):
    model = keras.models.load_model(path, compile=False, safe_mode=False)
    return model, model.get_layer("lidar_encoder"), model.get_layer("world_v3_transition")


def mse(a, b):
    return tf.reduce_mean(tf.square(a - b))


def rgb_perceptual_loss(target, pred):
    half_t = tf.nn.avg_pool2d(target, 2, 2, "SAME")
    half_p = tf.nn.avg_pool2d(pred, 2, 2, "SAME")
    quarter_t = tf.nn.avg_pool2d(half_t, 2, 2, "SAME")
    quarter_p = tf.nn.avg_pool2d(half_p, 2, 2, "SAME")
    edges = tf.abs(tf.image.sobel_edges(target) - tf.image.sobel_edges(pred))
    return (
        0.4 * tf.reduce_mean(tf.abs(half_t - half_p))
        + 0.3 * tf.reduce_mean(tf.abs(quarter_t - quarter_p))
        + 0.3 * tf.reduce_mean(edges)
    )


def rgb_loss(target, pred):
    target = tf.clip_by_value(target, 0.0, 1.0)
    pred = tf.clip_by_value(pred, 0.0, 1.0)
    recon = RGB_L1_MIX * tf.reduce_mean(tf.abs(target - pred)) + RGB_MSE_MIX * mse(target, pred)
    ssim_loss = tf.reduce_mean((1.0 - tf.image.ssim(target, pred, max_val=1.0)) * 0.5)
    return recon + RGB_SSIM_WEIGHT * ssim_loss + RGB_PERCEPTUAL_WEIGHT * rgb_perceptual_loss(target, pred)


def rollout_loss(encoder, transition, start, controls, targets, horizon, training,
                 scheduled_sampling=0.0, feedback_noise=0.0):
    """Open-loop rollout of `horizon` steps; predictions are fed back as the next input."""
    flat_lidar = tf.reshape(targets["lidar"], (-1,) + LIDAR_SHAPE)
    target_latents = tf.stop_gradient(
        tf.reshape(encoder(flat_lidar, training=False), (-1, horizon, LIDAR_LATENT_DIM))
    )
    state = [start["vision"], encoder(start["lidar"], training=training), start["telemetry"], start["waypoints"]]
    per_step = {"rgb": [], "lidar": [], "telemetry": [], "waypoints": []}
    for k in range(horizon):
        out = transition([*state, controls[:, k]], training=training)
        pred = [out["next_vision"], out["next_lidar_latent"], out["next_telemetry"], out["next_waypoints"]]
        true = [targets["vision"][:, k], target_latents[:, k], targets["telemetry"][:, k], targets["waypoints"][:, k]]
        per_step["rgb"].append(rgb_loss(true[0], pred[0]))
        per_step["lidar"].append(mse(true[1], pred[1]))
        per_step["telemetry"].append(mse(true[2], pred[2]))
        per_step["waypoints"].append(mse(true[3], pred[3]))

        state = pred
        if training and scheduled_sampling > 0.0:  # not used by the released model
            state = tf.cond(tf.random.uniform(()) < scheduled_sampling, lambda: true, lambda: pred)
        if training and feedback_noise > 0.0:  # not used by the released model
            state = [s + tf.random.normal(tf.shape(s), stddev=feedback_noise) for s in state]
            state[0] = tf.clip_by_value(state[0], 0.0, 1.0)

    losses = {f"{name}_loss": tf.add_n(values) / float(horizon) for name, values in per_step.items()}
    losses["loss"] = tf.add_n([LOSS_WEIGHTS[name] * losses[f"{name}_loss"] for name in per_step])
    losses["one_step_loss"] = per_step["rgb"][0]
    losses["final_step_loss"] = per_step["rgb"][-1]
    return losses


# ----------------------------------------------------------------------------- export


def export_onnx(encoder, transition, out_dir, stem, opset=17):
    """transition: vision, lidar_latent, telemetry, waypoints, controls -> next_* ; encoder: lidar -> lidar_latent."""
    import tf2onnx

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = (
        tf.TensorSpec((None,) + VISION_SHAPE, tf.float32, name="vision"),
        tf.TensorSpec((None, LIDAR_LATENT_DIM), tf.float32, name="lidar_latent"),
        tf.TensorSpec((None, TELEMETRY_DIM), tf.float32, name="telemetry"),
        tf.TensorSpec((None,) + WAYPOINTS_SHAPE, tf.float32, name="waypoints"),
        tf.TensorSpec((None, CONTROLS_DIM), tf.float32, name="controls"),
    )

    @tf.function(input_signature=spec)
    def transition_fn(vision, lidar_latent, telemetry, waypoints, controls):
        return dict(transition([vision, lidar_latent, telemetry, waypoints, controls], training=False))

    lidar_spec = (tf.TensorSpec((None,) + LIDAR_SHAPE, tf.float32, name="lidar"),)

    @tf.function(input_signature=lidar_spec)
    def encoder_fn(lidar):
        return {"lidar_latent": encoder(lidar, training=False)}

    paths = {"transition": out_dir / f"{stem}_transition.onnx", "lidar_encoder": out_dir / f"{stem}_lidar_encoder.onnx"}
    tf2onnx.convert.from_function(transition_fn, input_signature=spec, opset=opset, output_path=str(paths["transition"]))
    tf2onnx.convert.from_function(encoder_fn, input_signature=lidar_spec, opset=opset, output_path=str(paths["lidar_encoder"]))
    print(f"[onnx] {paths['transition']}\n[onnx] {paths['lidar_encoder']}")
    return paths


# ----------------------------------------------------------------------------- training


def train(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    windows = smoke_windows(args.horizon) if args.smoke and not args.manifest else load_windows(args.manifest, args.data_dir)
    train_ranges, val_ranges = split_ranges(windows_to_ranges(windows), args.train_split)
    print(f"[data] windows={len(windows)} train={count_windows(train_ranges)} val={count_windows(val_ranges)}")

    train_ds = make_dataset(train_ranges, args.horizon, args.batch_size, shuffle=True)
    val_iter = iter(make_dataset(val_ranges, args.horizon, args.batch_size, shuffle=False).repeat())
    batches_per_epoch = max(1, math.ceil(count_windows(train_ranges) / args.batch_size))

    if args.resume:
        model, encoder, transition = load_model(args.resume)
        match = re.search(r"epoch[_-](\d+)", Path(args.resume).name)
        start_epoch = args.resume_epoch if args.resume_epoch is not None else int(match.group(1)) if match else 0
    else:
        model, encoder, transition = build_model()
        start_epoch = 0
    optimizer = keras.optimizers.Adam(args.lr)

    config = {k: v for k, v in vars(args).items()}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    log = RunLog(out_dir, HISTORY_FIELDS, config, args.wandb_project)

    def loss_fn(batch, training):
        start, controls, targets = batch
        return rollout_loss(encoder, transition, start, controls, targets, args.horizon, training,
                            args.scheduled_sampling, args.feedback_noise)

    @tf.function
    def train_step(batch):
        with tf.GradientTape() as tape:
            losses = loss_fn(batch, training=True)
        grads = tape.gradient(losses["loss"], model.trainable_variables)
        optimizer.apply_gradients([(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None])
        return losses

    @tf.function
    def val_step(batch):
        return loss_fn(batch, training=False)

    step = start_epoch * batches_per_epoch
    frames_seen = step * args.batch_size
    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        epoch_start, batches = time.time(), 0
        for batch in train_ds:
            losses = train_step(batch)
            step += 1
            batches += 1
            frames_seen += int(batch[0]["vision"].shape[0])
            if step == 1 or step % args.log_every == 0 or args.smoke:
                val = {m: 0.0 for m in METRICS}
                for _ in range(args.val_steps):
                    val_losses = val_step(next(val_iter))
                    for m in METRICS:
                        val[m] += float(val_losses[m]) / args.val_steps
                row = {"progress/epoch": epoch + 1, "progress/batch_step": step, "progress/frames_seen": frames_seen,
                       "time/elapsed_sec": time.time() - t0}
                row.update({f"train/{m}": float(losses[m]) for m in METRICS})
                row.update({f"val/{m}": v for m, v in val.items()})
                log.log(row, step)
            if args.max_steps and batches >= args.max_steps:
                break
        if batches == 0:
            raise RuntimeError("Training dataset produced zero batches.")
        log.log({"progress/epoch": epoch + 1, "time/epoch_sec": time.time() - epoch_start}, step)

        if (epoch + 1) % args.checkpoint_every == 0 or epoch + 1 == args.epochs:
            path = out_dir / f"world_v3_epoch_{epoch + 1:03d}.keras"
            model.save(path)
            print(f"[checkpoint] {path}")
            export_onnx(encoder, transition, out_dir / "onnx", path.stem)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, help="windows manifest parquet")
    parser.add_argument("--data-dir", type=Path, default=Path("recordings_processed"))
    parser.add_argument("--out", type=Path, default=ROOT / "checkpoints/carla")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--val-steps", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=0, help="stop each epoch after this many steps (0 = full)")
    parser.add_argument("--scheduled-sampling", type=float, default=0.0,
                        help="probability of feeding the true state back (declared but unused originally)")
    parser.add_argument("--feedback-noise", type=float, default=0.0,
                        help="Gaussian std added to fed-back predictions (declared but unused originally)")
    parser.add_argument("--resume", type=Path, help="combined .keras checkpoint to continue from")
    parser.add_argument("--resume-epoch", type=int, help="completed epochs (default: parsed from --resume name)")
    parser.add_argument("--export-onnx-from", type=Path, help="export a .keras checkpoint to ONNX and exit")
    parser.add_argument("--wandb-project", default="carla-world-v3")
    parser.add_argument("--smoke", action="store_true", help="2 steps on samples/carla/ (no dataset needed)")
    args = parser.parse_args()

    if args.export_onnx_from:
        _, encoder, transition = load_model(args.export_onnx_from)
        export_onnx(encoder, transition, args.out / "onnx", args.export_onnx_from.stem)
        return
    if args.smoke:
        args.epochs, args.max_steps, args.batch_size, args.val_steps = 1, 2, 4, 1
    elif not args.manifest:
        parser.error("--manifest is required (or use --smoke)")
    train(args)


if __name__ == "__main__":
    main()
