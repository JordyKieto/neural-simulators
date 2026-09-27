"""Train the TinySkies spatial VAE (reproduction of the original `tinyskies_rgb_vae_v1` trainer).

RGB 96x96x3 -> z_mean, z_log_var (6x6x64) -> Sampling -> decoder -> RGB. The encoder and decoder
are the architectures in `tinyskies/model.py`; this file adds the training-only z_log_var head.

    loss = mean |x - x_hat|  +  1e-5 * KL(q(z|x) || N(0, I))     (KL summed per sample)

Adam lr 1e-4, batch 256. The released encoder/decoder are epoch 15 (PROVENANCE.md section 2).

Split: 90/10 by session. Sessions are sorted by name and every 10th one (sorted index % 10 == 9)
is validation. The original trainer applied the same "% 10 == 9" rule to the order in which
sessions came out of a parallel, autotuned TFRecord read, which is machine dependent and cannot
be reconstructed (PROVENANCE.md section 5); this deterministic split is the one used by eval/.

Input: a processed directory from tinyskies/build_dataset.py (manifest.json + GZIP shards).
Outputs (--out): tinyskies_rgb_vae_v1_epoch_XXX.keras (loadable with
tinyskies.model.load_checkpoints), ONNX exports of the runtime encoder (rgb -> z_mean) and decoder
(z -> reconstruction) in <out>/onnx/, config.json and history.csv. W&B logging is used only if
WANDB_API_KEY is set.

Usage:
    python tinyskies/train_vae.py --data tinyskies_session_tfrecords
    python tinyskies/train_vae.py --data ... --resume checkpoints/tinyskies_vae/tinyskies_rgb_vae_v1_epoch_006.keras
    python tinyskies/train_vae.py --smoke --out /tmp/vae_smoke    # 2 steps on samples/tinyskies/
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

import numpy as np
import tensorflow as tf
import keras
from keras import layers

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tinyskies.data import SESSION_FEATURES  # noqa: E402
from tinyskies.model import LATENT_SHAPE, VISION_SHAPE, Sampling, build_decoder, build_encoder  # noqa: E402

SAMPLE_SESSION = ROOT / "samples/tinyskies/session_0005"
KL_WEIGHT = 1e-5
METRICS = ("loss", "recon_loss", "kl_loss")


# ----------------------------------------------------------------------------- shared helpers
# (also used by tinyskies/train_latent_sim.py)


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


def history_fields(metrics, count_name):
    return (
        ["progress/epoch", "progress/batch_step", f"progress/{count_name}", "time/elapsed_sec", "time/epoch_sec"]
        + [f"train/{m}" for m in metrics]
        + [f"val/{m}" for m in metrics]
    )


def read_manifest(processed_dir):
    return json.loads((Path(processed_dir) / "manifest.json").read_text())


def split_sessions(manifest):
    """Sorted session names; sorted index % 10 == 9 -> validation."""
    names = sorted(manifest["sessions"])
    return [n for i, n in enumerate(names) if i % 10 != 9], [n for i, n in enumerate(names) if i % 10 == 9]


def parse_session(proto):
    """Graph-mode parse of one session Example -> (frames float32 [N,96,96,3] in [0,1], controls [N,3])."""
    ex = tf.io.parse_single_example(proto, SESSION_FEATURES)
    n = ex["num_frames"]
    frames = tf.reshape(tf.io.decode_raw(ex["frames"], tf.uint8), tf.stack([n, ex["vision_height"], ex["vision_width"], ex["vision_channels"]]))
    frames = tf.ensure_shape(tf.cast(frames, tf.float32) / 255.0, (None,) + VISION_SHAPE)
    controls = tf.reshape(tf.io.decode_raw(ex["controls"], tf.float32), tf.stack([n, ex["control_dim"]]))
    return frames, tf.ensure_shape(controls, (None, 3))


def session_dataset(processed_dir, manifest, names, shuffle):
    """tf.data.Dataset of (frames, controls), one element per session in `names`."""
    by_shard = {}
    for name in names:
        info = manifest["sessions"][name]
        by_shard.setdefault(info["shard"], []).append(info["record_index_in_shard"])
    shards = sorted(by_shard)
    width = max(max(v) for v in by_shard.values()) + 1
    mask = np.zeros((len(shards), width), dtype=bool)
    for row, shard in enumerate(shards):
        mask[row, by_shard[shard]] = True
    compression = manifest.get("compression") or ""

    def shard_records(path, keep):
        ds = tf.data.TFRecordDataset(path, compression_type=compression).enumerate()
        ds = ds.filter(lambda i, _: tf.logical_and(i < width, tf.gather(keep, tf.minimum(i, width - 1))))
        return ds.map(lambda _, proto: proto)

    ds = tf.data.Dataset.from_tensor_slices(([str(Path(processed_dir) / s) for s in shards], mask))
    if shuffle:
        ds = ds.shuffle(len(shards))
    ds = ds.interleave(shard_records, cycle_length=tf.data.AUTOTUNE, num_parallel_calls=tf.data.AUTOTUNE,
                       deterministic=not shuffle)
    return ds.map(parse_session, num_parallel_calls=tf.data.AUTOTUNE)


def load_sample_session(session_dir=SAMPLE_SESSION):
    """Frames (resized to 96x96, area) and controls of a raw recording, for --smoke runs."""
    rows = [json.loads(line) for line in (Path(session_dir) / "samples.jsonl").read_text().splitlines() if line.strip()]
    frames, controls = [], []
    for row in rows:
        path = Path(session_dir) / row.get("frame", f"frames/{Path(row.get('frame_path', '')).name}")
        image = tf.image.decode_jpeg(tf.io.read_file(str(path)), channels=3)
        frames.append(tf.image.resize(image, VISION_SHAPE[:2], method="area").numpy() / 255.0)
        controls.append(row["controls"] if "controls" in row else [
            float(row["w_down"]), float(row["s_down"]), float(row["d_down"]) - float(row["a_down"])])
    return np.stack(frames).astype(np.float32), np.asarray(controls, dtype=np.float32)


def resume_epoch(path, explicit):
    if explicit is not None:
        return explicit
    match = re.search(r"epoch[_-](\d+)", Path(path).name)
    return int(match.group(1)) if match else 0


def export_function_onnx(fn, spec, path, opset=17):
    """Export a tf.function returning {output_name: tensor}; ONNX I/O names follow the spec and dict keys."""
    import tf2onnx

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tf2onnx.convert.from_function(fn, input_signature=spec, opset=opset, output_path=str(path))
    print(f"[onnx] {path}")


# ----------------------------------------------------------------------------- model and loss


def build_vae():
    """Training VAE. Sub-models are named as tinyskies.model.load_checkpoints expects."""
    runtime = build_encoder()
    features = runtime.get_layer("z_mean").input
    z_mean = runtime.output
    z_log_var = layers.Conv2D(LATENT_SHAPE[-1], 1, padding="same", name="z_log_var")(features)
    z = Sampling(name="z")([z_mean, z_log_var])
    encoder = keras.Model(runtime.input, [z_mean, z_log_var, z], name="tinyskies_rgb_vae_encoder")
    dec = build_decoder()
    decoder = keras.Model(dec.input, dec.output, name="tinyskies_rgb_vae_decoder")

    rgb = keras.Input(VISION_SHAPE, name="rgb")
    z_mean, z_log_var, z = encoder(rgb)
    outputs = {"reconstruction": decoder(z), "z_mean": z_mean, "z_log_var": z_log_var, "z": z}
    return keras.Model(rgb, outputs, name="tinyskies_rgb_vae_v1")


def vae_loss(images, outputs):
    recon = tf.clip_by_value(outputs["reconstruction"], 0.0, 1.0)
    recon_loss = tf.reduce_mean(tf.reduce_mean(tf.abs(tf.clip_by_value(images, 0.0, 1.0) - recon), axis=(1, 2, 3)))
    z_mean, z_log_var = outputs["z_mean"], outputs["z_log_var"]
    kl = -0.5 * tf.reduce_sum(1.0 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=(1, 2, 3))
    kl_loss = tf.reduce_mean(kl)
    return {"loss": recon_loss + KL_WEIGHT * kl_loss, "recon_loss": recon_loss, "kl_loss": kl_loss}


def export_onnx(vae, out_dir, stem, opset=17):
    enc = vae.get_layer("tinyskies_rgb_vae_encoder")
    runtime_encoder = keras.Model(enc.input, enc.output[0])
    decoder = vae.get_layer("tinyskies_rgb_vae_decoder")
    enc_spec = (tf.TensorSpec((None,) + VISION_SHAPE, tf.float32, name="rgb"),)
    dec_spec = (tf.TensorSpec((None,) + LATENT_SHAPE, tf.float32, name="z"),)

    @tf.function(input_signature=enc_spec)
    def encode(rgb):
        return {"z_mean": runtime_encoder(rgb, training=False)}

    @tf.function(input_signature=dec_spec)
    def decode(z):
        return {"reconstruction": decoder(z, training=False)}

    export_function_onnx(encode, enc_spec, Path(out_dir) / f"{stem}_encoder.onnx", opset)
    export_function_onnx(decode, dec_spec, Path(out_dir) / f"{stem}_decoder.onnx", opset)


# ----------------------------------------------------------------------------- training


def frame_dataset(sessions, batch_size, shuffle):
    ds = sessions.flat_map(lambda frames, _: tf.data.Dataset.from_tensor_slices(frames))
    if shuffle:
        ds = ds.shuffle(8192, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def train(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        frames, controls = load_sample_session()
        cut = int(round(0.9 * len(frames)))
        train_sessions = tf.data.Dataset.from_tensors((frames[:cut], controls[:cut]))
        val_sessions = tf.data.Dataset.from_tensors((frames[cut:], controls[cut:]))
        train_frames = cut
    else:
        manifest = read_manifest(args.data)
        train_names, val_names = split_sessions(manifest)
        train_sessions = session_dataset(args.data, manifest, train_names, shuffle=True)
        val_sessions = session_dataset(args.data, manifest, val_names, shuffle=False)
        train_frames = sum(manifest["sessions"][n]["frames"] for n in train_names)
        print(f"[data] sessions train={len(train_names)} val={len(val_names)} train_frames={train_frames}")

    train_ds = frame_dataset(train_sessions, args.batch_size, shuffle=True)
    val_iter = iter(frame_dataset(val_sessions, args.batch_size, shuffle=False).repeat())
    batches_per_epoch = max(1, math.ceil(train_frames / args.batch_size))

    if args.resume:
        vae = keras.models.load_model(args.resume, compile=False, safe_mode=False)
        start_epoch = resume_epoch(args.resume, args.resume_epoch)
    else:
        vae, start_epoch = build_vae(), 0
    optimizer = keras.optimizers.Adam(args.lr)

    config = dict(vars(args), kl_weight=KL_WEIGHT)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    log = RunLog(out_dir, history_fields(METRICS, "frames_seen"), config, args.wandb_project)

    @tf.function
    def train_step(images):
        with tf.GradientTape() as tape:
            losses = vae_loss(images, vae(images, training=True))
        grads = tape.gradient(losses["loss"], vae.trainable_variables)
        optimizer.apply_gradients([(g, v) for g, v in zip(grads, vae.trainable_variables) if g is not None])
        return losses

    @tf.function
    def val_step(images):
        return vae_loss(images, vae(images, training=False))

    step = start_epoch * batches_per_epoch
    frames_seen = step * args.batch_size
    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        epoch_start, batches = time.time(), 0
        for images in train_ds:
            losses = train_step(images)
            step += 1
            batches += 1
            frames_seen += int(images.shape[0])
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
            path = out_dir / f"tinyskies_rgb_vae_v1_epoch_{epoch + 1:03d}.keras"
            vae.save(path)
            print(f"[checkpoint] {path}")
            export_onnx(vae, out_dir / "onnx", path.stem)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, help="processed directory (manifest.json + shards)")
    parser.add_argument("--out", type=Path, default=ROOT / "checkpoints/tinyskies_vae")
    parser.add_argument("--epochs", type=int, default=15, help="the released VAE is epoch 15")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-steps", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0, help="stop each epoch after this many steps (0 = full)")
    parser.add_argument("--resume", type=Path, help=".keras VAE checkpoint to continue from")
    parser.add_argument("--resume-epoch", type=int, help="completed epochs (default: parsed from --resume name)")
    parser.add_argument("--export-onnx-from", type=Path, help="export a .keras VAE to ONNX and exit")
    parser.add_argument("--wandb-project", default="tinyskies-vae")
    parser.add_argument("--smoke", action="store_true", help="2 steps on samples/tinyskies/session_0005")
    args = parser.parse_args()

    if args.export_onnx_from:
        vae = keras.models.load_model(args.export_onnx_from, compile=False, safe_mode=False)
        export_onnx(vae, args.out / "onnx", args.export_onnx_from.stem)
        return
    if args.smoke:
        args.epochs, args.max_steps, args.batch_size, args.val_steps = 1, 2, 16, 1
    elif not args.data:
        parser.error("--data is required (or use --smoke)")
    train(args)


if __name__ == "__main__":
    main()
