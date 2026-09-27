"""Train the TinySkies latent simulator (reproduction of the original `tinyskies_latent_sim_v1` trainer).

Frames are encoded once by the frozen VAE encoder (z_mean, 6x6x64). The simulator
(`build_latent_sim` in tinyskies/model.py) is then rolled out H steps in latent space from a
4-step history of real latents:

    window i:  latent_history   z_i .. z_{i+3}
               control_history  [0, c_i, c_{i+1}, c_{i+2}]     (action that led into each latent)
               step k controls  c_{i+3+k}                      target z_{i+4+k},  k = 0..H-1

After each step the prediction is fed back into the history, with scheduled sampling (the true
latent replaces the whole batch's prediction with p = 0.1) and Gaussian feedback noise (std 0.005).

    loss = mean over steps of [ MSE(z_true, z_pred) + 0.01 * mean(delta^2) ]

Adam lr 3e-4, global-norm gradient clipping 1.0, batch 200.

Split: sessions sorted by name, sorted index % 10 == 9 -> validation. This differs from the
original, which applied "% 10 == 9" to the order of a parallel, autotuned TFRecord read, so its
held-out sessions depend on the training machine and cannot be reconstructed (PROVENANCE.md §5).

Released model: a two-stage curriculum (PROVENANCE.md §2). Stage 1 used the epoch-6 VAE, horizon
13 and batch 250 for ~28 epochs. Stage 2 resumed the epoch-26 checkpoint with the final (epoch-15)
VAE, horizon 128 and batch 200, and the released simulator is epoch 40. To reproduce:

    python tinyskies/train_vae.py --data DATA --out ckpt/vae          # saves epochs 1..15
    python tinyskies/train_latent_sim.py --data DATA --out ckpt/sim_stage1 \\
        --vae ckpt/vae/tinyskies_rgb_vae_v1_epoch_006.keras --horizon 13 --batch-size 250 --epochs 28
    python tinyskies/train_latent_sim.py --data DATA --out ckpt/sim_stage2 \\
        --vae ckpt/vae/tinyskies_rgb_vae_v1_epoch_015.keras --horizon 128 --batch-size 200 --epochs 40 \\
        --resume ckpt/sim_stage1/tinyskies_latent_sim_v1_epoch_026.keras

Outputs (--out): tinyskies_latent_sim_v1_epoch_XXX.keras, ONNX exports (latent_history,
control_history, controls -> latent_delta, next_latent) in <out>/onnx/, config.json and
history.csv. W&B logging is used only if WANDB_API_KEY is set.

Usage:
    python tinyskies/train_latent_sim.py --data tinyskies_session_tfrecords [--vae artifacts/tinyskies/vae.keras]
    python tinyskies/train_latent_sim.py --smoke --out /tmp/sim_smoke    # 2 steps, horizon 4, sample session
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import tensorflow as tf
import keras

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tinyskies.model import CONTROLS_DIM, HISTORY, LATENT_SHAPE, VISION_SHAPE, build_latent_sim  # noqa: E402
from tinyskies.train_vae import (  # noqa: E402
    RunLog,
    export_function_onnx,
    history_fields,
    load_sample_session,
    read_manifest,
    resume_epoch,
    session_dataset,
    split_sessions,
)

DELTA_L2_WEIGHT = 0.01
METRICS = ("loss", "latent_loss", "delta_l2_loss", "one_step_latent_loss", "final_step_latent_loss")


def load_frozen_encoder(vae_path):
    """Runtime encoder (rgb -> z_mean) from a training VAE checkpoint, frozen."""
    vae = keras.models.load_model(vae_path, compile=False, safe_mode=False)
    enc = vae.get_layer("tinyskies_rgb_vae_encoder")
    encoder = keras.Model(enc.input, enc.output[0], name="tinyskies_encoder")
    encoder.trainable = False
    return encoder


def window_dataset(sessions, horizon, batch_size, shuffle):
    """(history_frames[B,4], history_controls[B,4,3], controls[B,H,3], target_frames[B,H]) windows."""

    def to_windows(frames, controls):
        count = tf.maximum(tf.shape(frames, out_type=tf.int64)[0] - horizon - HISTORY + 1, 0)
        zero = tf.zeros((1, CONTROLS_DIM), tf.float32)
        return tf.data.Dataset.range(count).map(
            lambda i: (
                frames[i: i + HISTORY],
                tf.concat([zero, controls[i: i + HISTORY - 1]], axis=0),
                controls[i + HISTORY - 1: i + HISTORY - 1 + horizon],
                frames[i + HISTORY: i + HISTORY + horizon],
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    ds = sessions.interleave(to_windows, cycle_length=tf.data.AUTOTUNE, num_parallel_calls=tf.data.AUTOTUNE,
                             deterministic=not shuffle)
    if shuffle:
        ds = ds.shuffle(2048, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def rollout_loss(model, encoder, history_frames, history_controls, controls, target_frames, horizon, training,
                 scheduled_sampling, feedback_noise):
    def encode(frames):
        shape = tf.shape(frames)
        z = encoder(tf.reshape(frames, (-1,) + VISION_SHAPE), training=False)
        return tf.stop_gradient(tf.reshape(z, tf.concat([shape[:2], LATENT_SHAPE], axis=0)))

    latent_history, control_history = encode(history_frames), history_controls
    target_latents = encode(target_frames)
    latent_losses, delta_losses = [], []
    for k in range(horizon):
        current = controls[:, k]
        out = model([latent_history, control_history, current], training=training)
        target = target_latents[:, k]
        latent_losses.append(tf.reduce_mean(tf.square(target - out["next_latent"])))
        delta_losses.append(tf.reduce_mean(tf.square(out["latent_delta"])))

        feedback = out["next_latent"]
        if training and scheduled_sampling > 0.0:
            feedback = tf.cond(tf.random.uniform(()) < scheduled_sampling, lambda: target, lambda: feedback)
        if training and feedback_noise > 0.0:
            feedback = feedback + tf.random.normal(tf.shape(feedback), stddev=feedback_noise)
        latent_history = tf.concat([latent_history[:, 1:], feedback[:, None]], axis=1)
        control_history = tf.concat([control_history[:, 1:], current[:, None]], axis=1)

    latent_loss = tf.add_n(latent_losses) / float(horizon)
    delta_l2_loss = tf.add_n(delta_losses) / float(horizon)
    return {
        "loss": latent_loss + DELTA_L2_WEIGHT * delta_l2_loss,
        "latent_loss": latent_loss,
        "delta_l2_loss": delta_l2_loss,
        "one_step_latent_loss": latent_losses[0],
        "final_step_latent_loss": latent_losses[-1],
    }


def export_onnx(model, out_dir, stem, opset=17):
    spec = (
        tf.TensorSpec((None, HISTORY) + LATENT_SHAPE, tf.float32, name="latent_history"),
        tf.TensorSpec((None, HISTORY, CONTROLS_DIM), tf.float32, name="control_history"),
        tf.TensorSpec((None, CONTROLS_DIM), tf.float32, name="controls"),
    )

    @tf.function(input_signature=spec)
    def step(latent_history, control_history, controls):
        return dict(model([latent_history, control_history, controls], training=False))

    export_function_onnx(step, spec, Path(out_dir) / f"{stem}.onnx", opset)


def train(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_session = lambda n: max(0, n - args.horizon - HISTORY + 1)  # noqa: E731
    if args.smoke:
        frames, controls = load_sample_session()
        train_sessions = val_sessions = tf.data.Dataset.from_tensors((frames, controls))
        train_windows = per_session(len(frames))
    else:
        manifest = read_manifest(args.data)
        train_names, val_names = split_sessions(manifest)
        train_sessions = session_dataset(args.data, manifest, train_names, shuffle=True)
        val_sessions = session_dataset(args.data, manifest, val_names, shuffle=False)
        train_windows = sum(per_session(manifest["sessions"][n]["frames"]) for n in train_names)
        print(f"[data] sessions train={len(train_names)} val={len(val_names)} train_windows={train_windows}")

    train_ds = window_dataset(train_sessions, args.horizon, args.batch_size, shuffle=True)
    val_iter = iter(window_dataset(val_sessions, args.horizon, args.batch_size, shuffle=False).repeat())
    batches_per_epoch = max(1, math.ceil(train_windows / args.batch_size))

    encoder = load_frozen_encoder(args.vae)
    if args.resume:
        model = keras.models.load_model(args.resume, compile=False, safe_mode=False)
        start_epoch = resume_epoch(args.resume, args.resume_epoch)
    else:
        model, start_epoch = build_latent_sim(), 0
    optimizer = keras.optimizers.Adam(args.lr)

    config = dict(vars(args), delta_l2_weight=DELTA_L2_WEIGHT)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    log = RunLog(out_dir, history_fields(METRICS, "windows_seen"), config, args.wandb_project)

    def loss_fn(batch, training):
        return rollout_loss(model, encoder, *batch, args.horizon, training, args.scheduled_sampling, args.feedback_noise)

    @tf.function
    def train_step(batch):
        with tf.GradientTape() as tape:
            losses = loss_fn(batch, training=True)
        grads = tape.gradient(losses["loss"], model.trainable_variables)
        pairs = [(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None]
        if args.grad_clip > 0:
            clipped, _ = tf.clip_by_global_norm([g for g, _ in pairs], args.grad_clip)
            pairs = list(zip(clipped, [v for _, v in pairs]))
        optimizer.apply_gradients(pairs)
        return losses

    @tf.function
    def val_step(batch):
        return loss_fn(batch, training=False)

    step = start_epoch * batches_per_epoch
    windows_seen = step * args.batch_size
    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        epoch_start, batches = time.time(), 0
        for batch in train_ds:
            losses = train_step(batch)
            step += 1
            batches += 1
            windows_seen += int(batch[0].shape[0])
            if step == 1 or step % args.log_every == 0 or args.smoke:
                val = {m: 0.0 for m in METRICS}
                for _ in range(args.val_steps):
                    val_losses = val_step(next(val_iter))
                    for m in METRICS:
                        val[m] += float(val_losses[m]) / args.val_steps
                row = {"progress/epoch": epoch + 1, "progress/batch_step": step, "progress/windows_seen": windows_seen,
                       "time/elapsed_sec": time.time() - t0}
                row.update({f"train/{m}": float(losses[m]) for m in METRICS})
                row.update({f"val/{m}": v for m, v in val.items()})
                log.log(row, step)
            if args.max_steps and batches >= args.max_steps:
                break
        if batches == 0:
            raise RuntimeError("Training dataset produced zero batches (sessions shorter than horizon + 4?).")
        log.log({"progress/epoch": epoch + 1, "time/epoch_sec": time.time() - epoch_start}, step)

        if (epoch + 1) % args.checkpoint_every == 0 or epoch + 1 == args.epochs:
            path = out_dir / f"tinyskies_latent_sim_v1_epoch_{epoch + 1:03d}.keras"
            model.save(path)
            print(f"[checkpoint] {path}")
            export_onnx(model, out_dir / "onnx", path.stem)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, help="processed directory (manifest.json + shards)")
    parser.add_argument("--vae", type=Path, default=ROOT / "artifacts/tinyskies/vae.keras",
                        help="training VAE .keras (frozen; only its z_mean encoder is used)")
    parser.add_argument("--out", type=Path, default=ROOT / "checkpoints/tinyskies_latent_sim")
    parser.add_argument("--epochs", type=int, default=40, help="total epochs (the released simulator is epoch 40)")
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--scheduled-sampling", type=float, default=0.1)
    parser.add_argument("--feedback-noise", type=float, default=0.005)
    parser.add_argument("--val-steps", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0, help="stop each epoch after this many steps (0 = full)")
    parser.add_argument("--resume", type=Path, help="latent-sim .keras checkpoint to continue from")
    parser.add_argument("--resume-epoch", type=int, help="completed epochs (default: parsed from --resume name)")
    parser.add_argument("--export-onnx-from", type=Path, help="export a latent-sim .keras to ONNX and exit")
    parser.add_argument("--wandb-project", default="tinyskies-latent-sim")
    parser.add_argument("--smoke", action="store_true", help="2 steps, horizon 4, on samples/tinyskies/session_0005")
    args = parser.parse_args()

    if args.export_onnx_from:
        model = keras.models.load_model(args.export_onnx_from, compile=False, safe_mode=False)
        export_onnx(model, args.out / "onnx", args.export_onnx_from.stem)
        return
    if args.smoke:
        args.epochs, args.max_steps, args.batch_size, args.val_steps, args.horizon = 1, 2, 8, 1, 4
    elif not args.data:
        parser.error("--data is required (or use --smoke)")
    train(args)


if __name__ == "__main__":
    main()
