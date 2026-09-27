# Provenance

This file traces every shipped artifact, sample and number to its origin. The claim audit (`paper/CLAIMS.md`) cites it. Evidence classes follow the audit:

- **M**: measured, with a recoverable artifact
- **Q**: qualitative (screenshot, video, author observation)
- **I**: inferred from M/Q
- **A**: author-confirmed
- **U**: unverified

Original repositories. The paper does not need any of them.

| Name | Role |
|---|---|
| `drift` | Training code for CARLA world models v1–v4 and TinySkies models. Private. |
| `drift-world` | TinySkies recorder, recordings and dataset builder. Earlier version of the demo site. |
| [`drift-sim`](https://github.com/JordyKieto/drift-sim) | Deployed browser simulator: <https://drift-sim-production.up.railway.app/> |
| [`tinyskies`](https://github.com/dannylimanseta/tinyskies) | The game, by Danny Limanseta. Unmodified for the Vercel recordings. |

W&B project names and run IDs appear only in `results/*/run.json`. The paper does not link to W&B.

---

## 1. Deployed models (what the browser runs)

All ONNX files were exported with tf2onnx 1.17.0, opset 17, fp32. Parameter counts are summed over ONNX initializers. Full signatures are in [`artifacts/MANIFEST.json`](artifacts/MANIFEST.json).

| Artifact | md5 | Params | Origin | Class |
|---|---|---|---|---|
| `artifacts/carla/world_transition.onnx` | `9ce807ab…2824` | 1,551,854 | `world_v3` epoch 20 (`s3://…/checkpoints/world_v3/world_v3_epoch_020_transition.onnx`) | M |
| `artifacts/carla/lidar_encoder.onnx` | `45a5af83…2318` | 27,462 | `world_v3` epoch 20 | M |
| `artifacts/tinyskies/encoder.onnx` | `d43a1d50…3365` | 660,480 | `tinyskies_rgb_vae_v1`, weights after epoch 14 (file `epoch_015`) | M |
| `artifacts/tinyskies/decoder.onnx` | `8c570ddc…19cb` | 1,282,933 | `tinyskies_rgb_vae_v1`, weights after epoch 14 (file `epoch_015`) | M |
| `artifacts/tinyskies/latent_sim.onnx` | `368f9a19…b4a8` | 1,842,071 | `tinyskies_latent_sim_v1` epoch 40 | M |
| `artifacts/carla/{world_transition,lidar_encoder}.keras` | see `MANIFEST.json` | same as ONNX | rebuilt from the ONNX weights by `tools/carla_onnx_to_keras.py` | M |

Runtime totals:
- **CARLA:** 1,579,316 params, about 6.4 MB.
- **TinySkies:** 3,785,484 params, about 15.2 MB.

Each file is byte-identical (same md5) to `drift-sim/public/models/*`, which is what the production site serves.

**CARLA deployment history (drift-sim git):**
- Added in `5f8d7b5` (2026-05-16) and never changed since.
- The same md5 appears in `drift-world/.asset-cache/world_v3_epoch_020_*.onnx`, downloaded 2026-05-13 23:54 PDT from the S3 URIs hard-coded in `drift-world/scripts/prepare_assets.py`.

**TinySkies deployment history (drift-sim git):**
- `3e84c65` (05-18): `tinyskies_world_v2_epoch_950`.
- `7023051` (05-24): latent pipeline with sim epoch 10.
- `dc10915` (05-25 17:33 PDT): encoder and decoder re-exported, sim epoch 40. This is the current deployment.

### Verification
- **CARLA architecture.** `carla_sim/model.py` rebuilt from the `world_v3` training code has 1,551,827 + 27,456 trainable parameters; the extra 27 + 6 values in the ONNX files are shape constants. `tools/carla_onnx_to_keras.py` maps every ONNX weight into it and reproduces all four outputs on real frames to within 2.4e-6 (M). This is the exact released architecture. The original epoch-20 `.keras` file (S3 only) is not needed.
- **TinySkies checkpoints to deployed ONNX.** For each ONNX weight tensor, we found a tensor in the Keras checkpoint with identical values (checked up to transpose and gate order):
  - `vae.keras` matches 31/31 tensors of encoder plus decoder.
  - `latent_sim.keras` matches 18/18 tensors.
  - As a negative control, `latent_sim` epoch 30 matches 0/18.

  Output parity through `tinyskies.model.load_checkpoints` is within 6.6e-6 (M). Re-exporting `latent_sim.keras` with `tinyskies/train_latent_sim.py` reproduces the deployed ONNX exactly.
- **VAE epoch.** The files `epoch_014` and `epoch_015` hold identical weights. Run `yvxlfm52` completed epoch 14 at 2026-05-24 17:40:56 UTC, which is the save time of `epoch_014`. It crashed during epoch 15, and `epoch_015` was re-saved at 18:19 UTC without new updates. The released VAE is therefore **the state after 14 completed epochs** (M).
- **Dataset builder.** `tinyskies/build_dataset.py` rebuilds `session_0005` bit-identically to the original shard (M).

---

## 2. Training runs

### TinySkies (full metric histories exported to `results/<run_id>/`)

The deployed models come from resumed run chains. Each run continues from the checkpoint of the one before.

**VAE `tinyskies_rgb_vae_v1`**
- Spatial 6×6×64 latent, MAE reconstruction + KL at weight 1e-5, lr 1e-4, batch 256, 90/10 session split.

| Run | Epochs | Notes |
|---|---|---|
| `3ixb3s35` | 1–6 | `epoch_006.keras` saved 05-23 16:08 PDT, when this run ended |
| `y0lnd3md` | 7 | crashed |
| `yvxlfm52` | 7–15 | crashed after epoch 15, 10.1 h. **Deployed encoder and decoder = epoch 15.** |

**Latent simulator `tinyskies_latent_sim_v1`**
- 2-layer GRU (256 units), 4-step latent history, residual latent delta.
- Loss: latent MSE + 0.01·‖Δ‖².
- Scheduled sampling 0.1, feedback noise 0.005, lr 3e-4.
- Two-stage curriculum, recorded in the run configs:

| Stage | Runs | Epochs | VAE | Rollout horizon | Batch |
|---|---|---|---|---|---|
| 1 | `jl1spotp` → `0b8k4g16` → `u8qj3r6l` | 1–28 | epoch 6 | 13 | 250 |
| 2 | `pwekx8km` / `7mi5pz2h` → `dukugksc` | 26–42 | epoch 15 | 128 | 200 |

- Stage 2 resumes from the epoch-26 checkpoint. It swaps in the retrained VAE and grows the rollout horizon from 13 to 128.
- `epoch_040.keras` was saved 2026-05-25 04:25 PDT, during `dukugksc` (04:18 UTC start + 7.06 h). **Deployed sim = epoch 40.**

**Lineage, not deployed** (configs in `results/wandb_run_index.txt`):
- VAE variants with a dense 128-d latent (05-22) and a dense 1280-d latent (05-22/23).
- A transformer latent-sim attempt (`latent_sim_v1_epoch_001`, 05-23).
- Pixel-space `tinyskies_world_v2` (in drift-sim git 05-18).
- `tinyskies_world_v3`, a VAE + transformer open-loop 16-step model (in drift-sim git 05-24).
- Latent and decoder refiners (`uaht98i9`, `zu68powr`, `xsxkrhr9`, 05-25).

### CARLA `world_v3` (deployed)

- **The metric history is not recoverable.** The W&B project `drift-world-v3` exists (created 2026-05-14 00:18 UTC) but has no runs, which is consistent with deletion. S3 was not reachable during this audit.
- **The training code** (`drift/agents_models/world_v3/train.py`) was first committed 2026-05-25, after training. Its current checkpoint prefix `world_v3/02/` differs from the deployed files' prefix `world_v3/`, so the deployed epoch 20 comes from an earlier run of this code.
  - To do: rebuild the model from the committed code and compare its parameter count and layer structure with the ONNX graph (1,551,854 params).
- **The only surviving training evidence** is `results/carla_world_v3/wandb_train_panel_screenshot.png`, class **Q**. It shows the train panel of the drift-sim write-up: waypoints, telemetry, rgb, one_step, total and lidar losses over about 72k steps. It was taken 2026-05-18, after the epoch-20 download on 05-13, so it may show a run that continued past epoch 20.
- **Consequence for the paper:** CARLA quantitative results come from new offline evaluations of the released ONNX model (`eval/`), not from training logs.

### CARLA `world_v4` (not deployed; lineage and future work only)

- W&B `drift-world-v4`: 22 runs, 2026-05-25/26, reaching epochs 1–5.
- Design: latent transition on a frozen RGB autoencoder (`drift-carla-ae`, run `vcu66nkv`, 43 epochs), with a 128-step open-loop rollout loss.
- Local ONNX epochs 1–4 (≈4.0M params total) are not shipped.

---

## 3. Datasets (not redistributed; statistics in `results/dataset/`)

### CARLA (`mlp_processed_v1`)

- Source: window manifest snapshot `76442aaa…c406`, identical in `drift-world/.asset-cache` (05-13 23:54 PDT) and `drift-sim/.asset-cache`.
- It contains only recordings dated 2026-05-09/10. It predates the 2026-05-17 recordings (the random and recovery tags) that the later `world_v4` manifest adds.
- `world_v3` trained on the full manifest at `manifests/mlp/v1/` (`DATASET_SAMPLE_FRACTION = 1.0`) before 05-14. That this is exactly this snapshot is an inference, class **I**.
- It is the same data as the companion imitation-learning report: 236,882 windows from 448 recordings.

| | Value | Class |
|---|---|---|
| Recordings | 448 (Town01 tags; see companion report for Town02) | M |
| Frames | 238,674 at 20 Hz = **3.32 h** | M |
| Five-frame windows | 236,882 | M |
| Direction mix | see `carla_summary.json` (≈89.5% straight) | M |
| \|steer\| > 0.1 | 9.6% of targets | M |

### TinySkies (`tinyskies_session_rollout_v3`)

| | Value | Class |
|---|---|---|
| Sessions | 75 × 60 s | M |
| Frames | 33,667 = **1.25 h** | M |
| Recording source | 62 from the unmodified live game (tinyskies.vercel.app); 13 from a local build with a state-export hook | M |
| Control policy | 67 random-WASD bot, 8 manual | M |
| Effective frame rate | 7.5 fps overall; per-session median 7.2, range 3.5–14.8. Nominal 15/20 fps; screenshot capture is the bottleneck, so frame spacing is irregular. | M |
| Controls | `[forward (W), brake (S), turn (D − A)]` | M |
| Split | 90/10 by session | M |

**Control semantics.** The recorder stores `[throttle, brake, steer]`, naming carried over from CARLA. The game code (`client/src/game/FlightControls.ts`, `Plane.ts`) confirms:
- W = `forward`, which accelerates to max speed;
- S = `brake`, which decelerates;
- A/D = `turnRate`.

The game README's "W/S pitch" row is stale. So the labels are semantically forward/brake/turn, and the paper uses those names.

---

## 4. Samples (redistributed)

| Path | Content | Source |
|---|---|---|
| `samples/carla/drive_20260509_110744_f240-359.processed.tfrecord` | 120 consecutive frames (6 s), straight into a left turn, all modalities | Byte-exact slice of `drive_20260509_110744.processed.tfrecord` (md5 `81a8ecef…`), a training recording |
| `samples/carla/browser_seeds.json` | The 6 seed states used by the browser | `drift-sim/public/samples/world_samples.json` |
| `samples/tinyskies/session_0005/` | 150 consecutive frames (10.3 s) at 320×180 plus the key log and controls | `drift-world/recordings/session_0005`, frames 100–249 (downscaled) |
| `samples/tinyskies/browser_seeds/session_0005/` | The 9 seed frames used by the browser | `drift-sim/public/tinysky_samples/session_0005` |

---

## 5. Evaluation splits

**CARLA.** The world_v3 split is reconstructed exactly from the training code: windows are sorted by recording path, and the first 80% are train. That gives 328 train recordings, 119 validation recordings, and 1 recording split across the boundary. Four recordings exist locally:

| Recording | Role | Frames |
|---|---|---|
| `drive_20260509_110744` | train | 500 |
| `drive_20260509_231436` | world_v3 validation (held out) | 400 |
| `drive_20260517_052600` | recorded after training (unseen) | 400 |
| `drive_20260517_063215` | recorded after training (unseen) | 400 |

**TinySkies.** The trainer assigns sessions to validation by `record_index % 10 == 9`. The record order comes from a parallel read of 15 shards with an autotuned cycle length, which depends on the training machine. So the exact held-out sessions cannot be reconstructed.

The evaluation therefore uses a fixed set: every 10th session in manifest order (sorted index `% 10 == 9`, 7 sessions), plus all 8 human-piloted sessions (0070–0077) as a separate control-policy condition. Results on these sessions are reported as **possibly seen during training**.
