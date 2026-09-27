# Claim Audit

Every substantive claim in `paper.md` is listed here with its evidence class. The protocol is the one used in the companion report:

- **M**: quantitatively measured, with a recoverable artifact (evaluation output, exported log, manifest, model file, code)
- **Q**: qualitatively observed (figure, capture, video), with no systematic measurement
- **I**: inferred interpretation of M/Q evidence
- **A**: confirmed by the author from first-hand use of the system; not independently recoverable from artifacts
- **U**: unsupported or unverified. It needs more evidence before it can be stated as fact.

**Sources.**
- **Eval** = `results/eval/` (produced by `eval/*.py` on the released ONNX models), summarised in `results/rollout_eval.ipynb` and `results/figures/*.csv`.
- **Lat** = `results/eval/latency_{python,browser}.json`, summarised in `results/latency.ipynb`.
- **Logs** = exported TinySkies training histories in `results/<run_id>/`.
- **Data** = `results/dataset/*` (from `tools/dataset_stats.py`).
- **Prov** = `PROVENANCE.md`.
- **Lit** = `paper/LITERATURE.md`.

## Models and data

| # | Claim | Class | Evidence |
|---|---|---|---|
| 1 | The released CARLA simulator is `world_v3` epoch 20; the browser serves byte-identical files | M | Prov §1: md5 of `artifacts/carla/*.onnx` = drift-sim deployed files = S3-named `world_v3_epoch_020_*` cache |
| 2 | CARLA model: 1,551,827 + 27,456 = 1,579,283 trainable parameters (the ONNX files add 33 shape constants), 6.4 MB fp32 | M | `artifacts/MANIFEST.json`; `carla_sim/model.py` count_params |
| 3 | `carla_sim/model.py` is the exact released architecture | M | `tools/carla_onnx_to_keras.py`: all outputs match ONNX within 2.4e-6 |
| 4 | The CARLA state head never sees the image; action reaches the image only via a 6×6×48 conditioning map | M | Architecture (claim 3); shuffle-RGB leaves structured outputs unchanged (`sensitivity.json`) |
| 5 | CARLA training: 8-step open-loop rollout loss; loss weights and optimiser as stated; scheduled sampling / noise declared but unused | M + I | Original training code (read for this report; reproduced in `carla_sim/train.py`). That epoch 20 was trained with this exact code version is **I**: the code was committed after training |
| 6 | The CARLA training history is not recoverable | M | Prov §2: W&B project exists with zero runs; S3 was not reachable |
| 7 | The surviving CARLA loss screenshot may postdate epoch 20 | M | Prov §2: screenshot 2026-05-18 vs ONNX download 2026-05-13 |
| 8 | CARLA data: 448 recordings, 238,674 frames, 3.32 h, 236,882 windows; 89.5% straight; 9.6% of targets have \|steer\|>0.1; max speed 10.0 m/s | M | Data (`carla_summary.json`); manifest snapshot md5 in Prov §3 |
| 9 | world_v3 trained on this manifest snapshot | I | Snapshot predates training and the later 05-17 recordings; `DATASET_SAMPLE_FRACTION=1.0` |
| 10 | Split: 328 train / 119 val recordings / 1 straddling | M | Reconstructed from the training code's split rule on the manifest (Prov §5) |
| 11 | Recorded camera frames contain green route markers drawn during collection | M + Q | Companion report; visible in `samples/carla` and `results/figures/carla_long_snapshots.png` |
| 12 | TinySkies released models: VAE after 14 completed epochs (file `epoch_015`), latent sim epoch 40 | M | Prov §1: tensor-by-tensor match 31/31 and 18/18, negative control 0/18; epoch_014 = epoch_015 weights; Logs timestamps |
| 13 | TinySkies trainable parameters 660,480 + 1,282,915 + 1,842,048 = 3,785,443 (the ONNX files add 41 constants); 15.2 MB | M | `artifacts/MANIFEST.json` |
| 14 | Two-stage curriculum (VAE e6 / horizon 13 → VAE e15 / horizon 128 from epoch 26) | M | Logs `config.json` of runs `jl1spotp`→`u8qj3r6l`, `7mi5pz2h`→`dukugksc` |
| 15 | At release, val last-step latent MSE is 10.6× one-step (0.0625 vs 0.0059); stage 2 did not reduce either | M | Logs; `results/figures/tinyskies_training_table.csv` |
| 16 | TinySkies data: 75 sessions, 33,667 frames, 1.25 h; 67 bot / 8 human; 62 public / 13 local build; effective 7.5 fps (session medians 3.5–14.8) | M | Data (`tinyskies_summary.json`, `tinyskies_sessions.csv`) |
| 17 | TinySkies controls are forward (W), brake (S), turn (A/D); the plane never stops | M | Game source `client/src/game/FlightControls.ts`, `Plane.ts` (`MIN_SPEED`); the game README's "pitch" row is stale |
| 18 | The exact TinySkies validation sessions cannot be reconstructed | M | Split depends on parallel-read order (Prov §5) |

## Size and speed

| # | Claim | Class | Evidence |
|---|---|---|---|
| 19 | Browser (onnxruntime-web 1.22 WASM, 4 threads, M1): CARLA 5.0 ms/step, TinySkies 18.3 ms/step; 1 thread 11.7 / 47.5 ms | M | Lat (browser), headless Chromium, closed loop, batch 1 |
| 20 | Python onnxruntime CPU: CARLA 6.1 / 2.6 ms; TinySkies 23.7 / 15.6 ms; TinySkies latent dynamics 0.50 / 0.35 ms (1 / 4 threads) | M | Lat (python) |
| 21 | Decoding takes ≈98% of TinySkies step time | M | Lat: decoder 23.5 ms of 23.7 ms (1 thread) |
| 22 | WebGPU fails on the GRU's `Split` kernel in onnxruntime-web 1.22 | M | Lat (browser) error string |
| 23 | Live demo readout (during capture): median 29 fps / 8.9 ms (driving), 24 fps / 37 ms (flying) | Q | HUD text logged during captures (`results/eval/captures/*/controls.jsonl`) |
| 24 | Simulated time in the demo runs ≈1.5× (CARLA) / ≈4× (TinySkies) faster than real time | I | 30 fps demo loop vs 20 Hz / 7.5 fps data |

## Offline evaluation (§6)

| # | Claim | Class | Evidence |
|---|---|---|---|
| 25 | CARLA predicted frames keep 10–33% of real high-frequency detail from step 1, flat over 5 s | M | Eval `replay.csv.gz`, Table 5 |
| 26 | CARLA rollout SSIM beats copy on the training recording after 0.3 s, is ≈copy on the held-out recording, and is below copy on post-training recordings | M | Eval, Fig. 4 |
| 27 | Held-out recording: predicted speed within 1.3 m/s of truth after 5 s (true 5.6–7.2 m/s) | M | Eval Table 5 |
| 28 | Post-training recordings are out of distribution (slow, off-route) and the model fails behaviourally there (≈8 m/s speed error) | M + Q | Eval; frames in `results/figures/carla_long_snapshots.png` and replay renders |
| 29 | Turning raises speed error 1.3–1.6× (0.5–2 s) and 5× (5 s), and waypoint error 1.8–1.9×; RGB almost unaffected | M | Eval `carla_turning_table.csv` |
| 30 | Controls move telemetry/waypoints 6–8× more than the image | M | `sensitivity.json` (shuffle controls: 0.057 / 0.045 vs 0.0074) |
| 31 | The CARLA state module is ≈0.1 M parameters | M | Layer arithmetic of the state MLPs and head in `carla_sim/model.py` (≈118 k) |
| 32 | Momentum survives while pixels blur *because* the dynamics live in a pathway pixels cannot corrupt | I | Claims 4, 25, 27 |
| 33 | TinySkies latent error below the copy baseline at every step to 64 | M | Eval `replay.csv.gz`, Table 7 |
| 34 | TinySkies VAE ceiling SSIM 0.79, 35% of fine detail; dynamics-only SSIM 0.95 → 0.71 | M | Eval `vae_ceiling.csv.gz`, Table 7 |
| 35 | TinySkies turning is not harder; human sessions modestly harder (0.070 vs 0.063 at step 64) | M | Eval |
| 36 | LiDAR embedding encodes scan information (ridge R² 0.80 / 0.77 vs raw-scan 0.89 / 0.96) | M | `lidar_probe.json` |
| 37 | LiDAR embeddings vary 0.17% across scans (inputs vary 129%); a constant predictor scores MSE 8e-6 | M | `lidar_probe.json` |
| 38 | Predicted LiDAR embedding does not track the true one (≈2,300 SDs off; correlation 0.03 → ≈0) | M | `lidar_probe.json` |
| 39 | The near-zero training LiDAR loss did not indicate a learned modality | I | Claims 37–38 plus the screenshot (Q) showing a near-zero lidar loss |
| 40 | "The model predicts LiDAR" | **Not claimed** | Only an embedding is carried; there is no decoder |

## Interactive behaviour (§7)

| # | Claim | Class | Evidence |
|---|---|---|---|
| 41 | Throttle raises and brake lowers predicted speed; steering moves route and lateral velocity in the commanded direction | M | Eval `action_response.csv.gz`, Fig. 6 |
| 42 | Low constant throttle settles (≈1 m/s idle, ≈3.1 m/s at 0.25); after full throttle, coasting keeps speed while braking lowers it | M | Eval |
| 43 | Acceleration changes how fast the world streams by | A | Author's driving sessions |
| 44 | …and the pixel metric supports this only weakly (0.0017 → 0.0019 → 0.0034 frame change, idle → 0.5 → full) | M | Eval (steps 10–60) |
| 45 | Throttle ≥0.75 drives speed beyond the 10 m/s data maximum to telemetry saturation in 6/6 seeds (2/6 at 0.5; median 2.75 s at full throttle; 2.55 s under left steer, not under right) | M | Eval `carla_long_table.csv`, `action_response.csv.gz` |
| 46 | Braking from saturation only reaches ≈22 m/s | M | Eval |
| 47 | The model lacks drag / a speed limit because the autopilot never demonstrated high speed | I | Claims 8, 45 |
| 48 | Replayed autopilot controls keep 6/6 seeds bounded and moving for 100 s (slow-first stream); the fast-first stream saturates 4/6 | M | Eval `carla_long_table.csv` |
| 49 | Constant inputs converge to fixed points (frozen frames) even when predicted speed is non-zero | M + Q | Eval; Fig. 7 |
| 50 | Out-of-distribution inputs lead to a shared noise-textured attractor, also reached in the live demo when throttle is held | M + Q | Eval snapshots; `assets/driving_runaway.gif` |
| 51 | A controller reading only predicted speed and waypoints keeps speed bounded for 100 s in all seeds but settles to near-constant actions and frozen frames | M | Eval `closed_loop.csv.gz` |
| 52 | The predicted structured state is usable for control | M (minimal) | Claim 51. **Not claimed:** that a learned policy trained inside the simulator would transfer |
| 53 | TinySkies terrain evolves coherently for ≈100–200 steps, with biome shifts, ocean regions and the plane sprite visible | Q + A | Fig. 8, Fig. 1, `assets/flying.gif`; author's play sessions |
| 54 | TinySkies UI elements, balloons and tricks are not reproduced | A + Q | Author; captures |
| 55 | TinySkies constant forward / brake / none: 9/9 seeds static by 2,000 steps; constant turning: 0–1/9 | M | Eval `tinyskies_long_table.csv` |
| 56 | TinySkies structure dissolves into colour fields after ≈500 steps | Q | Fig. 8 |
| 57 | Forward driving is the CARLA model's strongest regime; turning and large steering expose weak geometry | A + M | Author; claims 29, 45 |
| 58 | Vegetation and fine textures are imagined rather than reconstructed | Q + M | Claim 25; frames |
| 59 | Different seeds give different driving contexts | Q | Browser seeds (`samples/carla/browser_seeds.json`), Fig. 7 |

## Positioning and novelty

| # | Claim | Class | Evidence |
|---|---|---|---|
| 60 | Compact browser-run world models are not new (Ha & Schmidhuber 2018; 2025 demos) | M (lit.) | Lit §2.7 |
| 61 | To our knowledge, no system combines all five properties: sub-5 M transition model, continuous autoregressive rollout, real-time human control, predicted non-visual state, browser execution | I (scoped) | Lit §3 and §4 search log (web and arXiv through 2026-09-27); no Scholar/DBLP sweep |
| 62 | To our knowledge, this is the first browser-run learned CARLA transition model under continuous keyboard control | I (scoped) | Lit §3(a); DriveGAN is the closest (GPU) |
| 63 | To our knowledge, this is the first learned simulator of TinySkies | I (scoped) | Lit §3(c); weak as a contribution |
| 64 | Placement in Li's functional taxonomy: TinySkies = renderer; CARLA = renderer exposing simulator-style state | I | Lit §1 (essay verified 2026-09-27) |
| 65 | "First ever" / photorealistic / replaces CARLA / multi-agent / policy trained in the simulator | **Not claimed** | |
