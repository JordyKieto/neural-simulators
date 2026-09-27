# Paper outline (draft for review)

**Title.** *Neural Simulator: Lightweight World Models for Continuous Interactive Environments*
**Alternative.** *World Models You Can Play: Lightweight Neural Simulators for Continuous Interactive Environments*
**Format.** Technical report, same structure and evidence grading as the companion imitation-learning report.

**Spine.**
- Next-state prediction is the training problem. Sustained neural simulation is the research problem.
- The browser is the instrument that makes the second one directly executable and falsifiable.

**Evidence tags used below.**

| Tag | Meaning |
|---|---|
| M | Measured, with a recoverable artifact |
| Q | Qualitative: author observation, video |
| A | Author-confirmed |
| I | Inferred from other evidence |

---

## Abstract (≈200 words)
- **Question.** How little learned machinery does it take before a model behaves like an environment?
- **Setup.**
  - Two compact models, 1.6M and 3.8M parameters.
  - Two domains: CARLA multimodal driving and TinySkies visual flight.
  - Continuous autoregressive rollout under live human control, in a browser, with the source simulator never executed.
- **What survives:**
  - forward motion and speed persistence;
  - control-conditioned progression;
  - structured state: telemetry, waypoints, LiDAR latent.
- **What fails first:**
  - steering geometry;
  - high-frequency visual detail;
  - long-horizon coherence.
- **Contribution.** An executable, inspectable learned environment plus a characterization of where it stops behaving like one.

## 1. Introduction
- World models increasingly promise learned environments. Most evidence is latent imagination, offline prediction, generated video, or very large systems.
- **The distinction.** A model can have low one-step error and still be useless as a simulator. Simulation means consuming one's own predictions under actions that arrive continuously.
- **Deliberately concrete question.** Can compact models cross from predictor to interactive simulator?
- **Contributions:**
  1. We investigate sustained action-conditioned simulation, not short-horizon prediction.
  2. We test it in two different observation regimes: multimodal driving (CARLA) and visually grounded flight (TinySkies).
  3. We deploy both as directly controllable browser simulations with no source environment at inference.
  4. *(Measured.)* We characterize one-step versus rollout error, action response, long-horizon degradation, and latency or size.
- Figure 1: the human-in-the-loop diagram, with browser screenshots of both environments.

## 2. Related work and taxonomy
Draws on `paper/LITERATURE.md` and its verified taxonomy.
- **Where Neural Simulator sits in the taxonomy:**
  - action-conditioned;
  - learned transition;
  - autoregressive;
  - pixel or latent state, plus structured state;
  - interactive;
  - agent-compatible in principle.
- **Clusters:**
  - latent world models and model-based RL (World Models, Dreamer, IRIS, DIAMOND);
  - playable neural games (GameGAN, GameNGen, Oasis, Genie, Muse);
  - driving world models (GAIA-1/2, MILE, Think2Drive, and others);
  - evaluation beyond pixels;
  - browser and on-device demos.
- **Positioning.** The novelty is a combination: compact dynamics, continuous rollout, direct human control, multimodal predicted state and commodity execution. Not "a world model in JavaScript". Every "to our knowledge" claim is scoped by the documented search.

## 3. Problem formulation
- Environment $\mathcal{E}$, observations and state $s_t$, actions $a_t$. The learned transition is $\hat s_{t+1}=f_\theta(\hat s_t,a_t)$.
- **Training versus deployment:**
  - teacher-forced or short-horizon open-loop training;
  - unbounded closed-loop deployment on $\hat s$ with human $a_t$.
- **Two notions of quality:**
  - *perceptual fidelity*;
  - *behavioral or dynamical fidelity*.
- The operational definition of "behaves like an environment" is **a checklist tested in §7**:
  - responds to actions;
  - persists state;
  - progresses through the world;
  - stays coherent under its own distribution.

## 4. Environments and data
- **4.1 CARLA.**
  - Sensors, telemetry and waypoint definitions.
  - Data: 448 autopilot recordings, 3.32 h, Town01/02, 89% straight. It is the same data as the companion report.
  - Explicit about privileged inputs: HD-map waypoints and simulator telemetry.
- **4.2 TinySkies.**
  - The game (credited to its author).
  - Recorder: Playwright screenshots plus key log.
  - Data: 75 sessions, 1.25 h, 67 random-bot and 8 human.
  - Irregular ≈7.5 fps capture.
  - Controls are forward, brake and turn.
- Table 1: dataset sizes (M).

## 5. Models
- **5.1 CARLA transition model.**
  - Markov, one-step, deterministic. Image path with a broadcast state-action conditioning map, plus a structured head.
  - The LiDAR is encoded once and evolves as a latent.
  - 8-step open-loop rollout loss (RGB L1/MSE, SSIM, edge; latent, telemetry and waypoint MSE).
  - No scheduled sampling or noise was active.
- **5.2 TinySkies latent simulator.**
  - Frozen spatial VAE (6×6×64); 2×GRU-256 residual latent dynamics over 4-step latent and action history.
  - 128-step latent rollout loss with scheduled sampling and feedback noise.
  - Two-stage curriculum: VAE epoch 6 with a 13-step horizon, then the final VAE with a 128-step horizon.
- Table 2: parameters, sizes, inputs and outputs (M). Architecture figures for both.
- **5.3 Browser runtime.**
  - onnxruntime-web WASM; the closed loop; 30 fps pacing.
  - Simulated time runs faster than real time (≈1.5× and ≈4×).
  - Seeds.

## 6. Offline evaluation: one-step versus sustained rollout (M)
- **Replay protocol.**
  - Real seed state, recorded actions, H steps of free rollout.
  - Error at every horizon step against the real recording.
  - References: one-step (teacher-forced) error and copy-seed error.
  - TinySkies additionally separates dynamics error from the VAE reconstruction ceiling.
- **Figures:**
  - error against horizon;
  - one-step error against rollout error, which is the "low one-step loss ≠ simulator" point made quantitative;
  - sharpness against horizon, showing blur.
- **Structured state (CARLA):**
  - speed error over the horizon;
  - waypoint error;
  - cosine agreement between the predicted LiDAR latent and the encoding of the true LiDAR.
- **Splits.**
  - CARLA: train, validation (held out) and post-training recordings.
  - TinySkies: sessions possibly seen in training, stated explicitly.

## 7. Interactive behavior: what the learned environments capture (M + Q + A)
- **7.1 Action response.**
  - Throttle levels against predicted speed: speed rises and persists, which is momentum.
  - Throttle then coast versus throttle then brake.
  - Image motion rate against predicted speed: "the world moves faster".
  - Steering levels against lateral velocity, waypoint curvature and image rotation direction.
  - TinySkies: turn left or right against the direction of horizontal image motion; forward against brake.
- **7.2 Long-horizon rollouts.**
  - 2,000 steps (100 s of simulated CARLA time).
  - Frame change (does the world keep moving?), sharpness, distance from seed, collapse detection.
  - Snapshots every 5 s.
- **7.3 Human interaction** (Q/A: author sessions, videos).
  - Forward driving and flight feel continuous.
  - Environmental structure passes by the viewer.
  - Different seeds give different contexts.

## 8. Failure analysis (a first-class section)
**CARLA:**
- turning and strong steering push rollouts out of distribution (the data is ≈90% straight);
- visual degradation and hallucinated vegetation and texture;
- geometry inconsistency over long rollouts;
- no traffic or multi-agent behaviour;
- bounded by the data distribution.

**TinySkies:**
- low fidelity;
- UI elements and game features are missing (balloons, tricks, HUD);
- terrain evolves plausibly but is not a faithful globe;
- drift, collapse and action responsiveness as measured.

**Both:**
- what compact models learn first (coarse motion, momentum) against last (geometry, detail).

The two experiments are not forced into symmetry.

## 9. Perceptual versus behavioral fidelity (discussion)
- Evidence that dynamics survive while pixels degrade.
- Not claimed: sufficiency for policy learning.
- The browser as a falsification instrument: interactivity, inspectability, portability.
- Agent use is possible in principle through the structured CARLA state. No policy was trained inside the simulator; RGB fidelity and rollout error are why this iteration did not attempt it.

## 10. Limitations and threats to validity
- Small evaluation sets: 4 local CARLA recordings and possibly-seen TinySkies sessions.
- The CARLA training history was not recoverable. Evaluation of the released model stands in.
- Privileged CARLA inputs.
- The TinySkies frame rate.
- Human-interaction evidence is qualitative.
- Simulated time differs from real time.

## 11. Future work
- A multimodal training emphasis, with RGB treated as the first viability test.
- A LiDAR decoder.
- History and recurrence for CARLA.
- Turn-balanced data.
- Policy learning inside the simulator with transfer back to CARLA.
- Behavioral metrics.
- WebGPU.

## 12. Conclusion
Predicting worlds versus executing them. These environments are compact approximations of slices of dynamics that can be entered, controlled, observed, broken and studied.

## Appendices
- A. Artifact provenance (from `PROVENANCE.md`).
- B. Model lineage: CARLA v1→v4; TinySkies pixel v1/v2, transformer v3, dense VAEs, then the latent GRU.
- C. Training curves (TinySkies, exported), plus the surviving CARLA screenshot (Q).
- D. Reproducibility commands.
