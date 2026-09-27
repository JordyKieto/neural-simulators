# Literature Review: Neural Simulator

*Prepared 2026-09-27 for "Neural Simulator: Lightweight World Models for Continuous Interactive Environments" (J. Kieto, 2026, technical report).*

**How things were checked.** For every arXiv citation, the title, full author list, first-submission date and arXiv comment (which usually names the venue) were pulled from the arXiv export API (`export.arxiv.org/api/query?id_list=...`) on 2026-09-27. The BibTeX in §5 is generated directly from those API records. Nature papers were checked through Crossref (DOI metadata). Blogs, product pages and repositories were checked by fetching the official page or the GitHub/Hugging Face API. Anything that could not be confirmed from a primary source is marked **UNVERIFIED**. Where a venue is not stated in the arXiv metadata, the entry is cited as an arXiv preprint and no venue is claimed.

**Two corrections to the brief:**
- *"Learning to Drive from a World Model"* (arXiv:2504.19077) is by **comma.ai** (Goff, Hogan, Hotz, et al.), not Wayve. Wayve's closest works are GAIA-1/2, MILE and Ghost Gym.
- The TinySkies repository (`github.com/dannylimanseta/tinyskies`) is public, but the GitHub API reports **no license** (checked 2026-09-27). Say "publicly available" or "source-available" rather than "open-source" unless the author confirms a license.

---

## 1. Taxonomy

### 1.1 The Fei-Fei Li taxonomy (verified)

The work the author remembers is:

> **Fei-Fei Li. "A Functional Taxonomy of World Models." *Dr. Fei-Fei Li* (Substack), 3 June 2026.** https://drfeifei.substack.com/p/a-functional-taxonomy-of-world-models (also posted as an X article: https://x.com/drfeifei/article/2062247238143996275).

It is an essay, not a peer-reviewed paper. It sorts world models by **what they output**, into three functional categories. The definitions below are quoted from the essay:

| Category | Definition (verbatim) | Examples Li gives |
|---|---|---|
| **Renderer** | "A renderer outputs observations in the form of pixels meant for human eyes, and the quality that matters most is visual fidelity." | Genie 3; World Labs' RTFM ("generates frames in real time conditioned on user input"); text-to-video models |
| **Simulator** | "A simulator outputs state: a geometrically, physically or dynamically faithful representation of the world that humans and computer programs can both compute on and interact with." It demands "geometry that holds up under inspection, physics that respects Newton's laws", and serves as a training ground where agents "can interact with the world at scale." | Physics engines, RL training environments, robot controllers |
| **Planner** | "A planner outputs actions. Given an observation and a goal, a planner answers the question of what the agent should do next." | VLAs, "World Action Models" |

The essay's main thesis is that these categories are blending. It calls them "three projections of a single underlying understanding". It notes that renderers are becoming "action-conditioned", and it gives World Labs' Marble (which outputs renderings plus collision meshes) as an example of the renderer/simulator boundary dissolving.

**An earlier Li essay (also verified) that is often confused with this one:**

> Fei-Fei Li. "From Words to Worlds: Spatial Intelligence is AI's Next Frontier." 10 Nov 2025. Published on her Substack and syndicated on a16z.news (https://www.a16z.news/p/from-words-to-worlds-spatial-intelligence).

That essay does not give a taxonomy. It lists **three essential capabilities** a world model must have:
- **Generative**: "World models can generate worlds with perceptual, geometrical, and physical consistency".
- **Multimodal**: "World models are multimodal by design", with inputs including images, video, depth, text, gestures and actions.
- **Interactive**: "World models can output the next states based on input actions".

**Also related:** Li is a co-author of **WorldScore** (Duan, Yu, Chen, Fei-Fei, Wu; ICCV 2025; arXiv:2504.00983). It is a benchmark rather than a taxonomy, and it scores controllability and dynamics as well as image quality. See §2.6.

### 1.2 Other recent surveys and taxonomies (verified)

| Work | Taxonomy axes / categories |
|---|---|
| **Ding et al., "Understanding World or Predicting Future? A Comprehensive Survey of World Models"**, arXiv:2411.14499 (extended version of an ACM CSUR paper) | Two functions: (i) *Implicit representation of the external world* (world models in model-based RL / decision-making; world knowledge), and (ii) *Future prediction of the physical world*, split into *world model as video generation* and *world model as embodied environment* (indoor / outdoor / **dynamic** environments). Applications: game, embodied, urban (incl. autonomous driving) and societal intelligence. |
| **Li, He, Zhang, Wu, Li, Liu, "A Comprehensive Survey on World Models for Embodied AI"**, arXiv:2510.16732 | Three axes: *Functionality* (decision-coupled vs. general-purpose); *Temporal modeling* (sequential simulation and inference vs. global difference prediction); *Spatial representation* (global latent vector, token feature sequence, **spatial latent grid**, decomposed rendering representation). |
| **Yu et al., "A Survey of Interactive Generative Video"**, arXiv:2504.21853 | An ideal IGV system has five modules: **Generation, Control, Memory, Dynamics, Intelligence**. A companion position paper argues that IGV is the next-generation game engine (Yu et al., arXiv:2503.17359). |
| **Wang et al., "A Mechanistic View on Video Generation as World Models: State and Dynamics"**, arXiv:2601.17067 | Two pillars: *State construction* (implicit context management vs. explicit latent compression) and *Dynamics modeling*. Argues for moving evaluation from visual fidelity to functional benchmarks. |
| Driving-specific: Guan et al. arXiv:2403.02622; Feng, Wang, Yang arXiv:2501.11260; Zeng & Dong arXiv:2603.09086 ("Latent World Models for Automated Driving: A Unified Taxonomy...") | Useful when positioning the CARLA experiment. |
| Also relevant: Zhu et al. "Is Sora a World Simulator?" arXiv:2405.03520; Kong et al. "3D and 4D World Modeling: A Survey" arXiv:2509.07996; Yue et al. "Simulating the Visual World with AI: A Roadmap" arXiv:2511.08585; Chen et al. "A Definition and Roadmap for World Models" arXiv:2607.06401; Liu et al. "Towards Interactive Video World Modeling" arXiv:2606.01164 | Broader context. |

### 1.3 Where Neural Simulator fits

The relevant properties are:
- action-conditioned;
- learned transition dynamics;
- autoregressive closed-loop rollout;
- state in pixels (CARLA) or latents (TinySkies, 6x6x64);
- structured multimodal state in CARLA (RGB, 64-d LiDAR latent, 5-d telemetry, 25 waypoints);
- real-time human control;
- possible use as an agent environment;
- learned simulation, as opposed to passive video generation.

How it classifies under each scheme:

- **Li (2026), functional taxonomy.**
  - The TinySkies model is a small **action-conditioned renderer**. Its output is pixels for a human player, which is the same category Li gives Genie 3 and RTFM.
  - The CARLA model is a **renderer-simulator hybrid**. It also outputs *state* (telemetry, waypoints, a LiDAR embedding) that a program could compute on, which is the defining output of a "simulator". However, Li's simulator category also requires the state to be "geometrically, physically or dynamically faithful". Neural Simulator's state is learned and approximate, with no physical guarantees.
  - Recommended wording: *"a renderer that exposes simulator-style state"*, or *"a learned simulator in interface, not in guarantees"*. It is not a planner, because it outputs no actions.
  - This matches Li's convergence thesis: Neural Simulator is a small instance of the renderer/simulator boundary blurring.
- **Li (2025), three capabilities.** Neural Simulator is squarely **interactive** ("output the next states based on input actions"). The CARLA model is **multimodal** on the output side. It is only weakly **generative**: it cannot synthesize new worlds from prompts, and it rolls out from a seed frame.
- **Ding et al.** It falls under *future prediction → world model as embodied environment (dynamic, outdoor for CARLA)*, not implicit representation. The application domains are game intelligence and autonomous driving.
- **Li et al. (embodied survey).**
  - Functionality: *general-purpose*, because it is not trained jointly with a policy.
  - Temporal: *sequential simulation*.
  - Spatial representation: *spatial latent grid* for TinySkies; for CARLA, a pixel-plus-vector state that no single axis value fits cleanly. The closest is *global latent vector* plus decoded pixels.
- **Yu et al. (IGV).** It addresses **Generation** (low fidelity) and **Control** (real-time keyboard). **Dynamics** are learned, and the reported failure modes (turning, drift) are dynamics errors. **Memory** is minimal (Markov for CARLA, a 4-step history for TinySkies). There is no **Intelligence** module.

**Suggested one-line positioning:** *"Neural Simulator is an action-conditioned, autoregressive learned simulator, in Li's (2026) terms a renderer that additionally exposes simulator-style non-visual state, built at a compact scale and run client-side."*

---

## 2. Related work by cluster

### 2.1 Foundations: learned world models and model-based RL

- Ha & Schmidhuber, **World Models** (arXiv:1803.10122; NeurIPS 2018 version "Recurrent World Models Facilitate Policy Evolution", arXiv:1809.01999).
  - Architecture: VAE (V) + MDN-RNN (M) + linear controller (C). The agent is trained inside its own "dream".
  - The interactive article (https://worldmodels.github.io) runs the models **in the browser** using deeplearn.js and p5.js. In the CarRacing and VizDoom "dream" demos the user can override the agent with the arrow keys.
  - Parameter counts from the article: CarRacing V 4,348,547 + M 422,368 + C 867 (about **4.8M total**); DoomTakeCover V 4,446,915 + M 1,678,785. Frames are 64x64 with a 64-d latent.
- Hafner et al., **PlaNet** (arXiv:1811.04551); **Dreamer** (arXiv:1912.01603); **DreamerV2** (ICLR 2021, arXiv:2010.02193); **DreamerV3** (arXiv:2301.04104; *Nature* 640:647–653, 2025, doi:10.1038/s41586-025-08744-2); **Dreamer 4** (Hafner, Yan, Lillicrap, arXiv:2509.24527). Dreamer 4's Minecraft world model does "real-time interactive inference on a single GPU", and agents are trained purely in imagination.
- Schrittwieser et al., **MuZero** (arXiv:1911.08265; *Nature* 588:604–609, 2020). A value-equivalent latent model that is not meant to reconstruct observations.
- Micheli et al., **IRIS** (ICLR 2023, arXiv:2209.00588) and **Δ-IRIS** (ICML 2024, arXiv:2406.19320). Robine et al., **TWM** (ICLR 2023, arXiv:2303.07109). Zhang et al., **STORM** (arXiv:2310.09615). Alonso et al., **DIAMOND** (NeurIPS 2024, arXiv:2405.12399).

*Relation.* Neural Simulator inherits the V+M decomposition (TinySkies: VAE + GRU) and the idea of rollout in a learned model. It differs in its goal. These works optimise the world model as a *tool for policy learning*, and the model is judged by agent return. Neural Simulator treats the rollout itself, driven by a human in real time, as the product, and evaluates it as an environment. Ha & Schmidhuber is the closest ancestor both conceptually and in deployment (compact, browser-run, human-overridable), so the paper should acknowledge it explicitly.

### 2.2 Action-conditioned video prediction

- Oh et al. 2015, **Action-Conditional Video Prediction using Deep Networks in Atari Games** (NeurIPS 2015, arXiv:1507.08750).
- Chiappa et al. 2017, **Recurrent Environment Simulators** (ICLR 2017, arXiv:1704.02254). Studies long-horizon action-conditioned simulation in Atari, TORCS and 3D mazes, and the effect of prediction-dependent training on long rollouts.
- Finn, Goodfellow & Levine 2016 (NeurIPS 2016, arXiv:1605.07157).
- Babaeizadeh et al., **SV2P** (arXiv:1710.11252).
- Bamford & Lucas, **Neural Game Engine** (arXiv:2003.10520). Accurate pixel forward models for GVGAI games that generalise to other level sizes.

*Relation.* Neural Simulator's CARLA model is structurally an Oh/Chiappa-style deterministic, action-conditioned one-step predictor. Chiappa et al. already showed that training on the model's own predictions (multi-step objectives) is key for long rollouts, and Neural Simulator's 8-step open-loop loss and 128-step scheduled-sampling loss follow that line. What Neural Simulator adds is the joint prediction of non-visual state and deployment as a real-time, human-driven, in-browser environment.

### 2.3 Learned game engines and playable neural games

| Work | Domain | Scale / runtime | Notes |
|---|---|---|---|
| **GameGAN**, Kim et al., CVPR 2020 (arXiv:2005.12126) | Pac-Man, VizDoom | GPU | Memory module plus disentangled static/dynamic rendering. **GAN Theft Auto** (sentdex, 2021, github.com/sentdex/GANTheftAuto) is a GameGAN fork for GTA V highway driving that runs on a local CUDA GPU. |
| **DriveGAN**, Kim et al., CVPR 2021 (arXiv:2104.15060) | **CARLA**, Gibson, 160 h of real driving | GPU; the project page describes "an interactive user interface ... controls for the steering wheel and speed which can be controlled with the keyboard" | **The closest prior human-drivable learned CARLA simulator.** Pixels only. |
| **GameNGen**, Valevski et al., ICLR 2025 (arXiv:2408.14837) | DOOM | Diffusion; 20 fps on a TPU | "first game engine powered entirely by a neural model" (their claim) |
| **DIAMOND** (above) | Atari, CS:GO | ~10 fps on an RTX 3090 (secondary report); runs locally | |
| **Oasis**, Decart & Etched, 31 Oct 2024 (https://oasis-model.github.io; open 500M weights at github.com/etched-ai/open-oasis) | Minecraft | ViT autoencoder + DiT; 500M open weights | The web demo is **streamed from a server**, not run client-side. |
| **Lucid v1**, Lucid, 11 Nov 2024 (https://www.lucid.ai/notes/lucid-v1) | Minecraft | ~1B params, 15 tokens/frame, ~20–25 fps on an RTX 4090 | |
| **MarioVGG**, Virtuals Protocol 2024, "Video Game Generation: A Practical Study using Mario" (PDF at virtual-protocol.github.io/mario-videogamegen) | Super Mario Bros | Not real-time (~4 s for 6 frames on an RTX 4090, per secondary sources) | **No arXiv ID found**; cite as a technical report. |
| **PlayGen**, Yang et al. (arXiv:2412.00887) | Mario, DOOM | Real time on an RTX 2060, >1000 frames | |
| **Genie**, Bruce et al., ICML 2024 (arXiv:2402.15391); **Genie 2** (DeepMind blog, 4 Dec 2024); **Genie 3** (DeepMind blog, 5 Aug 2025, Parker-Holder & Fruchter) | 2D platformers → general 3D | Genie 3: 720p, 24 fps, minutes of consistency; tested with the SIMA agent | Genie 2 is "an autoregressive latent diffusion model"; Genie 3 "simulates the future based on the agent's actions". |
| **Muse / WHAM**, Kanervisto et al., *Nature* 638:656–663 (2025) | Bleeding Edge | 1.6B | **WHAMM** (MSR blog, 4 Apr 2025): Quake II at 10+ fps and 640x360, playable in Copilot Labs in the browser, **server-side**. |
| **GameGen-X** (arXiv:2411.00769); **GameFactory** (ICCV 2025, arXiv:2501.08325); **The Matrix** (arXiv:2412.03568); **Matrix-Game** (arXiv:2506.18701) and **Matrix-Game 2.0** (arXiv:2508.13009); **MineWorld** (arXiv:2504.08388); **Hunyuan-GameCraft** (arXiv:2506.17201) and **-2** (arXiv:2511.23429); **Yan** (arXiv:2508.08601) | Various | Large diffusion/AR models on GPU | |
| 2026: **DreamForge-World 0.1** (arXiv:2606.30292; 14–15 fps at 480p on one RTX 4090); **WanToFight** (arXiv:2607.12592); **ABot-World-0** (arXiv:2607.19191, single desktop GPU); **Overworld Waypoint-1 / 1.5** (company releases Jan / Apr 2026, local-first diffusion world model, over.world) | Various | Consumer GPU | |
| **Mirage / Mirage 2** (Dynamics Lab, Aug 2025) | UGC game worlds | Real time, hosted | **UNVERIFIED**: only press coverage was found; no primary technical report was located. |

*Relation.* Neural Simulator belongs to this "playable neural game" line but sits at the opposite end of the scale axis. These systems are 10^8–10^10 parameters, run on datacenter or high-end GPUs, and predict pixels only. Neural Simulator is 1.6–3.8M parameters, runs in a CPU/WASM browser runtime, and (for CARLA) predicts structured state. Its fidelity is far lower. The contribution is accessibility, state, and honest analysis, not visual quality.

### 2.4 Driving world models

- **Video / multiview generators.** GAIA-1 (Wayve, arXiv:2309.17080), GAIA-2 (arXiv:2503.20523), DriveDreamer (arXiv:2309.09777), Drive-WM (arXiv:2311.17918), GenAD (CVPR 2024, arXiv:2403.09630), Vista (NeurIPS 2024, arXiv:2405.17398), DrivingWorld (arXiv:2412.19505), ReSim (NeurIPS 2025, arXiv:2506.09981; mixes real data with *non-expert CARLA data* for controllability), DriveArena (arXiv:2408.00415; closed loop with a generative "World Dreamer"), NVIDIA OmniDreams (arXiv:2606.03159; real-time closed loop), ZYT-World (arXiv:2609.21712; real-time, one-step streaming).
- **CARLA latent world models for policy learning.** MILE (Hu et al., NeurIPS 2022, arXiv:2210.07729; decodes to BEV semantic segmentation and "can execute complex driving manoeuvres from plans entirely predicted in imagination"). Think2Drive (ECCV 2024, arXiv:2402.16720; a "neural simulator" of BEV latents for training an RL planner in CARLA v2). Raw2Drive (NeurIPS 2025, arXiv:2505.16394; raw multi-view images + IMU). CarDreamer (arXiv:2405.09111; DreamerV3-based platform in CARLA). Khanzada & Kwon (arXiv:2604.27994; DreamerV3-style agents across CARLA towns).
- **Multimodal / LiDAR driving world models.**
  - **MUVO** (Bogdoll et al., IV 2025, arXiv:2311.11762) is trained **entirely on CARLA**, is **conditioned on acceleration and steering**, builds on MILE, and **predicts RGB, LiDAR (range view) and 3D occupancy**. **It is the closest prior work to claim (b).**
  - BEVWorld (arXiv:2407.05679; image + LiDAR via BEV latents). Copilot4D (ICLR 2024, arXiv:2311.01017; LiDAR point-cloud world model). OccWorld (arXiv:2311.16038; 3D occupancy). ViDAR (arXiv:2312.17655). Khurana et al. (CVPR 2023, arXiv:2302.13130; 4D occupancy / point-cloud forecasting).
- **Training a driving policy in a learned world model.** comma.ai's "Learning to Drive from a World Model" (arXiv:2504.19077; CVPR 2025 workshop). Wayve's Ghost Gym (blog, 21 Dec 2023; a neural-rendering closed-loop simulator, not a learned transition model).

*Relation.* Among learned driving models, only MUVO and MILE share Neural Simulator's setting (CARLA, action-conditioned, multimodal or structured outputs), and MUVO goes further on LiDAR, since it decodes point clouds and Neural Simulator does not. Neural Simulator's CARLA model differs in four ways:
1. It jointly predicts *vehicle telemetry and route waypoints* alongside RGB and a LiDAR embedding.
2. It is Markov and deterministic, at about 1.6M parameters.
3. It is exposed as a real-time, human-drivable environment, not used offline or for policy learning.
4. It runs client-side in a browser.

Large driving world models (GAIA, Vista, OmniDreams) target photorealism and datacenter or GPU inference.

### 2.5 Training agents inside learned simulators ("dream" training)

World Models (Ha & Schmidhuber); SimPLe (Kaiser et al., arXiv:1903.00374, Atari); Dreamer v1–v4; IRIS / Δ-IRIS; DIAMOND; Genie 3 with SIMA; Think2Drive and Raw2Drive (CARLA); comma.ai's learned-world-model policy (arXiv:2504.19077); UniSim, *Learning Interactive Real-World Simulators* (Yang et al., arXiv:2310.06114).

*Relation.* Neural Simulator does **not** train an agent. Its claim should stay at *"exposes an interface (state vector, telemetry, waypoints) that could serve as an agent environment"*. This literature, especially SimPLe and DIAMOND, shows that the gap between model and real returns is the key test. Stating this as future work is safer than implying agent utility.

### 2.6 Evaluating world models beyond pixel fidelity; compounding error

- **Benchmarks.** WorldScore (Duan, Yu, Chen, **Fei-Fei**, Wu, ICCV 2025, arXiv:2504.00983; controllability, quality, dynamics). WorldModelBench (arXiv:2502.20694; instruction-following and physics adherence). VBench (arXiv:2311.17982). WorldSimBench (arXiv:2410.18072). ACT-Bench (arXiv:2412.05337; **action controllability of driving world models**). MIND (arXiv:2602.08025; memory consistency and action control). WBench (arXiv:2605.25874). WorldRoamBench (arXiv:2606.31672; long-horizon stability). SmallWorlds (arXiv:2511.23465; dynamics understanding and rollout degradation in isolated environments).
- **Physics critiques.** Kang et al., "How Far is Video Generation from World Model: A Physical Law Perspective" (ICML 2025, arXiv:2411.02385). Motamed et al., Physics-IQ (arXiv:2501.09038). OpenAI, "Video generation models as world simulators" (2024 technical report, https://openai.com/index/video-generation-models-as-world-simulators/; the page returned 403 to automated fetch, so the author list is **UNVERIFIED** here). Zhu et al., "Is Sora a World Simulator?" (arXiv:2405.03520).
- **Exposure bias and compounding error.** Bengio et al., Scheduled Sampling (arXiv:1506.03099). Ranzato et al. (arXiv:1511.06732). Talvitie, Self-Correcting Models (AAAI 2017, arXiv:1612.06018). Lambert et al., compounding prediction errors (arXiv:2203.09637). Diffusion Forcing (arXiv:2407.01392). Self Forcing (NeurIPS 2025, arXiv:2506.08009).

*Relation.* Neural Simulator's honest failure analysis (turning vs. forward motion, blur, drift) fits this move towards *action-following* and *long-horizon* metrics. Recommendation: report per-action (steer vs. throttle) rollout error against horizon, and telemetry/waypoint error against horizon, in the spirit of ACT-Bench and WorldScore's controllability axis, not only PSNR/SSIM.

### 2.7 On-device, browser and tiny interactive world models (key cluster for novelty)

| Work | Date | What it demonstrates | Size | Runtime | State |
|---|---|---|---|---|---|
| Ha & Schmidhuber interactive article, https://worldmodels.github.io | 2018 | Human can override the agent with the arrow keys inside the learned CarRacing and VizDoom "dreams" (MDN-RNN + VAE decoder) | ~4.8M (CarRacing), ~6.1M (Doom) | **Browser**, deeplearn.js | 64x64 pixels / 64-d latent |
| O. Boer Bohan, "World Emulation via Neural Network", https://madebyoll.in/posts/world_emulation_via_dnn/ | 25 Apr 2025 | Playable forest trail learned from ~15 min of phone video, 6-DoF stick control | ~5M params, ~1 GFLOP/frame, 192x256 | **Browser**, ONNX Runtime Web (WebGL) on an iPhone 13 Pro, 60 fps cap | Pixels; 32-frame memory |
| njkumar, Flappy Bird world model, https://www.njkumar.com/optimizing-flappy-bird-world-model-to-run-in-a-web-browser/ | 23 Jul 2025 | DIAMOND-style diffusion world model distilled from 381M to ~5M (denoiser) | ~5M + AE | **Browser**, ONNX Runtime WebGPU (WASM fallback); 30 fps on an M2 Pro, 12–15 fps on an iPhone 14 Pro | Pixels |
| codelion, neural-drive-model (SuperTuxKart), https://huggingface.co/codelion/neural-drive-model ; Space `codelion/neural-drive` | Jul 2026 (HF createdAt 2026-07-19) | Drivable kart game; ConvVAE + latent DiT, 7-frame context | 130.8M | **Browser**, ONNX/WebGPU fp16 | RGB only |
| physicalai-bmi "nano-world-model(-multi)", https://physicalai-bmi.org/research/world-model | Jul 2026 (HF createdAt 2026-07-13) | Drive a rover in six "worlds"; fully convolutional next-frame model | 48,963 params | On-device, in-browser (claimed) | Pixels. **Provenance UNVERIFIED**: the institute's identity could not be confirmed and there is no paper. |
| GaganGutta, neural-game-engine, https://github.com/GaganGutta/neural-game-engine | Aug 2026 | VizDoom via VQ-VAE + MaskGIT transformer | 2.0M shipped | **Desktop CPU**, 35 fps (not browser) | Pixels |
| AlmondGod, tinyworlds, https://github.com/AlmondGod/tinyworlds | May 2025 | Minimal Genie reimplementation (latent actions) | Small | Local | Tokens |
| Overworld Waypoint-1/1.5 | 2026 | Local-first real-time diffusion world model on consumer hardware | Large | Local GPU | Pixels |
| Oasis, WHAMM, Genie 3 (web app) | 2024–26 | Browser *access* | 10^8–10^10 | **Server-side** | Pixels |

*Relation.*
- **Compact, browser-run, human-controllable learned world models are not new.** Ha & Schmidhuber did this in 2018, and several 2025–2026 demos run at about 5M parameters or less in ONNX Runtime Web / WebGPU.
- **No browser-run learned CARLA simulator was found**, and **no browser-run model predicting non-visual state (telemetry, waypoints, LiDAR embedding)** was found either.
- Most of these demos are blog posts or repositories, not papers. Cite them as web resources and give access dates.

---

## 3. Novelty assessment

The search scope is given in §4 (web and arXiv, through 2026-09-27).

| Claim | Verdict | Closest prior work | Recommended safe wording |
|---|---|---|---|
| **a.** First browser-executable learned CARLA environment continuously drivable by a human | **PARTIALLY SUPPORTED.** No prior browser-run learned CARLA simulator was found, but the claim's novelty rests mostly on the specific pairing of "CARLA" and "browser". | DriveGAN (CVPR 2021): learned CARLA/real-driving simulator with a keyboard-driven UI, run on GPU. Ha & Schmidhuber 2018: browser-run learned CarRacing dream with keyboard override. neural-drive-model (2026): browser WebGPU drivable kart world model. | "To our knowledge (search scope in Appendix X), this is the first learned CARLA transition model released to run client-side in a web browser under continuous human keyboard control; related human-drivable learned driving simulators (DriveGAN) run on GPUs, and prior browser-run world models (Ha & Schmidhuber 2018; others) target other environments." |
| **b.** Learned CARLA simulator jointly predicting RGB + LiDAR + telemetry + waypoints | **PARTIALLY SUPPORTED.** MUVO already predicts RGB + LiDAR + occupancy, action-conditioned, in CARLA. MILE predicts BEV. Joint prediction of **telemetry + route waypoints** with RGB and a LiDAR embedding was not found. **Caveat:** Neural Simulator predicts a LiDAR *latent* with no decoder, so it must not be described as "predicting LiDAR". | MUVO (arXiv:2311.11762); MILE; BEVWorld; Think2Drive (BEV latent) | "a compact action-conditioned CARLA transition model whose state jointly includes RGB, a LiDAR-derived embedding, ego telemetry and route waypoints; unlike multimodal driving world models such as MUVO, which decode LiDAR and occupancy, we carry LiDAR only as a latent and instead expose vehicle-level state." |
| **c.** First learned neural simulator of TinySkies | **SUPPORTED, but weak as a contribution.** No prior learned model of TinySkies was found. The game is niche, so this is expected and has little significance on its own. | None found | "We also train, to our knowledge for the first time, a learned simulator of TinySkies, a small publicly available three.js flight game." Check the license before saying "open-source". |
| **d.** Compact (<5M params) world models sustaining real-time human-controlled autoregressive rollout in commodity browsers | **NOT SUPPORTED as novel.** | Ha & Schmidhuber 2018 (~4.8M, browser, keyboard override); Boer Bohan 2025 (~5M, ONNX Runtime Web, 60 fps on iPhone); njkumar 2025 (~5M denoiser, WebGPU, 30 fps); nano-world-model 2026 (49K, provenance unverified) | Present it as a *property*, not a first: "Like prior browser-run world models (Ha & Schmidhuber 2018; Boer Bohan 2025), both models run client-side in real time; ours do so at 1.6M / 3.8M parameters on the WASM (CPU) backend." |
| **e.** The combination: compact dynamics + continuous AR rollout + direct human control + multimodal predicted state + browser | **PARTIALLY SUPPORTED** (defensible if scoped). No single prior work found has all five. Ha 2018 lacks multimodal/structured state and CARLA. MUVO/MILE lack real-time human control and browser deployment. DriveGAN lacks non-visual state and browser deployment. | Ha & Schmidhuber 2018; MUVO; DriveGAN; Boer Bohan 2025 | "To our knowledge, no prior system combines all of: (i) a sub-5M-parameter learned transition model, (ii) continuous autoregressive rollout, (iii) direct real-time human control, (iv) predicted non-visual state, and (v) client-side browser execution. Individual components have clear precedents (Table X)." |

General advice:
- Avoid "first ever".
- Put a "Scope of search" footnote pointing to §4.
- Include a comparison table with rows for Ha 2018, DriveGAN, MUVO, GameNGen, DIAMOND, Oasis, Boer Bohan 2025, njkumar 2025 and Neural Simulator, and columns for params, action-conditioned, human real-time, non-visual state, runtime, and domain.

---

## 4. Search log

All searches were run on **2026-09-27** with WebSearch (US web index), plus WebFetch of primary pages and the arXiv export API for metadata. Queries:

1. `Fei-Fei Li world models taxonomy categories 2026`
2. `"From Words to Worlds" Fei-Fei Li spatial intelligence world model`
3. `Fei-Fei Li co-author survey world models arXiv 2025 embodied`
4. `drfeifei.substack.com "From Words to Worlds" generative multimodal interactive`
5. `MarioVGG video game generation Super Mario Bros arXiv Virtuals Protocol`
6. `"Video Game Generation: A Practical Study using Mario" arxiv`
7. `World and Human Action Models towards gameplay ideation Nature 2025 Kanervisto WHAM Muse Bleeding Edge`
8. `Oasis Decart Etched Minecraft real-time world model 2024 open source 500M`
9. `Mirage real-time generative game engine Dynamics Lab arXiv 2025`
10. `Lucid v1 Minecraft world model real-time consumer GPU Rami Seid`
11. `Genie 3 DeepMind blog "a new frontier for world models" 2025`
12. `world model runs in the browser WebGPU onnxruntime-web playable neural game`
13. `tinyworlds github minimal world model Genie reproduction`
14. `neural network game engine running in web browser demo world model 2025 2026 tiny`
15. `"world model" "runs entirely in your browser" playable`
16. `"world model" "in the browser" playable neural simulation tensorflow.js OR onnx OR webgpu game`
17. `"world model" browser demo WebGPU playable Doom OR Pong OR Snake OR Mario neural network frames 2025`
18. `on-device real-time world model phone playable game diffusion small parameters 2026`
19. `learned neural simulator of CARLA drivable interactive world model predicts lidar telemetry`
20. `CARLA world model action-conditioned video prediction human drive neural simulator replace CARLA`
21. `Raw2Drive reinforcement learning aligned world models end-to-end autonomous driving CARLA arXiv`
22. `Wayve "Learning to Drive from a World Model" arXiv 2025`
23. `DrivingWorld OR "neural driving simulator" CARLA generative world model interactive closed-loop 2025 2026 arXiv`
24. `TinySkies three.js plane game globe github`
25. `"tinyskies" world model OR neural OR "learned simulator"`
26. `"world model" tiny playable in browser "no physics engine" on your device 2026`
27. `"world model" "onnxruntime-web" OR "ONNX Runtime Web" playable game neural`
28. `drive a car inside a neural network world model browser demo keyboard CarRacing dream`
29. `Copilot Gaming Experiences WHAMM Quake II real-time browser Microsoft 2025`
30. `Overworld Waypoint-1 local diffusion world model consumer hardware January 2026`
31. `arXiv 2026 world model "in-browser" OR "web browser" real-time interactive inference lightweight`
32. `"Diamond" world model "CS:GO" play locally; "Diamond" browser; "world model" demo huggingface space playable small`
33. `lightweight world model CARLA "million parameters" autoregressive rollout real-time`
34. `CARLA "world model" interactive "keyboard" drive learned simulator demo latent dynamics GRU VAE`
35. `OpenAI "Video generation models as world simulators" technical report 2024`
36. `Hafner "Mastering diverse control tasks through world models" Nature 2025 volume pages`
37. `driving world model playable in browser neural driving simulator web demo drive yourself generated`
38. `"GTA" OR "CARLA" neural network world model "play" GameGAN DriveGAN interactive demo drive keyboard real-time`

Primary pages fetched:
- drfeifei.substack.com (functional taxonomy); a16z.news (From Words to Worlds)
- worldmodels.github.io; madebyoll.in; njkumar.com
- huggingface.co/codelion/neural-drive-model; physicalai-bmi.org
- github.com/{GaganGutta/neural-game-engine, sentdex/GANTheftAuto, dannylimanseta/tinyskies, dweam-team/awesome-world-model-games}
- research.nvidia.com DriveGAN page
- deepmind.google Genie 2 / Genie 3 posts
- oasis-model.github.io; lucid.ai/notes/lucid-v1; MSR WHAMM article; wayve.ai Ghost Gym
- arXiv abs/html pages for 2311.11762, 2411.14499, 2510.16732, 2504.21853, 2607.06401, 2606.30292, 2104.15060, 2210.07729
- Crossref for the Nature DOIs; the GitHub and Hugging Face APIs for repository and model dates

**Limitations of this search:**
- No systematic Google Scholar, Semantic Scholar or DBLP sweep was done.
- Non-English sources were not searched.
- Hacker News, Reddit and X posts were covered only through the web index.
- Demos hosted without descriptive text may have been missed.

Re-run a scoped search right before submission.

---

## 5. References (BibTeX)

The arXiv entries were generated from arXiv API metadata. Non-arXiv entries were checked as described above; any that remain UNVERIFIED are marked in their `note` field.

```bibtex
% ---------- arXiv entries (metadata from arXiv API, 2026-09-27) ----------

@article{ha2018worldmodels,
  title   = {{World Models}},
  author  = {David Ha and Jürgen Schmidhuber},
  journal = {arXiv preprint arXiv:1803.10122},
  year    = {2018},
  eprint  = {1803.10122},
  archivePrefix = {arXiv},
}

@article{ha2018recurrent,
  title   = {{Recurrent World Models Facilitate Policy Evolution}},
  author  = {David Ha and Jürgen Schmidhuber},
  journal = {arXiv preprint arXiv:1809.01999},
  year    = {2018},
  eprint  = {1809.01999},
  archivePrefix = {arXiv},
  note    = {To appear at NIPS 2018, selected for an oral presentation. arXiv admin note: substantial text overlap with arXiv:1803.10122}
}

@article{hafner2019planet,
  title   = {{Learning Latent Dynamics for Planning from Pixels}},
  author  = {Danijar Hafner and Timothy Lillicrap and Ian Fischer and Ruben Villegas and David Ha and Honglak Lee and James Davidson},
  journal = {arXiv preprint arXiv:1811.04551},
  year    = {2018},
  eprint  = {1811.04551},
  archivePrefix = {arXiv},
  note    = {20 pages, 12 figures, 1 table}
}

@article{hafner2020dreamer,
  title   = {{Dream to Control: Learning Behaviors by Latent Imagination}},
  author  = {Danijar Hafner and Timothy Lillicrap and Jimmy Ba and Mohammad Norouzi},
  journal = {arXiv preprint arXiv:1912.01603},
  year    = {2019},
  eprint  = {1912.01603},
  archivePrefix = {arXiv},
  note    = {9 pages, 12 figures}
}

@article{hafner2021dreamerv2,
  title   = {{Mastering Atari with Discrete World Models}},
  author  = {Danijar Hafner and Timothy Lillicrap and Mohammad Norouzi and Jimmy Ba},
  journal = {arXiv preprint arXiv:2010.02193},
  year    = {2020},
  eprint  = {2010.02193},
  archivePrefix = {arXiv},
  note    = {Published at ICLR 2021. Website: https://danijar.com/dreamerv2}
}

@article{hafner2023dreamerv3,
  title   = {{Mastering Diverse Domains through World Models}},
  author  = {Danijar Hafner and Jurgis Pasukonis and Jimmy Ba and Timothy Lillicrap},
  journal = {arXiv preprint arXiv:2301.04104},
  year    = {2023},
  eprint  = {2301.04104},
  archivePrefix = {arXiv},
  note    = {Website: https://danijar.com/dreamerv3}
}

@article{schrittwieser2020muzero,
  title   = {{Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model}},
  author  = {Julian Schrittwieser and Ioannis Antonoglou and Thomas Hubert and Karen Simonyan and Laurent Sifre and Simon Schmitt and Arthur Guez and Edward Lockhart and Demis Hassabis and Thore Graepel and Timothy Lillicrap and David Silver},
  journal = {arXiv preprint arXiv:1911.08265},
  year    = {2019},
  eprint  = {1911.08265},
  archivePrefix = {arXiv},
}

@article{micheli2023iris,
  title   = {{Transformers are Sample-Efficient World Models}},
  author  = {Vincent Micheli and Eloi Alonso and François Fleuret},
  journal = {arXiv preprint arXiv:2209.00588},
  year    = {2022},
  eprint  = {2209.00588},
  archivePrefix = {arXiv},
  note    = {ICLR 2023 (notable top 5%)}
}

@article{alonso2024diamond,
  title   = {{Diffusion for World Modeling: Visual Details Matter in Atari}},
  author  = {Eloi Alonso and Adam Jelley and Vincent Micheli and Anssi Kanervisto and Amos Storkey and Tim Pearce and François Fleuret},
  journal = {arXiv preprint arXiv:2405.12399},
  year    = {2024},
  eprint  = {2405.12399},
  archivePrefix = {arXiv},
  note    = {NeurIPS 2024 (Spotlight)}
}

@article{robine2023twm,
  title   = {{Transformer-based World Models Are Happy With 100k Interactions}},
  author  = {Jan Robine and Marc Höftmann and Tobias Uelwer and Stefan Harmeling},
  journal = {arXiv preprint arXiv:2303.07109},
  year    = {2023},
  eprint  = {2303.07109},
  archivePrefix = {arXiv},
  note    = {Published as a conference paper at ICLR 2023. Code is available at https://github.com/jrobine/twm}
}

@article{zhang2023storm,
  title   = {{STORM: Efficient Stochastic Transformer based World Models for Reinforcement Learning}},
  author  = {Weipu Zhang and Gang Wang and Jian Sun and Yetian Yuan and Gao Huang},
  journal = {arXiv preprint arXiv:2310.09615},
  year    = {2023},
  eprint  = {2310.09615},
  archivePrefix = {arXiv},
}

@article{kaiser2020simple,
  title   = {{Model-Based Reinforcement Learning for Atari}},
  author  = {Lukasz Kaiser and Mohammad Babaeizadeh and Piotr Milos and Blazej Osinski and Roy H Campbell and Konrad Czechowski and Dumitru Erhan and Chelsea Finn and Piotr Kozakowski and Sergey Levine and Afroz Mohiuddin and Ryan Sepassi and George Tucker and Henryk Michalewski},
  journal = {arXiv preprint arXiv:1903.00374},
  year    = {2019},
  eprint  = {1903.00374},
  archivePrefix = {arXiv},
}

@article{oh2015action,
  title   = {{Action-Conditional Video Prediction using Deep Networks in Atari Games}},
  author  = {Junhyuk Oh and Xiaoxiao Guo and Honglak Lee and Richard Lewis and Satinder Singh},
  journal = {arXiv preprint arXiv:1507.08750},
  year    = {2015},
  eprint  = {1507.08750},
  archivePrefix = {arXiv},
  note    = {Published at NIPS 2015 (Advances in Neural Information Processing Systems 28)}
}

@article{chiappa2017recurrent,
  title   = {{Recurrent Environment Simulators}},
  author  = {Silvia Chiappa and Sébastien Racaniere and Daan Wierstra and Shakir Mohamed},
  journal = {arXiv preprint arXiv:1704.02254},
  year    = {2017},
  eprint  = {1704.02254},
  archivePrefix = {arXiv},
}

@article{finn2016unsupervised,
  title   = {{Unsupervised Learning for Physical Interaction through Video Prediction}},
  author  = {Chelsea Finn and Ian Goodfellow and Sergey Levine},
  journal = {arXiv preprint arXiv:1605.07157},
  year    = {2016},
  eprint  = {1605.07157},
  archivePrefix = {arXiv},
  note    = {To appear in NIPS '16; Video results, code, and data available at: http://www.sites.google.com/site/robotprediction}
}

@article{babaeizadeh2018sv2p,
  title   = {{Stochastic Variational Video Prediction}},
  author  = {Mohammad Babaeizadeh and Chelsea Finn and Dumitru Erhan and Roy H. Campbell and Sergey Levine},
  journal = {arXiv preprint arXiv:1710.11252},
  year    = {2017},
  eprint  = {1710.11252},
  archivePrefix = {arXiv},
}

@article{kim2020gamegan,
  title   = {{Learning to Simulate Dynamic Environments with GameGAN}},
  author  = {Seung Wook Kim and Yuhao Zhou and Jonah Philion and Antonio Torralba and Sanja Fidler},
  journal = {arXiv preprint arXiv:2005.12126},
  year    = {2020},
  eprint  = {2005.12126},
  archivePrefix = {arXiv},
  note    = {CVPR 2020}
}

@article{valevski2025gamengen,
  title   = {{Diffusion Models Are Real-Time Game Engines}},
  author  = {Dani Valevski and Yaniv Leviathan and Moab Arar and Shlomi Fruchter},
  journal = {arXiv preprint arXiv:2408.14837},
  year    = {2024},
  eprint  = {2408.14837},
  archivePrefix = {arXiv},
  note    = {ICLR 2025. Project page: https://gamengen.github.io/}
}

@article{bruce2024genie,
  title   = {{Genie: Generative Interactive Environments}},
  author  = {Jake Bruce and Michael Dennis and Ashley Edwards and Jack Parker-Holder and Yuge Shi and Edward Hughes and Matthew Lai and Aditi Mavalankar and Richie Steigerwald and Chris Apps and Yusuf Aytar and Sarah Bechtle and Feryal Behbahani and Stephanie Chan and Nicolas Heess and Lucy Gonzalez and Simon Osindero and Sherjil Ozair and Scott Reed and Jingwei Zhang and Konrad Zolna and Jeff Clune and Nando de Freitas and Satinder Singh and Tim Rocktäschel},
  journal = {arXiv preprint arXiv:2402.15391},
  year    = {2024},
  eprint  = {2402.15391},
  archivePrefix = {arXiv},
  note    = {https://sites.google.com/corp/view/genie-2024/}
}

@article{zhang2025matrixgame,
  title   = {{Matrix-Game: Interactive World Foundation Model}},
  author  = {Yifan Zhang and Chunli Peng and Boyang Wang and Puyi Wang and Qingcheng Zhu and Fei Kang and Biao Jiang and Zedong Gao and Eric Li and Yang Liu and Yahui Zhou},
  journal = {arXiv preprint arXiv:2506.18701},
  year    = {2025},
  eprint  = {2506.18701},
  archivePrefix = {arXiv},
  note    = {Technical Report}
}

@article{he2025matrixgame2,
  title   = {{Matrix-game 2.0: An open-source real-time and streaming interactive world model}},
  author  = {Xianglong He and Chunli Peng and Zexiang Liu and Boyang Wang and Yifan Zhang and Qi Cui and Fei Kang and Biao Jiang and Mengyin An and Yangyang Ren and Baixin Xu and Hao-Xiang Guo and Kaixiong Gong and Size Wu and Wei Li and Xuchen Song and Yang Liu and Yangguang Li and Yahui Zhou},
  journal = {arXiv preprint arXiv:2508.13009},
  year    = {2025},
  eprint  = {2508.13009},
  archivePrefix = {arXiv},
  note    = {Project Page: https://matrix-game-v2.github.io}
}

@article{yu2025gamefactory,
  title   = {{GameFactory: Creating New Games with Generative Interactive Videos}},
  author  = {Jiwen Yu and Yiran Qin and Xintao Wang and Pengfei Wan and Di Zhang and Xihui Liu},
  journal = {arXiv preprint arXiv:2501.08325},
  year    = {2025},
  eprint  = {2501.08325},
  archivePrefix = {arXiv},
  note    = {ICCV 2025 Highlight, Project Page: https://yujiwen.github.io/gamefactory}
}

@article{hu2023gaia1,
  title   = {{GAIA-1: A Generative World Model for Autonomous Driving}},
  author  = {Anthony Hu and Lloyd Russell and Hudson Yeo and Zak Murez and George Fedoseev and Alex Kendall and Jamie Shotton and Gianluca Corrado},
  journal = {arXiv preprint arXiv:2309.17080},
  year    = {2023},
  eprint  = {2309.17080},
  archivePrefix = {arXiv},
  note    = {Technical Report}
}

@article{russell2025gaia2,
  title   = {{GAIA-2: A Controllable Multi-View Generative World Model for Autonomous Driving}},
  author  = {Lloyd Russell and Anthony Hu and Lorenzo Bertoni and George Fedoseev and Jamie Shotton and Elahe Arani and Gianluca Corrado},
  journal = {arXiv preprint arXiv:2503.20523},
  year    = {2025},
  eprint  = {2503.20523},
  archivePrefix = {arXiv},
  note    = {Technical Report}
}

@article{wang2023drivedreamer,
  title   = {{DriveDreamer: Towards Real-world-driven World Models for Autonomous Driving}},
  author  = {Xiaofeng Wang and Zheng Zhu and Guan Huang and Xinze Chen and Jiagang Zhu and Jiwen Lu},
  journal = {arXiv preprint arXiv:2309.09777},
  year    = {2023},
  eprint  = {2309.09777},
  archivePrefix = {arXiv},
  note    = {Project Page: https://drivedreamer.github.io}
}

@article{wang2024drivewm,
  title   = {{Driving into the Future: Multiview Visual Forecasting and Planning with World Model for Autonomous Driving}},
  author  = {Yuqi Wang and Jiawei He and Lue Fan and Hongxin Li and Yuntao Chen and Zhaoxiang Zhang},
  journal = {arXiv preprint arXiv:2311.17918},
  year    = {2023},
  eprint  = {2311.17918},
  archivePrefix = {arXiv},
  note    = {Project page: https://drive-wm.github.io. Code: https://github.com/BraveGroup/Drive-WM}
}

@article{gao2024vista,
  title   = {{Vista: A Generalizable Driving World Model with High Fidelity and Versatile Controllability}},
  author  = {Shenyuan Gao and Jiazhi Yang and Li Chen and Kashyap Chitta and Yihang Qiu and Andreas Geiger and Jun Zhang and Hongyang Li},
  journal = {arXiv preprint arXiv:2405.17398},
  year    = {2024},
  eprint  = {2405.17398},
  archivePrefix = {arXiv},
  note    = {NeurIPS 2024. Code and model: https://github.com/OpenDriveLab/Vista, demo page: https://vista-demo.github.io}
}

@article{yang2024genad,
  title   = {{GenAD: Generalized Predictive Model for Autonomous Driving}},
  author  = {Jiazhi Yang and Shenyuan Gao and Yihang Qiu and Li Chen and Tianyu Li and Bo Dai and Kashyap Chitta and Penghao Wu and Jia Zeng and Ping Luo and Jun Zhang and Andreas Geiger and Yu Qiao and Hongyang Li},
  journal = {arXiv preprint arXiv:2403.09630},
  year    = {2024},
  eprint  = {2403.09630},
  archivePrefix = {arXiv},
  note    = {CVPR 2024 Highlight Paper. Dataset: https://github.com/OpenDriveLab/DriveAGI}
}

@article{zheng2024occworld,
  title   = {{OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving}},
  author  = {Wenzhao Zheng and Weiliang Chen and Yuanhui Huang and Borui Zhang and Yueqi Duan and Jiwen Lu},
  journal = {arXiv preprint arXiv:2311.16038},
  year    = {2023},
  eprint  = {2311.16038},
  archivePrefix = {arXiv},
  note    = {Code is available at: https://github.com/wzzheng/OccWorld}
}

@article{zhang2024copilot4d,
  title   = {{Copilot4D: Learning Unsupervised World Models for Autonomous Driving via Discrete Diffusion}},
  author  = {Lunjun Zhang and Yuwen Xiong and Ze Yang and Sergio Casas and Rui Hu and Raquel Urtasun},
  journal = {arXiv preprint arXiv:2311.01017},
  year    = {2023},
  eprint  = {2311.01017},
  archivePrefix = {arXiv},
  note    = {ICLR 2024}
}

@article{hu2022mile,
  title   = {{Model-Based Imitation Learning for Urban Driving}},
  author  = {Anthony Hu and Gianluca Corrado and Nicolas Griffiths and Zak Murez and Corina Gurau and Hudson Yeo and Alex Kendall and Roberto Cipolla and Jamie Shotton},
  journal = {arXiv preprint arXiv:2210.07729},
  year    = {2022},
  eprint  = {2210.07729},
  archivePrefix = {arXiv},
  note    = {NeurIPS 2022}
}

@article{li2024think2drive,
  title   = {{Think2Drive: Efficient Reinforcement Learning by Thinking in Latent World Model for Quasi-Realistic Autonomous Driving (in CARLA-v2)}},
  author  = {Qifeng Li and Xiaosong Jia and Shaobo Wang and Junchi Yan},
  journal = {arXiv preprint arXiv:2402.16720},
  year    = {2024},
  eprint  = {2402.16720},
  archivePrefix = {arXiv},
  note    = {Accepted by ECCV 2024}
}

@article{gao2024cardreamer,
  title   = {{CarDreamer: Open-Source Learning Platform for World Model based Autonomous Driving}},
  author  = {Dechen Gao and Shuangyu Cai and Hanchu Zhou and Hang Wang and Iman Soltani and Junshan Zhang},
  journal = {arXiv preprint arXiv:2405.09111},
  year    = {2024},
  eprint  = {2405.09111},
  archivePrefix = {arXiv},
  note    = {Dechen Gao, Shuangyu Cai, Hanchu Zhou, Hang Wang contributed equally}
}

@article{yang2024drivearena,
  title   = {{DriveArena: A Closed-loop Generative Simulation Platform for Autonomous Driving}},
  author  = {Xuemeng Yang and Licheng Wen and Yukai Ma and Jianbiao Mei and Xin Li and Tiantian Wei and Wenjie Lei and Daocheng Fu and Pinlong Cai and Min Dou and Botian Shi and Liang He and Yong Liu and Yu Qiao},
  journal = {arXiv preprint arXiv:2408.00415},
  year    = {2024},
  eprint  = {2408.00415},
  archivePrefix = {arXiv},
  note    = {19 pages, 9 figures}
}

@article{yang2023unisim,
  title   = {{UniSim: A Neural Closed-Loop Sensor Simulator}},
  author  = {Ze Yang and Yun Chen and Jingkang Wang and Sivabalan Manivasagam and Wei-Chiu Ma and Anqi Joyce Yang and Raquel Urtasun},
  journal = {arXiv preprint arXiv:2308.01898},
  year    = {2023},
  eprint  = {2308.01898},
  archivePrefix = {arXiv},
  note    = {CVPR 2023 Highlight. Project page: https://waabi.ai/research/unisim/}
}

@article{yang2024unisimlearning,
  title   = {{Learning Interactive Real-World Simulators}},
  author  = {Sherry Yang and Yilun Du and Kamyar Ghasemipour and Jonathan Tompson and Leslie Kaelbling and Dale Schuurmans and Pieter Abbeel},
  journal = {arXiv preprint arXiv:2310.06114},
  year    = {2023},
  eprint  = {2310.06114},
  archivePrefix = {arXiv},
  note    = {https://universal-simulator.github.io}
}

@article{zyrianov2024lidardm,
  title   = {{LidarDM: Generative LiDAR Simulation in a Generated World}},
  author  = {Vlas Zyrianov and Henry Che and Zhijian Liu and Shenlong Wang},
  journal = {arXiv preprint arXiv:2404.02903},
  year    = {2024},
  eprint  = {2404.02903},
  archivePrefix = {arXiv},
}

@article{bengio2015scheduled,
  title   = {{Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks}},
  author  = {Samy Bengio and Oriol Vinyals and Navdeep Jaitly and Noam Shazeer},
  journal = {arXiv preprint arXiv:1506.03099},
  year    = {2015},
  eprint  = {1506.03099},
  archivePrefix = {arXiv},
}

@article{duan2025worldscore,
  title   = {{WorldScore: A Unified Evaluation Benchmark for World Generation}},
  author  = {Haoyi Duan and Hong-Xing Yu and Sirui Chen and Li Fei-Fei and Jiajun Wu},
  journal = {arXiv preprint arXiv:2504.00983},
  year    = {2025},
  eprint  = {2504.00983},
  archivePrefix = {arXiv},
  note    = {ICCV 2025. Project website: https://haoyi-duan.github.io/WorldScore/ The first two authors contributed equally}
}

@article{kang2025howfar,
  title   = {{How Far is Video Generation from World Model: A Physical Law Perspective}},
  author  = {Bingyi Kang and Yang Yue and Rui Lu and Zhijie Lin and Yang Zhao and Kaixin Wang and Gao Huang and Jiashi Feng},
  journal = {arXiv preprint arXiv:2411.02385},
  year    = {2024},
  eprint  = {2411.02385},
  archivePrefix = {arXiv},
  note    = {ICML 2025}
}

@article{huang2024vbench,
  title   = {{VBench: Comprehensive Benchmark Suite for Video Generative Models}},
  author  = {Ziqi Huang and Yinan He and Jiashuo Yu and Fan Zhang and Chenyang Si and Yuming Jiang and Yuanhan Zhang and Tianxing Wu and Qingyang Jin and Nattapol Chanpaisit and Yaohui Wang and Xinyuan Chen and Limin Wang and Dahua Lin and Yu Qiao and Ziwei Liu},
  journal = {arXiv preprint arXiv:2311.17982},
  year    = {2023},
  eprint  = {2311.17982},
  archivePrefix = {arXiv},
  note    = {Equal contributions from first four authors. Project page: https://vchitect.github.io/VBench-project/ Code: https://github.com/Vchitect/VBench}
}

@article{li2025worldmodelbench,
  title   = {{WorldModelBench: Judging Video Generation Models As World Models}},
  author  = {Dacheng Li and Yunhao Fang and Yukang Chen and Shuo Yang and Shiyi Cao and Justin Wong and Michael Luo and Xiaolong Wang and Hongxu Yin and Joseph E. Gonzalez and Ion Stoica and Song Han and Yao Lu},
  journal = {arXiv preprint arXiv:2502.20694},
  year    = {2025},
  eprint  = {2502.20694},
  archivePrefix = {arXiv},
}

@article{kim2021drivegan,
  title   = {{DriveGAN: Towards a Controllable High-Quality Neural Simulation}},
  author  = {Seung Wook Kim and Jonah Philion and Antonio Torralba and Sanja Fidler},
  journal = {arXiv preprint arXiv:2104.15060},
  year    = {2021},
  eprint  = {2104.15060},
  archivePrefix = {arXiv},
  note    = {CVPR 2021 Oral}
}

@article{bogdoll2025muvo,
  title   = {{MUVO: A Multimodal Generative World Model for Autonomous Driving with Geometric Representations}},
  author  = {Daniel Bogdoll and Yitian Yang and Tim Joseph and Melih Yazgan and J. Marius Zöllner},
  journal = {arXiv preprint arXiv:2311.11762},
  year    = {2023},
  eprint  = {2311.11762},
  archivePrefix = {arXiv},
  note    = {Daniel Bogdoll and Yitian Yang contributed equally. Accepted for publication at IV 2025}
}

@article{khurana2023pointcloud,
  title   = {{Point Cloud Forecasting as a Proxy for 4D Occupancy Forecasting}},
  author  = {Tarasha Khurana and Peiyun Hu and David Held and Deva Ramanan},
  journal = {arXiv preprint arXiv:2302.13130},
  year    = {2023},
  eprint  = {2302.13130},
  archivePrefix = {arXiv},
  note    = {CVPR 2023. Project page: https://www.cs.cmu.edu/~tkhurana/ff4d/index.html Code: https://github.com/tarashakhurana/4d-occ-forecasting}
}

@article{zhang2024bevworld,
  title   = {{BEVWorld: A Multimodal World Simulator for Autonomous Driving via Scene-Level BEV Latents}},
  author  = {Yumeng Zhang and Shi Gong and Kaixin Xiong and Xiaoqing Ye and Xiaofan Li and Xiao Tan and Fan Wang and Jizhou Huang and Hua Wu and Haifeng Wang},
  journal = {arXiv preprint arXiv:2407.05679},
  year    = {2024},
  eprint  = {2407.05679},
  archivePrefix = {arXiv},
  note    = {10 pages}
}

@article{hafner2025dreamer4,
  title   = {{Training Agents Inside of Scalable World Models}},
  author  = {Danijar Hafner and Wilson Yan and Timothy Lillicrap},
  journal = {arXiv preprint arXiv:2509.24527},
  year    = {2025},
  eprint  = {2509.24527},
  archivePrefix = {arXiv},
  note    = {Website: https://danijar.com/dreamer4/}
}

@article{ayupov2026dreamforge,
  title   = {{DreamForge-World 0.1 Preview: A Low-Compute Real-Time Controllable World Model}},
  author  = {Daniyel Ayupov and Artur Markov-Tsoy},
  journal = {arXiv preprint arXiv:2606.30292},
  year    = {2026},
  eprint  = {2606.30292},
  archivePrefix = {arXiv},
  note    = {Project page: https://trydreamforge.com/}
}

@article{yang2024playgen,
  title   = {{Playable Game Generation}},
  author  = {Mingyu Yang and Junyou Li and Zhongbin Fang and Sheng Chen and Yangbin Yu and Qiang Fu and Wei Yang and Deheng Ye},
  journal = {arXiv preprint arXiv:2412.00887},
  year    = {2024},
  eprint  = {2412.00887},
  archivePrefix = {arXiv},
}

@article{guo2025mineworld,
  title   = {{MineWorld: a Real-Time and Open-Source Interactive World Model on Minecraft}},
  author  = {Junliang Guo and Yang Ye and Tianyu He and Haoyu Wu and Yushu Jiang and Tim Pearce and Jiang Bian},
  journal = {arXiv preprint arXiv:2504.08388},
  year    = {2025},
  eprint  = {2504.08388},
  archivePrefix = {arXiv},
  note    = {Technical report. Project page https://aka.ms/mineworld}
}

@article{chen2026definition,
  title   = {{A Definition and Roadmap for World Models}},
  author  = {Xinyuan Chen and Haoyu Guo and Shi Guo and Bingqi Jiang and Chunhua Shen and Xing Shen and Tianfan Xue and Yufei Xue and Mulin Yu and Weinan Zhang and Bin Zhao and Bowen Zhou and Ming Zhou},
  journal = {arXiv preprint arXiv:2607.06401},
  year    = {2026},
  eprint  = {2607.06401},
  archivePrefix = {arXiv},
  note    = {Technical report, 58 pages, 10 figures}
}

@article{li2025embodiedsurvey,
  title   = {{A Comprehensive Survey on World Models for Embodied AI}},
  author  = {Xinqing Li and Xin He and Le Zhang and Min Wu and Xiaoli Li and Yun Liu},
  journal = {arXiv preprint arXiv:2510.16732},
  year    = {2025},
  eprint  = {2510.16732},
  archivePrefix = {arXiv},
  note    = {https://github.com/Li-Zn-H/AwesomeWorldModels}
}

@article{ding2024understanding,
  title   = {{Understanding World or Predicting Future? A Comprehensive Survey of World Models}},
  author  = {Jingtao Ding and Yunke Zhang and Yu Shang and Jie Feng and Yuheng Zhang and Zefang Zong and Yuan Yuan and Hongyuan Su and Nian Li and Jinghua Piao and Yucheng Deng and Nicholas Sukiennik and Chen Gao and Fengli Xu and Yong Li},
  journal = {arXiv preprint arXiv:2411.14499},
  year    = {2024},
  eprint  = {2411.14499},
  archivePrefix = {arXiv},
  note    = {Extended version of the original ACM CSUR paper, 49 pages, 6 figures, 8 tables}
}

@article{yu2025igvsurvey,
  title   = {{A Survey of Interactive Generative Video}},
  author  = {Jiwen Yu and Yiran Qin and Haoxuan Che and Quande Liu and Xintao Wang and Pengfei Wan and Di Zhang and Kun Gai and Hao Chen and Xihui Liu},
  journal = {arXiv preprint arXiv:2504.21853},
  year    = {2025},
  eprint  = {2504.21853},
  archivePrefix = {arXiv},
}

@article{guan2024drivingsurvey,
  title   = {{World Models for Autonomous Driving: An Initial Survey}},
  author  = {Yanchen Guan and Haicheng Liao and Zhenning Li and Jia Hu and Runze Yuan and Yunjian Li and Guohui Zhang and Chengzhong Xu},
  journal = {arXiv preprint arXiv:2403.02622},
  year    = {2024},
  eprint  = {2403.02622},
  archivePrefix = {arXiv},
}

@article{zhu2024sora,
  title   = {{Is Sora a World Simulator? A Comprehensive Survey on General World Models and Beyond}},
  author  = {Zheng Zhu and Xiaofeng Wang and Wangbo Zhao and Chen Min and Bohan Li and Nianchen Deng and Min Dou and Yuqi Wang and Botian Shi and Kai Wang and Chi Zhang and Yang You and Zhaoxiang Zhang and Dawei Zhao and Liang Xiao and Jian Zhao and Jiwen Lu and Guan Huang},
  journal = {arXiv preprint arXiv:2405.03520},
  year    = {2024},
  eprint  = {2405.03520},
  archivePrefix = {arXiv},
  note    = {This survey will be regularly updated at: https://github.com/GigaAI-research/General-World-Models-Survey}
}

@article{tang2025gamecraft2,
  title   = {{Hunyuan-GameCraft-2: Instruction-following Interactive Game World Model}},
  author  = {Junshu Tang and Jiacheng Liu and Jiaqi Li and Longhuang Wu and Haoyu Yang and Penghao Zhao and Siruis Gong and Xiang Yuan and Shuai Shao and Linfeng Zhang and Qinglin Lu},
  journal = {arXiv preprint arXiv:2511.23429},
  year    = {2025},
  eprint  = {2511.23429},
  archivePrefix = {arXiv},
  note    = {Technical Report, Project page:https://hunyuan-gamecraft-2.github.io/, Demo:https://hunyuan.tencent.com/game/game-craft}
}

@article{xu2026worldroambench,
  title   = {{WorldRoamBench: An Open-World Benchmark for Long-Horizon Stability of Interactive World Models}},
  author  = {Ting-Bing Xu and Jiacheng Sui and Zhe Gao and Kewei Shi and Wenjin Yang and Zhicheng Liu and Zhaoxu Sun and Mingchao Sun and Hongyu Pan and Fan Jiang and Mu Xu and Qi Fan and Yang Gao and Yong Li and Baoquan Chen},
  journal = {arXiv preprint arXiv:2606.31672},
  year    = {2026},
  eprint  = {2606.31672},
  archivePrefix = {arXiv},
}

@article{ying2026wbench,
  title   = {{WBench: A Comprehensive Multi-turn Benchmark for Interactive Video World Model Evaluation}},
  author  = {Kaining Ying and Hengrui Hu and Siyu Ren and Jiamu Li and Fengjiao Chen and Ziwen Wang and Xuezhi Cao and Xunliang Cai and Henghui Ding},
  journal = {arXiv preprint arXiv:2605.25874},
  year    = {2026},
  eprint  = {2605.25874},
  archivePrefix = {arXiv},
  note    = {Technical report of WBench. Homepage: https://meituan-longcat.github.io/WBench/}
}

@article{hu2026wantofight,
  title   = {{WanToFight: Real-Time Generative Game Engine for Multi-Player Combat Interaction}},
  author  = {Li Hu and Guangyuan Wang and Peng Zhang and Bang Zhang},
  journal = {arXiv preprint arXiv:2607.12592},
  year    = {2026},
  eprint  = {2607.12592},
  archivePrefix = {arXiv},
  note    = {Project Page: https://humanaigc.github.io/wantofight/}
}

@article{zhao2026miniworld,
  title   = {{MiniWorld: Democratizing the Training of Video World Models from Scratch}},
  author  = {Yian Zhao and Ruochong Zheng and Hongcan Guo and Yu Yan and Jian Zhang and Jie Chen},
  journal = {arXiv preprint arXiv:2608.01127},
  year    = {2026},
  eprint  = {2608.01127},
  archivePrefix = {arXiv},
  note    = {19 pages}
}

@article{yang2025raw2drive,
  title   = {{Raw2Drive: Reinforcement Learning with Aligned World Models for End-to-End Autonomous Driving (in CARLA v2)}},
  author  = {Zhenjie Yang and Xiaosong Jia and Qifeng Li and Xue Yang and Maoqing Yao and Junchi Yan},
  journal = {arXiv preprint arXiv:2505.16394},
  year    = {2025},
  eprint  = {2505.16394},
  archivePrefix = {arXiv},
  note    = {Accepted by NeurIPS 2025}
}

@article{goff2025learning,
  title   = {{Learning to Drive from a World Model}},
  author  = {Mitchell Goff and Greg Hogan and George Hotz and Armand du Parc Locmaria and Kacper Raczy and Harald Schäfer and Adeeb Shihadeh and Weixing Zhang and Yassine Yousfi},
  journal = {arXiv preprint arXiv:2504.19077},
  year    = {2025},
  eprint  = {2504.19077},
  archivePrefix = {arXiv},
}

@article{yang2025resim,
  title   = {{ReSim: Reliable World Simulation for Autonomous Driving}},
  author  = {Jiazhi Yang and Kashyap Chitta and Shenyuan Gao and Long Chen and Yuqian Shao and Xiaosong Jia and Hongyang Li and Andreas Geiger and Xiangyu Yue and Li Chen},
  journal = {arXiv preprint arXiv:2506.09981},
  year    = {2025},
  eprint  = {2506.09981},
  archivePrefix = {arXiv},
  note    = {NeurIPS 2025 Spotlight. Project page: https://opendrivelab.com/ReSim}
}

@article{basant2026omnidreams,
  title   = {{NVIDIA OmniDreams: Real-Time Generative World Model for Closed-Loop Autonomous Vehicle Simulation}},
  author  = {Aarti Basant and Amlan Kar and Despoina Paschalidou and Fangyin Wei and Francesco Ferroni and Guillermo Garcia Cobo and Haithem Turki and Huan Ling and Jaewoo Seo and James Lucas and Jay Zhangjie Wu and Jialiang Wang and Jonathan Lorraine and Jun Gao and Kai He and Katarina Tothova and Kevin Xie and Michal Tyszkiewicz and Qi Wu and Riccardo de Lutio and Ruilong Li and Sanja Fidler and Seung Wook Kim and Tianchang Shen and Tianshi Cao and Tobias Pfaff and William Lew and Xindi Wu and Xuanchi Ren and Yifan Lu and Yuxuan Zhang and Zan Gojcic and Zian Wang},
  journal = {arXiv preprint arXiv:2606.03159},
  year    = {2026},
  eprint  = {2606.03159},
  archivePrefix = {arXiv},
  note    = {Research blog: https://research.nvidia.com/labs/sil/projects/omnidreams-blog/, GitHub: https://github.com/nv-tlabs/omni-dreams, Model weights: https://huggingfa}
}

@article{hu2024drivingworld,
  title   = {{DrivingWorld: Constructing World Model for Autonomous Driving via Video GPT}},
  author  = {Xiaotao Hu and Mingkai Jia and Xiaoyang Guo and Qian Zhang and Xiao-xiao Long and Wei Yin},
  journal = {arXiv preprint arXiv:2412.19505},
  year    = {2024},
  eprint  = {2412.19505},
  archivePrefix = {arXiv},
}

@article{feng2025drivingsurvey,
  title   = {{A Survey of World Models for Autonomous Driving}},
  author  = {Tuo Feng and Wenguan Wang and Yi Yang},
  journal = {arXiv preprint arXiv:2501.11260},
  year    = {2025},
  eprint  = {2501.11260},
  archivePrefix = {arXiv},
  note    = {Ongoing project. Paper list: https://github.com/FengZicai/AwesomeWMAD Benchmark: https://github.com/FengZicai/WMAD-Benchmarks}
}

@article{zeng2026latentdriving,
  title   = {{Latent World Models for Automated Driving: A Unified Taxonomy, Evaluation Framework, and Open Challenges}},
  author  = {Rongxiang Zeng and Yongqi Dong},
  journal = {arXiv preprint arXiv:2603.09086},
  year    = {2026},
  eprint  = {2603.09086},
  archivePrefix = {arXiv},
  note    = {17 pages, 6 figures, under review by IEEE Transactions on Intelligent Transportation Systems (IEEE-T-ITS)}
}

@article{yu2025position,
  title   = {{Position: Interactive Generative Video as Next-Generation Game Engine}},
  author  = {Jiwen Yu and Yiran Qin and Haoxuan Che and Quande Liu and Xintao Wang and Pengfei Wan and Di Zhang and Xihui Liu},
  journal = {arXiv preprint arXiv:2503.17359},
  year    = {2025},
  eprint  = {2503.17359},
  archivePrefix = {arXiv},
}

@article{che2024gamegenx,
  title   = {{GameGen-X: Interactive Open-world Game Video Generation}},
  author  = {Haoxuan Che and Xuanhua He and Quande Liu and Cheng Jin and Hao Chen},
  journal = {arXiv preprint arXiv:2411.00769},
  year    = {2024},
  eprint  = {2411.00769},
  archivePrefix = {arXiv},
  note    = {Homepage: https://gamegen-x.github.io/ Github: https://github.com/GameGen-X/GameGen-X}
}

@article{li2025gamecraft,
  title   = {{Hunyuan-GameCraft: High-dynamic Interactive Game Video Generation with Hybrid History Condition}},
  author  = {Jiaqi Li and Junshu Tang and Zhiyong Xu and Longhuang Wu and Yuan Zhou and Shuai Shao and Tianbao Yu and Zhiguo Cao and Qinglin Lu},
  journal = {arXiv preprint arXiv:2506.17201},
  year    = {2025},
  eprint  = {2506.17201},
  archivePrefix = {arXiv},
  note    = {Project page: https://hunyuan-gamecraft.github.io/}
}

@article{feng2024matrix,
  title   = {{The Matrix: Infinite-Horizon World Generation with Real-Time Moving Control}},
  author  = {Ruili Feng and Han Zhang and Zhantao Yang and Jie Xiao and Zhilei Shu and Zhiheng Liu and Andy Zheng and Yukun Huang and Yu Liu and Hongyang Zhang},
  journal = {arXiv preprint arXiv:2412.03568},
  year    = {2024},
  eprint  = {2412.03568},
  archivePrefix = {arXiv},
}

@article{kong2025survey3d4d,
  title   = {{3D and 4D World Modeling: A Survey}},
  author  = {Lingdong Kong and Yu Yang and Jianbiao Mei and Youquan Liu and Ao Liang and Dekai Zhu and Dongyue Lu and Wei Yin and Xiaotao Hu and Mingkai Jia and Junyuan Deng and Kaiwen Zhang and Yang Wu and Tianyi Yan and Shenyuan Gao and Song Wang and Linfeng Li and Liang Pan and Yong Liu and Jianke Zhu and Wei Tsang Ooi and Steven C. H. Hoi and Ziwei Liu},
  journal = {arXiv preprint arXiv:2509.07996},
  year    = {2025},
  eprint  = {2509.07996},
  archivePrefix = {arXiv},
  note    = {Survey; Project Page at https://worldbench.github.io/survey GitHub Repo at https://github.com/worldbench/awesome-3d-4d-world-models}
}

@article{yue2025roadmap,
  title   = {{Simulating the Visual World with Artificial Intelligence: A Roadmap}},
  author  = {Jingtong Yue and Ziqi Huang and Zhaoxi Chen and Xintao Wang and Pengfei Wan and Ziwei Liu},
  journal = {arXiv preprint arXiv:2511.08585},
  year    = {2025},
  eprint  = {2511.08585},
  archivePrefix = {arXiv},
  note    = {Project page: https://world-model-roadmap.github.io/ Github Repo: https://github.com/ziqihuangg/Awesome-From-Video-Generation-to-World-Model}
}

@article{qin2024worldsimbench,
  title   = {{WorldSimBench: Towards Video Generation Models as World Simulators}},
  author  = {Yiran Qin and Zhelun Shi and Jiwen Yu and Xijun Wang and Enshen Zhou and Lijun Li and Zhenfei Yin and Xihui Liu and Lu Sheng and Jing Shao and Lei Bai and Wanli Ouyang and Ruimao Zhang},
  journal = {arXiv preprint arXiv:2410.18072},
  year    = {2024},
  eprint  = {2410.18072},
  archivePrefix = {arXiv},
}

@article{micheli2024deltairis,
  title   = {{Efficient World Models with Context-Aware Tokenization}},
  author  = {Vincent Micheli and Eloi Alonso and François Fleuret},
  journal = {arXiv preprint arXiv:2406.19320},
  year    = {2024},
  eprint  = {2406.19320},
  archivePrefix = {arXiv},
  note    = {ICML 2024}
}

@article{ranzato2016sequence,
  title   = {{Sequence Level Training with Recurrent Neural Networks}},
  author  = {Marc'Aurelio Ranzato and Sumit Chopra and Michael Auli and Wojciech Zaremba},
  journal = {arXiv preprint arXiv:1511.06732},
  year    = {2015},
  eprint  = {1511.06732},
  archivePrefix = {arXiv},
}

@article{huang2025selfforcing,
  title   = {{Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion}},
  author  = {Xun Huang and Zhengqi Li and Guande He and Mingyuan Zhou and Eli Shechtman},
  journal = {arXiv preprint arXiv:2506.08009},
  year    = {2025},
  eprint  = {2506.08009},
  archivePrefix = {arXiv},
  note    = {NeurIPS 2025 spotlight. Project website: http://self-forcing.github.io/}
}

@article{chen2024diffusionforcing,
  title   = {{Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion}},
  author  = {Boyuan Chen and Diego Marti Monso and Yilun Du and Max Simchowitz and Russ Tedrake and Vincent Sitzmann},
  journal = {arXiv preprint arXiv:2407.01392},
  year    = {2024},
  eprint  = {2407.01392},
  archivePrefix = {arXiv},
  note    = {Project website: https://boyuan.space/diffusion-forcing}
}

@article{lambert2022compounding,
  title   = {{Investigating Compounding Prediction Errors in Learned Dynamics Models}},
  author  = {Nathan Lambert and Kristofer Pister and Roberto Calandra},
  journal = {arXiv preprint arXiv:2203.09637},
  year    = {2022},
  eprint  = {2203.09637},
  archivePrefix = {arXiv},
  note    = {25 pages, 19 figures}
}

@article{bamford2020neural,
  title   = {{Neural Game Engine: Accurate learning of generalizable forward models from pixels}},
  author  = {Chris Bamford and Simon Lucas},
  journal = {arXiv preprint arXiv:2003.10520},
  year    = {2020},
  eprint  = {2003.10520},
  archivePrefix = {arXiv},
}

@article{ye2025yan,
  title   = {{Yan: Foundational Interactive Video Generation}},
  author  = {Deheng Ye and Fangyun Zhou and Jiacheng Lv and Jianqi Ma and Jun Zhang and Junyan Lv and Junyou Li and Minwen Deng and Mingyu Yang and Qiang Fu and Wei Yang and Wenkai Lv and Yangbin Yu and Yewen Wang and Yonghang Guan and Zhihao Hu and Zhongbin Fang and Zhongqian Sun},
  journal = {arXiv preprint arXiv:2508.08601},
  year    = {2025},
  eprint  = {2508.08601},
  archivePrefix = {arXiv},
}

@article{talvitie2017selfcorrecting,
  title   = {{Self-Correcting Models for Model-Based Reinforcement Learning}},
  author  = {Erik Talvitie},
  journal = {arXiv preprint arXiv:1612.06018},
  year    = {2016},
  eprint  = {1612.06018},
  archivePrefix = {arXiv},
  note    = {Original paper appeared in Proceedings of the 31st AAAI Conference on Artificial Intelligence, 2017. This version incorporates the appendix into document (rathe}
}

@article{arai2024actbench,
  title   = {{ACT-Bench: Towards Action Controllable World Models for Autonomous Driving}},
  author  = {Hidehisa Arai and Keishi Ishihara and Tsubasa Takahashi and Yu Yamaguchi},
  journal = {arXiv preprint arXiv:2412.05337},
  year    = {2024},
  eprint  = {2412.05337},
  archivePrefix = {arXiv},
}

@article{motamed2025physicsiq,
  title   = {{Do generative video models understand physical principles?}},
  author  = {Saman Motamed and Laura Culp and Kevin Swersky and Priyank Jaini and Robert Geirhos},
  journal = {arXiv preprint arXiv:2501.09038},
  year    = {2025},
  eprint  = {2501.09038},
  archivePrefix = {arXiv},
}

@article{yang2024vidar,
  title   = {{Visual Point Cloud Forecasting enables Scalable Autonomous Driving}},
  author  = {Zetong Yang and Li Chen and Yanan Sun and Hongyang Li},
  journal = {arXiv preprint arXiv:2312.17655},
  year    = {2023},
  eprint  = {2312.17655},
  archivePrefix = {arXiv},
}

@article{kendall2019learning,
  title   = {{Learning to Drive in a Day}},
  author  = {Alex Kendall and Jeffrey Hawke and David Janz and Przemyslaw Mazur and Daniele Reda and John-Mark Allen and Vinh-Dieu Lam and Alex Bewley and Amar Shah},
  journal = {arXiv preprint arXiv:1807.00412},
  year    = {2018},
  eprint  = {1807.00412},
  archivePrefix = {arXiv},
  note    = {Further results and demo videos can be viewed at: https://wayve.ai/blog/l2diad}
}

@article{wang2026mechanistic,
  title   = {{A Mechanistic View on Video Generation as World Models: State and Dynamics}},
  author  = {Luozhou Wang and Zhifei Chen and Yihua Du and Dongyu Yan and Wenhang Ge and Guibao Shen and Xinli Xu and Leyi Wu and Man Chen and Tianshuo Xu and Peiran Ren and Xin Tao and Pengfei Wan and Ying-Cong Chen},
  journal = {arXiv preprint arXiv:2601.17067},
  year    = {2026},
  eprint  = {2601.17067},
  archivePrefix = {arXiv},
}

@article{li2025smallworlds,
  title   = {{SmallWorlds: Assessing Dynamics Understanding of World Models in Isolated Environments}},
  author  = {Xinyi Li and Zaishuo Xia and Weyl Lu and Chenjie Hao and Yubei Chen},
  journal = {arXiv preprint arXiv:2511.23465},
  year    = {2025},
  eprint  = {2511.23465},
  archivePrefix = {arXiv},
}

@article{ye2026mind,
  title   = {{MIND: Benchmarking Memory Consistency and Action Control in World Models}},
  author  = {Yixuan Ye and Xuanyu Lu and Yuxin Jiang and Yuchao Gu and Rui Zhao and Qiwei Liang and Jiachun Pan and Fengda Zhang and Weijia Wu and Alex Jinpeng Wang},
  journal = {arXiv preprint arXiv:2602.08025},
  year    = {2026},
  eprint  = {2602.08025},
  archivePrefix = {arXiv},
}

@article{hu2026zytworld,
  title   = {{ZYT-World: A Real-Time Controllable World Model for Closed-Loop Autonomous-Driving Simulation}},
  author  = {Boni Hu and Xiong Wei and Haoming Huang and Yong Huang and Chenbo Wang and Yi Yang and Jiancheng Wang and Ruicheng Zhu and Zhimin Yang and Guanglai Liu and Qiaowan Jin and Dongzhuo Wang and Haiwei Kuang and Jiajun Fan and Yue Wu and Jiaxin Wei and Hao Sun and Feihong Yan and Yuyao Zhou and Wei Bi and Kaixuan Wang and Zichao Guo and Xiaozhi Chen},
  journal = {arXiv preprint arXiv:2609.21712},
  year    = {2026},
  eprint  = {2609.21712},
  archivePrefix = {arXiv},
  note    = {v2: updated author list. Technical Report. Videos and additional results are available at zyt-aim.github.io/ZYT-World}
}

@article{khanzada2026dreaming,
  title   = {{Dreaming Across Towns: Semantic Rollout and Town-Adversarial Regularization for Zero-Shot Held-Out-Town Fixed-Route Driving in CARLA}},
  author  = {Feeza Khan Khanzada and Jaerock Kwon},
  journal = {arXiv preprint arXiv:2604.27994},
  year    = {2026},
  eprint  = {2604.27994},
  archivePrefix = {arXiv},
}

@article{liu2026interactive,
  title   = {{Towards Interactive Video World Modeling: Frontiers, Challenges, Benchmarks, and Future Trends}},
  author  = {Jiuming Liu and Chaojun Ni and Mengmeng Liu and Chensheng Peng and Fangjinhua Wang and Sitian Shen and Marc Pollefeys and Masayoshi Tomizuka and Ayush Tewari and Per Ola Kristensson},
  journal = {arXiv preprint arXiv:2606.01164},
  year    = {2026},
  eprint  = {2606.01164},
  archivePrefix = {arXiv},
  note    = {Under review. The GitHub repository is publicly available at: https://github.com/liujiuming123/Awesome-Interactive-World-Model}
}

@article{luo2026aiforgames,
  title   = {{AI for Games in the Foundation Model Era}},
  author  = {Meng Luo and Yanlin Li and Hao Li and Hongzhan Lin and Pengfei Zhou and Tianjie Ju and Ran Zhang and Yeying Jin and Mong-Li Lee and Wynne Hsu},
  journal = {arXiv preprint arXiv:2609.16679},
  year    = {2026},
  eprint  = {2609.16679},
  archivePrefix = {arXiv},
  note    = {120 pages, 27 figures, 21 tables. Project page: https://eurekaleo.github.io/awesome-ai-for-games}
}

@article{jiang2026abotworld,
  title   = {{ABot-World-0: Infinite Interactive World Rollout on a Single Desktop GPU}},
  author  = {Fan Jiang and Zhaoxu Sun and Mengchao Wang and Ziyu Zhu and Chiyu Wang and Yunpeng Zhang and Wenlin Liu and Yun Wang and Xue Zheng and Rui Sun and Junfeng Ni and Hongyu Pan and Zhongxu Sun and Fei Yu and Zengye Ge and Mengmeng Du and Nianfei Fan and Mingchao Sun and Yu Liu and  Yongchang and Yanqing Zhu and Jiahang Wang and Ning Ying and Yuze Xuan and Di Yang and Zhicheng Liu and Zhe Gao and Tingbing Xu and Jiacheng Sui and Wenjin Yang and Junnan Lai and Shufeng Liu and Yuan Liu and Zheng Zhou and Yingliang Peng and Dawei Cao and Kaifeng Sheng and Yuxiang Cai and Fei Lu and Mu Xu and Ning Guo},
  journal = {arXiv preprint arXiv:2607.19191},
  year    = {2026},
  eprint  = {2607.19191},
  archivePrefix = {arXiv},
}

@article{dosovitskiy2017carla,
  title   = {{CARLA: An Open Urban Driving Simulator}},
  author  = {Alexey Dosovitskiy and German Ros and Felipe Codevilla and Antonio Lopez and Vladlen Koltun},
  journal = {arXiv preprint arXiv:1711.03938},
  year    = {2017},
  eprint  = {1711.03938},
  archivePrefix = {arXiv},
  note    = {Published at the 1st Conference on Robot Learning (CoRL)}
}

% ---------- Non-arXiv / web sources ----------

@misc{li2026functional,
  author = {Fei-Fei Li},
  title  = {A Functional Taxonomy of World Models},
  howpublished = {Dr. Fei-Fei Li (Substack), \url{https://drfeifei.substack.com/p/a-functional-taxonomy-of-world-models}},
  year   = {2026},
  month  = jun,
  note   = {Published 3 June 2026. Accessed 2026-09-27}
}

@misc{li2025words,
  author = {Fei-Fei Li},
  title  = {From Words to Worlds: Spatial Intelligence is {AI}'s Next Frontier},
  howpublished = {Essay, \url{https://www.a16z.news/p/from-words-to-worlds-spatial-intelligence} (also on the author's Substack)},
  year   = {2025},
  month  = nov,
  note   = {Published 10 November 2025. Accessed 2026-09-27}
}

@article{hafner2025nature,
  author  = {Danijar Hafner and Jurgis Pasukonis and Jimmy Ba and Timothy Lillicrap},
  title   = {Mastering diverse control tasks through world models},
  journal = {Nature},
  volume  = {640},
  pages   = {647--653},
  year    = {2025},
  doi     = {10.1038/s41586-025-08744-2}
}

@article{schrittwieser2020nature,
  author  = {Julian Schrittwieser and Ioannis Antonoglou and Thomas Hubert and Karen Simonyan and Laurent Sifre and Simon Schmitt and Arthur Guez and Edward Lockhart and Demis Hassabis and Thore Graepel and Timothy Lillicrap and David Silver},
  title   = {Mastering {Atari}, {Go}, chess and shogi by planning with a learned model},
  journal = {Nature},
  volume  = {588},
  pages   = {604--609},
  year    = {2020},
  doi     = {10.1038/s41586-020-03051-4}
}

@article{kanervisto2025wham,
  author  = {Anssi Kanervisto and Dave Bignell and Linda Yilin Wen and Martin Grayson and Raluca Georgescu and Sergio Valcarcel Macua and Shan Zheng Tan and Tabish Rashid and Tim Pearce and Yuhan Cao and Abdelhak Lemkhenter and Chentian Jiang and Gavin Costello and Gunshi Gupta and Marko Tot and Shu Ishida and Tarun Gupta and Udit Arora and Ryen W. White and Sam Devlin and Cecily Morrison and Katja Hofmann},
  title   = {World and Human Action Models towards gameplay ideation},
  journal = {Nature},
  volume  = {638},
  pages   = {656--663},
  year    = {2025},
  doi     = {10.1038/s41586-025-08600-3}
}

@misc{ha2018interactive,
  author = {David Ha and J{\"u}rgen Schmidhuber},
  title  = {World Models (interactive article with in-browser demos)},
  howpublished = {\url{https://worldmodels.github.io}},
  year   = {2018},
  note   = {Demos built with p5.js and deeplearn.js; accessed 2026-09-27}
}

@misc{genie2_2024,
  author = {Jack Parker-Holder and Philip Ball and Jake Bruce and others},
  title  = {Genie 2: A Large-Scale Foundation World Model},
  howpublished = {Google DeepMind blog, \url{https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/}},
  year   = {2024},
  month  = dec,
  note   = {Published 4 December 2024; full author list abbreviated (21 contributors listed on page)}
}

@misc{genie3_2025,
  author = {Jack Parker-Holder and Shlomi Fruchter},
  title  = {Genie 3: A New Frontier for World Models},
  howpublished = {Google DeepMind blog, \url{https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/}},
  year   = {2025},
  month  = aug,
  note   = {Published 5 August 2025}
}

@misc{oasis2024,
  author = {{Decart} and {Etched}},
  title  = {Oasis: A Universe in a Transformer},
  howpublished = {\url{https://oasis-model.github.io/}; weights: \url{https://github.com/etched-ai/open-oasis}},
  year   = {2024},
  month  = oct,
  note   = {Released 31 October 2024}
}

@misc{lucid2024,
  author = {{Lucid}},
  title  = {Lucid v1: Minecraft World Model},
  howpublished = {\url{https://www.lucid.ai/notes/lucid-v1}},
  year   = {2024},
  month  = nov,
  note   = {Dated 2024-11-11 on page; authorship on page not listed (announced by Rami Seid)}
}

@misc{whamm2025,
  author = {{Microsoft Research Game Intelligence}},
  title  = {{WHAMM!} Real-time world modelling of interactive environments},
  howpublished = {\url{https://www.microsoft.com/en-us/research/articles/whamm-real-time-world-modelling-of-interactive-environments/}},
  year   = {2025},
  month  = apr,
  note   = {Published 4 April 2025}
}

@techreport{mariovgg2024,
  author      = {{Virtuals Protocol}},
  title       = {Video Game Generation: A Practical Study using {Mario}},
  institution = {Virtuals Protocol},
  year        = {2024},
  url         = {https://virtual-protocol.github.io/mario-videogamegen/static/pdfs/VideoGameGen.pdf},
  note        = {MarioVGG. No arXiv ID found; individual author names UNVERIFIED}
}

@misc{boerbohan2025world,
  author = {Ollin Boer Bohan},
  title  = {World Emulation via Neural Network},
  howpublished = {\url{https://madebyoll.in/posts/world_emulation_via_dnn/}},
  year   = {2025},
  month  = apr,
  note   = {Published 25 April 2025; runs in-browser via ONNX Runtime Web}
}

@misc{njkumar2025flappy,
  author = {{njkumar}},
  title  = {Building real-time, on-device generative world models (Flappy Bird world model in the browser)},
  howpublished = {\url{https://www.njkumar.com/optimizing-flappy-bird-world-model-to-run-in-a-web-browser/}},
  year   = {2025},
  month  = jul,
  note   = {Published 23 July 2025; author full name UNVERIFIED}
}

@misc{codelion2026neuraldrive,
  author = {{codelion}},
  title  = {neural-drive-model: an action-conditioned world model for {SuperTuxKart}},
  howpublished = {\url{https://huggingface.co/codelion/neural-drive-model}; demo \url{https://huggingface.co/spaces/codelion/neural-drive}},
  year   = {2026},
  note   = {Hugging Face createdAt 2026-07-19}
}

@misc{physicalai2026nano,
  author = {{physicalai-bmi}},
  title  = {A neural world model, playable (nano-world-model)},
  howpublished = {\url{https://physicalai-bmi.org/research/world-model}; \url{https://huggingface.co/physicalai-bmi/nano-world-model-multi}},
  year   = {2026},
  note   = {HF createdAt 2026-07-13. Institutional provenance UNVERIFIED}
}

@misc{gutta2026neuralgameengine,
  author = {{GaganGutta}},
  title  = {neural-game-engine: Play a game inside a neural network},
  howpublished = {\url{https://github.com/GaganGutta/neural-game-engine}},
  year   = {2026},
  note   = {Repository created 2026-08-17}
}

@misc{almondgod2025tinyworlds,
  author = {{AlmondGod}},
  title  = {tinyworlds: A minimal implementation of DeepMind's Genie world model},
  howpublished = {\url{https://github.com/AlmondGod/tinyworlds}},
  year   = {2025},
  note   = {Repository created 2025-05-11}
}

@misc{sentdex2021gantheftauto,
  author = {{sentdex}},
  title  = {{GAN Theft Auto}},
  howpublished = {\url{https://github.com/sentdex/GANTheftAuto}},
  year   = {2021},
  note   = {Fork of GameGAN; repository created 2021-06-11}
}

@misc{overworld2026waypoint,
  author = {{Overworld}},
  title  = {Waypoint-1 / Waypoint-1.5: real-time local diffusion world models},
  howpublished = {\url{https://over.world/waypoint-1.5}; \url{https://huggingface.co/collections/Overworld/waypoint-1}},
  year   = {2026},
  note   = {Waypoint-1 research preview announced 20 January 2026; Waypoint-1.5 on 9 April 2026 (press release)}
}

@misc{wayve2023ghostgym,
  author = {{Wayve}},
  title  = {Ghost Gym: A Neural Simulator for Autonomous Driving},
  howpublished = {\url{https://wayve.ai/thinking/ghost-gym-neural-simulator/}},
  year   = {2023},
  month  = dec,
  note   = {Published 21 December 2023}
}

@misc{openai2024sora,
  author = {{OpenAI}},
  title  = {Video generation models as world simulators},
  howpublished = {\url{https://openai.com/index/video-generation-models-as-world-simulators/}},
  year   = {2024},
  note   = {Technical report. Individual author list UNVERIFIED (page returned HTTP 403 to automated fetch)}
}

@misc{tinyskies,
  author = {Danny Limanseta},
  title  = {TinySkies (GlobeFly): multiplayer Three.js globe flight game},
  howpublished = {\url{https://github.com/dannylimanseta/tinyskies}; live at \url{https://tinyskies.vercel.app}},
  year   = {2026},
  note   = {Repository created 2026-04-03; no license detected by GitHub API as of 2026-09-27. Author real name inferred from handle: UNVERIFIED}
}

@misc{mirage2_2025,
  author = {{Dynamics Lab}},
  title  = {Mirage 2: real-time generative game world engine},
  year   = {2025},
  note   = {UNVERIFIED: only secondary press coverage (Aug 2025) located; no primary technical report found}
}
```
