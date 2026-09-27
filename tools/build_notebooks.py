"""Generate and execute the results notebooks (maintainer tool).

    python tools/build_notebooks.py [training_curves rollout_eval latency]

Each notebook reads only files inside results/ and writes its figures to results/figures/.
"""

import sys
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

RESULTS = Path(__file__).resolve().parents[1] / "results"
md = nbformat.v4.new_markdown_cell
code = nbformat.v4.new_code_cell

SETUP = """\
import json, sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "..")
from eval import plotstyle as ps

ps.apply()
FIGURES = Path("figures")
FIGURES.mkdir(exist_ok=True)
pd.set_option("display.precision", 4)
"""

TRAINING_CURVES = [
    md(
        "# Training curves: TinySkies\n\n"
        "Metric histories of the runs that produced the released TinySkies models, exported from the original "
        "training logs into `results/<run_id>/` (see `PROVENANCE.md`). The CARLA `world_v3` history was not "
        "recoverable; its only surviving record is a screenshot, shown at the end.\n\n"
        "Both TinySkies models come from **resumed run chains**: each run continues from the previous run's checkpoint."
    ),
    code(SETUP),
    code(
        """\
def load(run):
    h = pd.read_csv(Path(run) / "history.csv")
    return h, json.loads((Path(run) / "config.json").read_text())

VAE_CHAIN = ["3ixb3s35", "y0lnd3md", "yvxlfm52"]
SIM_STAGE1 = ["jl1spotp", "0b8k4g16", "u8qj3r6l"]          # VAE epoch 6, rollout horizon 13
SIM_STAGE2 = ["7mi5pz2h", "dukugksc"]                      # VAE epoch 15, rollout horizon 128 (resumed at epoch 26)
DEPLOYED_SIM_EPOCH, DEPLOYED_VAE_EPOCHS = 40, 14

def chain(runs):
    return pd.concat([load(r)[0].assign(run=r) for r in runs], ignore_index=True)

vae, s1, s2 = chain(VAE_CHAIN), chain(SIM_STAGE1), chain(SIM_STAGE2)
pd.DataFrame(
    [
        {"model": "VAE", "runs": " → ".join(VAE_CHAIN), "epochs": f"{vae['progress/epoch'].min():.0f}–{vae['progress/epoch'].max():.0f}",
         "steps": f"{vae['progress/batch_step'].max():,.0f}", "note": "deployed = weights after epoch 14"},
        {"model": "latent sim, stage 1", "runs": " → ".join(SIM_STAGE1), "epochs": f"{s1['progress/epoch'].min():.0f}–{s1['progress/epoch'].max():.0f}",
         "steps": f"{s1['progress/batch_step'].max():,.0f}", "note": "VAE epoch 6 latents, horizon 13"},
        {"model": "latent sim, stage 2", "runs": " → ".join(SIM_STAGE2), "epochs": f"{s2['progress/epoch'].min():.0f}–{s2['progress/epoch'].max():.0f}",
         "steps": f"{s2['progress/batch_step'].max():,.0f}", "note": "VAE epoch 15 latents, horizon 128; deployed = epoch 40"},
    ]
)"""
    ),
    md("## VAE reconstruction loss\n\nL1 reconstruction (the KL term has weight 1e-5). Validation uses held-out sessions."),
    code(
        """\
fig, ax = plt.subplots(figsize=(7, 3.4))
for split, color in (("train", ps.BLUE), ("val", ps.ORANGE)):
    d = vae.dropna(subset=[f"{split}/recon_loss"])
    ax.plot(d["progress/batch_step"], d[f"{split}/recon_loss"].rolling(15, min_periods=1).mean(), color=color, label=split)
end14 = vae.loc[vae["progress/epoch_completed"] == DEPLOYED_VAE_EPOCHS, "_step"]
if len(end14):
    ax.axvline(end14.iloc[0], color=ps.MUTED, ls="--", lw=1)
    ax.text(end14.iloc[0], ax.get_ylim()[1], " released", color=ps.INK_2, va="top", fontsize=8)
ax.set(xlabel="training step", ylabel="L1 reconstruction", title="TinySkies VAE")
ax.legend()
fig.savefig(FIGURES / "tinyskies_vae_loss.png")"""
    ),
    md(
        "## Latent simulator: one-step vs. end-of-rollout error\n\n"
        "Every training step unrolls the model on its own predictions. `one_step` is the latent MSE of the first "
        "prediction, `final_step` of the last one (step 13 in stage 1, step 128 in stage 2). The gap between them "
        "is the compounding error that a one-step metric hides. Stage 1 and stage 2 losses are not directly "
        "comparable: the horizon and the latent space (VAE epoch 6 vs. 15) both change at epoch 26."
    ),
    code(
        """\
fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), sharey=False)
for ax, d, title in ((axes[0], s1, "stage 1: horizon 13, VAE epoch 6"), (axes[1], s2, "stage 2: horizon 128, VAE epoch 15")):
    v = d.dropna(subset=["val/one_step_latent_loss"])
    ax.plot(v["progress/batch_step"], v["val/one_step_latent_loss"].rolling(9, min_periods=1).mean(), color=ps.BLUE, label="one step")
    ax.plot(v["progress/batch_step"], v["val/final_step_latent_loss"].rolling(9, min_periods=1).mean(), color=ps.ORANGE, label="last rollout step")
    ax.set(xlabel="training step", ylabel="val latent MSE", title=title, yscale="log")
    ax.legend()
released_step = s2.loc[s2["progress/epoch_completed"] == DEPLOYED_SIM_EPOCH, "_step"].iloc[0]
axes[1].axvline(released_step, color=ps.MUTED, ls="--", lw=1)
axes[1].text(released_step, axes[1].get_ylim()[1], " released", color=ps.INK_2, va="top", fontsize=8)
fig.savefig(FIGURES / "tinyskies_sim_one_vs_final.png")"""
    ),
    code(
        """\
rows = []
for name, d in (("stage 1", s1), ("stage 2", s2)):
    v = d.dropna(subset=["val/latent_loss"])
    last = v[v["progress/epoch"] >= v["progress/epoch"].max() - 1]
    rows.append({"stage": name, "epochs": f"{last['progress/epoch'].min():.0f}–{last['progress/epoch'].max():.0f}",
                 "val one-step": last["val/one_step_latent_loss"].mean(), "val last-step": last["val/final_step_latent_loss"].mean(),
                 "ratio last/one": last["val/final_step_latent_loss"].mean() / last["val/one_step_latent_loss"].mean(),
                 "val mean over rollout": last["val/latent_loss"].mean()})
at = s2[(s2["progress/epoch"] >= DEPLOYED_SIM_EPOCH - 1) & (s2["progress/epoch"] < DEPLOYED_SIM_EPOCH)].dropna(subset=["val/latent_loss"])
rows.append({"stage": "released (epoch 40)", "epochs": "39–40", "val one-step": at["val/one_step_latent_loss"].mean(),
             "val last-step": at["val/final_step_latent_loss"].mean(),
             "ratio last/one": at["val/final_step_latent_loss"].mean() / at["val/one_step_latent_loss"].mean(),
             "val mean over rollout": at["val/latent_loss"].mean()})
table = pd.DataFrame(rows)
table.to_csv(FIGURES / "tinyskies_training_table.csv", index=False)
table"""
    ),
    md(
        "## CARLA `world_v3` (screenshot only)\n\n"
        "The run history of the released CARLA model was not recoverable. This W&B panel screenshot "
        "(evidence class **Q**) was taken on 2026-05-18, after the epoch-20 checkpoint was exported, and may show a "
        "run that continued past it. It is shown for completeness; no number in the paper is read from it."
    ),
    code(
        """\
from PIL import Image
img = Image.open("carla_world_v3/wandb_train_panel_screenshot.png")
fig, ax = plt.subplots(figsize=(12, 6))
ax.imshow(img); ax.axis("off")"""
    ),
]


def build(name, cells):
    nb = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
    ExecutePreprocessor(timeout=1200, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(RESULTS)}})
    nbformat.write(nb, RESULTS / f"{name}.ipynb")
    print("wrote", RESULTS / f"{name}.ipynb")


ROLLOUT_EVAL = [
    md(
        "# Rollout evaluation: one-step prediction vs. sustained simulation\n\n"
        "All numbers come from `eval/carla_eval.py`, `eval/carla_sensitivity.py` and `eval/tinyskies_eval.py`, "
        "run on the released ONNX models. Outputs live in `results/eval/`. Horizons are in model steps: "
        "CARLA runs at 20 Hz (1 step = 50 ms); TinySkies data averages about 7.5 fps (1 step ≈ 133 ms)."
    ),
    code(SETUP + "E = Path('eval')\nROLE_ORDER = ['train', 'val', 'unseen']"),
    md(
        "## CARLA: replay against the real recording\n\n"
        "Each rollout starts from a real recorded state, receives the recorded controls, and then consumes only its own "
        "predictions for 100 steps (5 s). `one step` feeds the true state at every step (the training-style metric). "
        "`copy` holds the seed frame. Recording roles: *train* (a world_v3 training recording), *val* (a held-out "
        "validation recording), *unseen* (recorded after training, with a different driving style)."
    ),
    code(
        """\
rep = pd.read_csv(E / "carla/replay.csv.gz")
rep["speed_abs_err"] = (rep.speed_pred_mps - rep.speed_true_mps).abs()
fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
for role in ROLE_ORDER:
    g = rep[rep.role == role].groupby("k").mean(numeric_only=True)
    c = ps.ROLE_COLORS[role]
    axes[0].plot(g.index / 20, g.rgb_ssim, color=c, label=f"{ps.ROLE_LABELS[role]}: rollout")
    axes[0].plot(g.index / 20, g.one_step_ssim, color=c, ls=":", lw=1.5, label="one step (true state in)")
    axes[0].plot(g.index / 20, g.copy_ssim, color=c, ls="--", lw=1, label="copy seed frame")
    axes[1].plot(g.index / 20, g.speed_abs_err, color=c, label=ps.ROLE_LABELS[role])
    axes[2].plot(g.index / 20, g.waypoints_mae, color=c, label=ps.ROLE_LABELS[role])
axes[0].set(xlabel="seconds of free rollout", ylabel="SSIM vs. real frame", title="RGB")
axes[1].set(xlabel="seconds of free rollout", ylabel="|speed error| (m/s)", title="predicted speed (telemetry)", yscale="log")
axes[2].set(xlabel="seconds of free rollout", ylabel="MAE (normalized coords)", title="predicted route waypoints")
handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(handles[:3], ["rollout", "one step (true state in)", "copy seed frame"], loc="lower left")
axes[1].legend(loc="upper left")
fig.savefig(FIGURES / "carla_replay.png")"""
    ),
    code(
        """\
rows = []
for role in ROLE_ORDER:
    for k in (1, 20, 100):
        d = rep[(rep.role == role) & (rep.k == k)]
        rows.append({"recording": role, "horizon": f"{k} ({k / 20:.2f} s)", "rollouts": d.start.nunique() * d.recording.nunique(),
                     "SSIM rollout": d.rgb_ssim.mean(), "SSIM one-step": d.one_step_ssim.mean(), "SSIM copy": d.copy_ssim.mean(),
                     "true speed (m/s)": d.speed_true_mps.mean(), "|speed err| (m/s)": d.speed_abs_err.mean(),
                     "waypoint MAE": d.waypoints_mae.mean(),
                     "sharpness pred / true": d.sharpness_pred.mean() / d.sharpness_true.mean()})
carla_table = pd.DataFrame(rows)
carla_table.to_csv(FIGURES / "carla_replay_table.csv", index=False)
carla_table"""
    ),
    md("### Turning vs. straight driving (all recordings, recorded |steer| > 0.1 at the step)"),
    code(
        """\
rep["turning"] = rep.steer.abs() > 0.1
turn = rep[rep.k.isin([10, 40, 100])].groupby(["k", "turning"])[["rgb_ssim", "speed_abs_err", "waypoints_mae"]].mean()
turn.to_csv(FIGURES / "carla_turning_table.csv")
turn"""
    ),
    md(
        "## CARLA: which inputs does the model use?\n\n"
        "Mean absolute change of each output when one input is taken from a different real frame (*shuffle*). "
        "The structured-state head never sees the image. The LiDAR latent barely moves the outputs; `eval/carla_lidar_probe.py` "
        "shows that it encodes scan information in a tiny range, but that this information is not propagated by the dynamics."
    ),
    code(
        """\
sens = json.loads((E / "carla/sensitivity.json").read_text())
shuffle = pd.DataFrame(sens["shuffle"]).T.rename(columns=lambda c: c.replace("next_", "Δ next "))
shuffle.index.name = "shuffled input"
shuffle.to_csv(FIGURES / "carla_sensitivity_table.csv")
display(shuffle)
probe = json.loads((E / "carla/lidar_probe.json").read_text())
pd.json_normalize(probe, sep=" / ").T.rename(columns={0: "value"})"""
    ),
    md(
        "## CARLA: action response and momentum\n\n"
        "Scripted inputs from the six browser seeds (mean over seeds). Predicted speed is decoded from telemetry "
        "(channel 0 = tanh(v / 20 m/s)); the plot clips at 60 m/s. The training data rarely exceeds about 10 m/s."
    ),
    code(
        """\
resp = pd.read_csv(E / "carla/action_response.csv.gz")
fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
for level, c in zip((0.0, 0.25, 0.5, 1.0), ps.SERIES):
    g = resp[resp.condition == f"throttle_{level}"].groupby("k").speed_mps.mean()
    axes[0].plot(g.index / 20, g.clip(upper=60), color=c, label=f"throttle {level}")
for cond, c in (("throttle_then_coast", ps.BLUE), ("throttle_then_brake", ps.ORANGE)):
    g = resp[resp.condition == cond].groupby("k").speed_mps.mean()
    axes[1].plot(g.index / 20, g.clip(upper=60), color=c, label=cond.replace("_", " "))
axes[1].axvline(3, color=ps.MUTED, ls="--", lw=1)
for steer, c in zip((-1.0, -0.5, 0.5, 1.0), ps.SERIES):
    g = resp[resp.condition == f"steer_{steer}"].groupby("k").waypoint_lateral_10.mean()
    axes[2].plot(g.index / 20, g, color=c, label=f"steer {steer:+}")
axes[0].set(xlabel="seconds", ylabel="predicted speed (m/s)", title="constant throttle")
axes[1].set(xlabel="seconds", ylabel="predicted speed (m/s)", title="throttle 3 s, then coast or brake")
axes[2].set(xlabel="seconds", ylabel="lateral offset of waypoint 10 (norm.)", title="steering (throttle 0.5)")
for ax in axes: ax.legend()
fig.savefig(FIGURES / "carla_action_response.png")"""
    ),
    md(
        "## CARLA: long free-running rollouts (100 s)\n\n"
        "Scripted constant inputs, real autopilot control sequences replayed open-loop (`recorded_controls`), and a "
        "simple controller that reads only the model's *predicted* speed and waypoints (`closed loop`). "
        "`frame change` is the mean absolute difference between consecutive frames: near zero means the image has frozen."
    ),
    code(
        """\
long = pd.read_csv(E / "carla/long_rollouts.csv.gz")
cl = pd.read_csv(E / "carla/closed_loop.csv.gz").assign(script=lambda d: "closed loop " + d.target_mps.astype(str) + " m/s")
both = pd.concat([long, cl], ignore_index=True)
summary = []
for script, d in both.groupby("script"):
    last = d[d.k > d.k.max() - 200]
    first_sat = d[d.telemetry_0 >= 0.999].groupby("seed").k.min() if "telemetry_0" in d else pd.Series(dtype=float)
    summary.append({"condition": script,
                    "speed at 100 s (m/s, median seed)": float(np.median(last.groupby("seed").speed_mps.mean())),
                    "seeds reaching speed ceiling": f"{first_sat.size}/{d.seed.nunique()}",
                    "median time to ceiling (s)": float(first_sat.median() / 20) if first_sat.size else np.nan,
                    "frame change, last 10 s": last.frame_change.mean(),
                    "seeds frozen (<1e-3), last 10 s": f"{(last.groupby('seed').frame_change.mean() < 1e-3).sum()}/{d.seed.nunique()}"})
long_table = pd.DataFrame(summary).set_index("condition")
long_table.to_csv(FIGURES / "carla_long_table.csv")
long_table"""
    ),
    code(
        """\
snap = {**np.load(E / "carla/long_rollout_snapshots.npz"),
        **{f"closed loop {k.split('_')[1]} m/s": v for k, v in np.load(E / "carla/closed_loop_snapshots.npz").items() if k != "snapshot_steps"}}
steps = list(snap.pop("snapshot_steps"))
show = ["recorded_controls", "recorded_controls_fast_first", "weave", "closed loop 3.0 m/s", "full_throttle"]
cols = [steps.index(k) for k in (0, 100, 200, 400, 1000, 2000)]  # step numbers; 100 steps = 5 s
seed = 2
fig, axes = plt.subplots(len(show), len(cols), figsize=(2 * len(cols), 2.1 * len(show)))
for r, name in enumerate(show):
    for c, j in enumerate(cols):
        ax = axes[r, c]; ax.imshow(snap[name][seed, j]); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        if r == 0: ax.set_title(f"{steps[j] // 20} s" if steps[j] else "seed")
        if c == 0: ax.set_ylabel(name.replace("_", " "), fontsize=9)
fig.savefig(FIGURES / "carla_long_snapshots.png")"""
    ),
    md(
        "## TinySkies: replay against the real session\n\n"
        "Seeded with 4 real frames and their controls, then 64 steps of latent-only rollout under the recorded keys. "
        "`dyn` compares against the VAE reconstruction of the real frame, which removes reconstruction error and isolates "
        "dynamics error. The dashed line is the VAE ceiling (reconstruction of the real frame, no dynamics)."
    ),
    code(
        """\
vae = pd.read_csv(E / "tinyskies/vae_ceiling.csv.gz")
tr = pd.read_csv(E / "tinyskies/replay.csv.gz")
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for human, c, lab in ((False, ps.BLUE, "random-bot sessions"), (True, ps.ORANGE, "human sessions")):
    g = tr[tr.human == human].groupby("k").mean(numeric_only=True)
    axes[0].plot(g.index, g.latent_mse, color=c, label=f"{lab}: model")
    axes[0].plot(g.index, g.copy_latent_mse, color=c, ls="--", lw=1, label=f"{lab}: copy seed")
    axes[1].plot(g.index, g.dyn_ssim, color=c, label=f"{lab}: vs. VAE recon (dynamics only)")
    axes[1].plot(g.index, g.rgb_ssim, color=c, ls=":", lw=1.5, label=f"{lab}: vs. real frame")
axes[1].axhline(vae.ssim.mean(), color=ps.MUTED, ls="--", lw=1)
axes[1].text(64, vae.ssim.mean(), "VAE ceiling ", ha="right", va="bottom", fontsize=8, color=ps.INK_2)
axes[0].set(xlabel="rollout step", ylabel="latent MSE", title="latent error")
axes[1].set(xlabel="rollout step", ylabel="SSIM", title="decoded frames")
for ax in axes: ax.legend(fontsize=7.5)
fig.savefig(FIGURES / "tinyskies_replay.png")"""
    ),
    code(
        """\
rows = [{"horizon": "VAE ceiling", "SSIM vs real": vae.ssim.mean(), "PSNR": vae.psnr.mean(),
         "sharpness recon / true": vae.sharpness_recon.mean() / vae.sharpness_true.mean()}]
for k in (1, 8, 32, 64):
    d = tr[tr.k == k]
    rows.append({"horizon": f"{k}", "latent MSE": d.latent_mse.mean(), "latent MSE copy": d.copy_latent_mse.mean(),
                 "SSIM vs real": d.rgb_ssim.mean(), "SSIM vs recon (dynamics)": d.dyn_ssim.mean(),
                 "sharpness pred / true recon": d.sharpness_pred.mean() / d.sharpness_recon.mean()})
tiny_table = pd.DataFrame(rows)
tiny_table.to_csv(FIGURES / "tinyskies_replay_table.csv", index=False)
tiny_table"""
    ),
    md("## TinySkies: long free-running rollouts (2,000 steps from the 9 browser seeds)"),
    code(
        """\
tl = pd.read_csv(E / "tinyskies/long_rollouts.csv.gz")
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for (script, c) in zip(["random_bot", "forward", "none", "left"], ps.SERIES):
    g = tl[tl.script == script].groupby("k").mean(numeric_only=True)
    axes[0].plot(g.index, g.frame_change.rolling(20, min_periods=1).mean(), color=c, label=script)
    axes[1].plot(g.index, g.dist_from_seed, color=c, label=script)
axes[0].set(xlabel="step", ylabel="frame change", title="is the world still moving?", yscale="log")
axes[1].set(xlabel="step", ylabel="MAE from seed frame", title="how far has it travelled from the seed?")
for ax in axes: ax.legend()
fig.savefig(FIGURES / "tinyskies_long.png")
summary = []
for script, d in tl.groupby("script"):
    last = d[d.k > d.k.max() - 200]
    summary.append({"script": script, "frame change (last 200)": last.frame_change.mean(),
                    "seeds frozen (<1e-3)": f"{(last.groupby('seed').frame_change.mean() < 1e-3).sum()}/{d.seed.nunique()}",
                    "sharpness (last 200)": last.sharpness.mean(), "dist from seed": last.dist_from_seed.mean()})
tiny_long = pd.DataFrame(summary).set_index("script")
tiny_long.to_csv(FIGURES / "tinyskies_long_table.csv")
tiny_long"""
    ),
    code(
        """\
ts = dict(np.load(E / "tinyskies/long_rollout_snapshots.npz"))
steps = list(ts.pop("snapshot_steps"))
show, cols, seed = ["random_bot", "forward", "left", "none"], [steps.index(k) for k in (0, 100, 200, 500, 1000, 2000)], 0
fig, axes = plt.subplots(len(show), len(cols), figsize=(2 * len(cols), 2.1 * len(show)))
for r, name in enumerate(show):
    for c, j in enumerate(cols):
        ax = axes[r, c]; ax.imshow(ts[name][seed, j]); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        if r == 0: ax.set_title(f"step {steps[j]}" if steps[j] else "seed")
        if c == 0: ax.set_ylabel(name.replace("_", " "), fontsize=9)
fig.savefig(FIGURES / "tinyskies_long_snapshots.png")"""
    ),
]

LATENCY = [
    md(
        "# Size and speed\n\n"
        "How small and how fast the released simulators are. Python numbers come from `eval/latency.py` (onnxruntime, CPU). "
        "Browser numbers come from `eval/browser_bench/` (onnxruntime-web 1.22, the same runtime as the deployed demo, "
        "closed loop, batch 1, headless Chromium). All on one Apple M1 laptop."
    ),
    code(SETUP),
    code(
        """\
manifest = json.loads(Path("../artifacts/MANIFEST.json").read_text())
rows = []
for path, m in manifest.items():
    if path.endswith(".onnx"):
        rows.append({"model": path.replace("artifacts/", ""), "ONNX values (weights + shape constants)": m["params"], "fp32 size (MB)": m["bytes"] / 1e6})
sizes = pd.DataFrame(rows)
sizes.loc[len(sizes)] = {"model": "CARLA total", "ONNX values (weights + shape constants)": sizes[sizes.model.str.startswith("carla")]["ONNX values (weights + shape constants)"].sum(),
                         "fp32 size (MB)": sizes[sizes.model.str.startswith("carla")]["fp32 size (MB)"].sum()}
sizes.loc[len(sizes)] = {"model": "TinySkies total", "ONNX values (weights + shape constants)": sizes[sizes.model.str.startswith("tinyskies/")]["ONNX values (weights + shape constants)"].sum(),
                         "fp32 size (MB)": sizes[sizes.model.str.startswith("tinyskies/")]["fp32 size (MB)"].sum()}
sizes.to_csv(FIGURES / "model_sizes.csv", index=False)
sizes"""
    ),
    code(
        """\
py = json.loads((Path("eval") / "latency_python.json").read_text())
rows = []
for threads in ("threads_1", "threads_4"):
    for name, r in py[threads].items():
        rows.append({"runtime": f"python onnxruntime ({threads.replace('_', ' ')})", "measurement": name,
                     "median ms": r["median_ms"], "p90 ms": r["p90_ms"], "max steps/s": 1000 / r["median_ms"]})
br = json.loads((Path("eval") / "latency_browser.json").read_text())
for r in br["results"]:
    for key, label in (("carla_step", "carla_step (transition)"), ("tinyskies_step", "tinyskies_step (latent_sim + decoder)")):
        if key in r:
            rows.append({"runtime": f"browser {r['ep']} ({r['threads']} thread{'s' if r['threads'] > 1 else ''})",
                         "measurement": label, "median ms": r[key]["median_ms"], "p90 ms": r[key]["p90_ms"],
                         "max steps/s": 1000 / r[key]["median_ms"]})
        elif "error" in r:
            rows.append({"runtime": f"browser {r['ep']}", "measurement": label, "median ms": np.nan, "p90 ms": np.nan, "max steps/s": np.nan})
latency = pd.DataFrame(rows)
latency.to_csv(FIGURES / "latency_table.csv", index=False)
print(py["machine"], br["machine"])
latency"""
    ),
]

NOTEBOOKS = {"training_curves": TRAINING_CURVES, "rollout_eval": ROLLOUT_EVAL, "latency": LATENCY}

if __name__ == "__main__":
    for name in sys.argv[1:] or NOTEBOOKS:
        build(name, NOTEBOOKS[name])
