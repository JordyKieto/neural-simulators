"""Export a W&B run's full metric history into results/<run_id>/ (maintainer tool).

The paper repository never links to W&B. Every number in the paper is read from these
exported files instead:

    results/<run_id>/history.csv    one row per logged step (all scalar metrics)
    results/<run_id>/config.json    run config (secrets stripped)
    results/<run_id>/summary.json   final summary scalars
    results/<run_id>/run.json       entity/project/run, name, state, timestamps

Usage:
    python tools/export_wandb_run.py drift-ai-models/drift-tinyskies-latent-sim/dukugksc [...]

Needs `wandb` and an API key (WANDB_API_KEY or ~/.netrc).
"""

import argparse
import csv
import json
import math
from pathlib import Path

import wandb

RESULTS = Path(__file__).resolve().parents[1] / "results"
SECRET_HINTS = ("key", "secret", "token", "password")


def _scalar(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return None if isinstance(value, float) and math.isnan(value) else value
    if isinstance(value, str):
        return value
    return None


def _clean_config(config):
    return {
        k: v
        for k, v in config.items()
        if not k.startswith("_") and not any(h in k.lower() for h in SECRET_HINTS)
    }


def export(path, out_root=RESULTS):
    api = wandb.Api(timeout=120)
    run = api.run(path)
    out = out_root / run.id
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for row in run.scan_history(page_size=1000):
        rows.append({k: _scalar(v) for k, v in row.items() if _scalar(v) is not None})
    rows.sort(key=lambda r: r.get("_step", 0))
    columns = sorted({k for r in rows for k in r}, key=lambda k: (not k.startswith("_"), k))
    with open(out / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    summary = {k: _scalar(v) for k, v in run.summary._json_dict.items() if _scalar(v) is not None}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    (out / "config.json").write_text(json.dumps(_clean_config(run.config), indent=2, sort_keys=True))
    (out / "run.json").write_text(
        json.dumps(
            {
                "entity_project_run": path,
                "name": run.name,
                "state": run.state,
                "created_at": run.created_at,
                "heartbeat_at": getattr(run, "heartbeat_at", None),
                "exported_rows": len(rows),
            },
            indent=2,
        )
    )
    print(f"{path}: {len(rows)} rows, {len(columns)} columns -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="entity/project/run_id")
    parser.add_argument("--out", type=Path, default=RESULTS)
    args = parser.parse_args()
    for path in args.runs:
        export(path, args.out)
