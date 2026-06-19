"""Dump every wandb run in a project to a flat CSV (name + config + numeric summary).

Run this on a machine with `wandb login` done, once per project, then commit the CSVs.
The plotting code reads these CSVs offline, so figures no longer depend on wandb access.

Usage:
    python dump_runs.py jingyuan-sun03-tu-delft/cl_ablation figures/ablation/dump_gcn.csv
    python dump_runs.py johnario-tu-delft/cl_ablation     figures/ablation/dump_baselines.csv
"""

import sys
from pathlib import Path

import pandas as pd
import wandb


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    project = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "wandb_dump.csv"

    api = wandb.Api()
    rows = []
    for r in api.runs(project):
        row = {"project": project, "run_name": r.name, "run_id": r.id, "state": r.state}
        # full config (prefixed so it never collides with a metric key)
        for k, v in r.config.items():
            row[f"cfg.{k}"] = v
        # numeric summary metrics only (eval/*, train/*, etc.)
        for k, v in r.summary.items():
            if isinstance(v, (int, float)):
                row[k] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{len(df)} runs from {project} -> {out}")
    if not df.empty:
        print("columns:", [c for c in df.columns if c.startswith("eval/")][:12], "...")


if __name__ == "__main__":
    main()
