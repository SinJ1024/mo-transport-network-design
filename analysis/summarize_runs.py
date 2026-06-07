#!/usr/bin/env python3
"""Summarize wandb runs of the distance-ref / spatiotemporal-lambda ablation.

Pulls runs from a wandb project and produces:
  --table : a mean +/- std LaTeX table of FINAL metric values, grouped by config.
  --plot  : a median + 25/75-percentile band of a metric over training (across seeds).

Run names are expected like:
  motndp_xian-v0__GCN-Xian-<config>__<seed>__<timestamp>
  motndp_xian-v0__PCN-Xian__<seed>__<timestamp>

Usage:
  python analysis/summarize_runs.py --table
  python analysis/summarize_runs.py --plot eval/hypervolume --out figures/hv_band.png \
      --configs i3_l0 i3_spatial i3_curriculum i3_spatiotemporal
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

# Display/row order; also fixes plot colors.
ORDER = [
    "nd", "pareto", "nash",
    "i2_l0", "i2_l05", "i2_l1", "i2_spatial", "i2_curriculum", "i2_spatiotemporal",
    "i3_l0", "i3_l05", "i3_l1", "i3_spatial", "i3_curriculum", "i3_spatiotemporal",
    "pcn",
]

# Final-value metrics for the table (edit to taste).
TABLE_METRICS = [
    "eval/served_floor_median", "eval/demand_coverage_median",
    "eval/spatial_sw_ratio", "eval/price_equity_ratio_median",
    "eval/hypervolume", "eval/eum", "eval/cardinality",
    "eval/gini_median", "eval/sen_welfare_median", "eval/nash_welfare_geom_median",
]


def parse_run(run):
    """(config, seed) from the wandb run display name."""
    parts = run.name.split("__")
    if len(parts) < 3:
        return None, None
    exp, seed = parts[1], parts[2]
    cfg = exp.replace("GCN-Xian-", "").replace("PCN-Xian", "pcn")
    try:
        seed = int(seed)
    except ValueError:
        return None, None
    return cfg, seed


def fetch_runs(entity, project):
    api = wandb.Api()
    by_cfg = {}
    for r in api.runs(f"{entity}/{project}"):
        cfg, seed = parse_run(r)
        if cfg is None:
            continue
        by_cfg.setdefault(cfg, []).append(r)
    return by_cfg


def make_table(by_cfg, metrics=TABLE_METRICS):
    short = [m.replace("eval/", "").replace("_median", "") for m in metrics]
    print("\\begin{tabular}{l" + "c" * len(metrics) + "}")
    print("\\toprule")
    print("config & " + " & ".join(s.replace("_", "\\_") for s in short) + " \\\\ \\midrule")
    for cfg in [c for c in ORDER if c in by_cfg]:
        cells = []
        for m in metrics:
            vals = [r.summary.get(m) for r in by_cfg[cfg]]
            vals = [v for v in vals if isinstance(v, (int, float))]
            cells.append(f"{np.mean(vals):.3g} $\\pm$ {np.std(vals):.2g}" if vals else "--")
        print(cfg.replace("_", "\\_") + " & " + " & ".join(cells) + " \\\\")
    print("\\bottomrule\n\\end{tabular}")


def make_plot(by_cfg, metric, out, configs=None, n_bins=40):
    configs = configs or [c for c in ORDER if c in by_cfg]
    plt.figure(figsize=(7, 4.5))
    for cfg in configs:
        if cfg not in by_cfg:
            continue
        xs, ys = [], []
        for r in by_cfg[cfg]:
            h = r.history(keys=["global_step", metric], pandas=True).dropna()
            if len(h):
                xs.append(h["global_step"].values)
                ys.append(h[metric].values)
        if not xs:
            continue
        allx, ally = np.concatenate(xs), np.concatenate(ys)
        bins = np.linspace(allx.min(), allx.max(), n_bins + 1)
        idx = np.digitize(allx, bins)
        bx, med, lo, hi = [], [], [], []
        for b in range(1, n_bins + 1):
            sel = ally[idx == b]
            if len(sel) == 0:
                continue
            bx.append((bins[b - 1] + bins[b]) / 2)
            med.append(np.median(sel))
            lo.append(np.percentile(sel, 25))
            hi.append(np.percentile(sel, 75))
        line, = plt.plot(bx, med, label=cfg)
        plt.fill_between(bx, lo, hi, alpha=0.2, color=line.get_color())
    plt.xlabel("global_step")
    plt.ylabel(metric)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print("saved", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default="johnario-tu-delft")
    ap.add_argument("--project", default="cl_ablation")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--plot", default=None, help="metric key, e.g. eval/hypervolume")
    ap.add_argument("--configs", nargs="+", default=None, help="config subset for the plot")
    ap.add_argument("--out", default="figures/band.png")
    a = ap.parse_args()

    by_cfg = fetch_runs(a.entity, a.project)
    print(f"# project {a.entity}/{a.project}: "
          f"{ {c: len(v) for c, v in sorted(by_cfg.items())} }")
    if a.table:
        make_table(by_cfg)
    if a.plot:
        make_plot(by_cfg, a.plot, a.out, configs=a.configs)


if __name__ == "__main__":
    main()
