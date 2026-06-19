"""Plot the full set of comparison figures from committed training logs (offline, no wandb).

The stdout logs do not contain the eval metrics (those went to wandb), but they DO
contain the per-step return vectors. Per run we rebuild an achieved front (non-dominated
set of the converged-regime return vectors, capped to FRONT_CAP points) and recompute the
GROUP-LEVEL metrics: HV (per-dim), EUM, Sen welfare, Gini, Efficiency.

Cell-level metrics (served_floor, demand_coverage, spatial_sw) are NOT recoverable from
logs and are omitted. These numbers are recomputed from training returns and will not
match the wandb eval-front figures exactly; they are internally consistent across methods.

Figures (per env) -> figures/from_logs/:
    bar_<env>_<metric>.png     one figure per metric (x=method, bars per G)
    bar_<env>.png              all metrics in one grid
    groups_<env>.png           metric vs nr_groups, one line per method
    <env>_methods_g<G>.png     fixed-G method comparison

Usage: python analysis_from_logs.py
"""

import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pymoo.indicators.hv import HV

ENVS = ["xian", "amsterdam"]
GROUPS = [3, 5, 7, 10]
METHODS = ["l0", "temporal", "spatial", "spatiotemporal", "pcn", "ncn"]
LABELS = {
    "l0": "GCN-Lorenz", "temporal": "GCN-Temporal", "spatial": "GCN-Spatial",
    "spatiotemporal": "GCN-Spatiotemporal", "pcn": "PCN", "ncn": "NCN (Nash)",
}
PALETTE = {
    "l0": "#4C72B0", "temporal": "#DD8452", "spatial": "#55A868",
    "spatiotemporal": "#C44E52", "pcn": "#8172B3", "ncn": "#937860",
}
LOG_DIRS = ["logs/gcn_ablation", "logs/baselines", "logs/gcn_nash"]
SAVEDIR = Path("figures/from_logs")
SAVEDIR.mkdir(parents=True, exist_ok=True)
FRONT_CAP = 20
RET_RE = re.compile(r"return\s*\[([^\]]*)\]", re.DOTALL)


# ---- metrics (match morl_baselines definitions) -------------------------------
def gini(x):
    x = np.asarray(x, float)
    n = x.shape[1]
    xs = np.sort(x, axis=1)
    cum = np.cumsum(xs, axis=1)
    tot = cum[:, -1]
    safe = np.where(tot == 0, 1.0, tot)
    g = (n + 1 - 2 * np.sum(cum, axis=1) / safe) / n
    g = np.where(tot == 0, 0.0, g)
    return g * (n / (n - 1)) if n > 1 else g


def non_dominated(pts):
    pts = np.asarray(pts, float)
    keep = np.ones(len(pts), bool)
    for i in range(len(pts)):
        if not keep[i]:
            continue
        dom = np.all(pts >= pts[i], axis=1) & np.any(pts > pts[i], axis=1)
        keep[i] = not dom.any()
    return pts[keep]


def cap_front(front):
    if len(front) <= FRONT_CAP:
        return front
    order = np.argsort(front.sum(1))
    idx = np.linspace(0, len(front) - 1, FRONT_CAP).astype(int)
    return front[order[idx]]


def hv_pdim(front, d):
    arr = np.asarray(front, float)
    if arr.size == 0:
        return 0.0
    dim_max = arr.max(axis=0)
    norm = np.where(dim_max > 0, dim_max, 1.0)
    nf = arr / norm
    eff_ref = np.full(d, -0.05)
    hv = HV(ref_point=eff_ref * -1)(nf * -1)
    return float(hv) ** (1.0 / d) if hv > 0 else 0.0


def eum(front, d, nw=64):
    fr = np.asarray(front, float)
    if fr.size == 0:
        return 0.0
    W = np.random.default_rng(0).dirichlet(np.ones(d), size=nw)
    return float(np.mean(np.max(fr @ W.T, axis=0)))


METRICS = {
    "HV (per-dim)": lambda fr, d: hv_pdim(fr, d),
    "EUM": lambda fr, d: eum(fr, d),
    # best front point: max for higher-is-better, min for Gini (lower-is-better)
    "Sen Welfare": lambda fr, d: float(np.max(fr.sum(1) * (1 - gini(fr)))),
    "Gini": lambda fr, d: float(np.min(gini(fr))),
    "Efficiency": lambda fr, d: float(np.max(fr.sum(1))),
}


def classify(env, stem):
    m = re.match(rf"^{env}_g(\d+)_(l0|temporal|spatial|spatiotemporal)_s(\d+)$", stem)
    if m:
        return (m.group(2), int(m.group(1)), int(m.group(3)))
    m = re.match(rf"^{env}_pcn_g(\d+)_s(\d+)$", stem)
    if m:
        return ("pcn", int(m.group(1)), int(m.group(2)))
    m = re.match(rf"^{env}_nsw_g(\d+)_s(\d+)$", stem)
    if m:
        return ("ncn", int(m.group(1)), int(m.group(2)))
    return None


def parse_log(path):
    text = path.read_text(errors="ignore")
    vecs = []
    for m in RET_RE.finditer(text):
        try:
            v = np.array([float(x) for x in m.group(1).split()])
        except ValueError:
            continue
        if v.size:
            vecs.append(v)
    if not vecs:
        return None
    width = max(v.size for v in vecs)
    vecs = [v for v in vecs if v.size == width]
    return np.array(vecs)


def collect(env):
    rows = []
    for d in LOG_DIRS:
        for f in sorted(Path(d).glob(f"{env}_*.log")):
            info = classify(env, f.stem)
            if info is None:
                continue
            method, g, seed = info
            vecs = parse_log(f)
            if vecs is None or len(vecs) < 5:
                continue
            front = cap_front(non_dominated(vecs[-80:]))
            row = dict(method=method, g=g, seed=seed)
            for name, fn in METRICS.items():
                row[name] = fn(front, g)
            rows.append(row)
    return rows


def agg(rows, method, g, metric):
    vals = [r[metric] for r in rows if r["method"] == method and r["g"] == g]
    return (np.mean(vals), np.std(vals), len(vals)) if vals else (np.nan, 0.0, 0)


def present_methods(rows):
    return [m for m in METHODS if any(r["method"] == m for r in rows)]


def tag(metric):
    return metric.lower().replace(" (per-dim)", "_pdim").replace(" ", "_")


def gcolors():
    return dict(zip(GROUPS, plt.cm.viridis(np.linspace(0.15, 0.85, len(GROUPS)))))


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(SAVEDIR / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.png")


def plot_bar_per_metric(rows, env, methods):
    gc = gcolors()
    x = np.arange(len(methods))
    width = 0.8 / len(GROUPS)
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(1.3 * len(methods) + 2, 4))
        for gi, g in enumerate(GROUPS):
            means = [agg(rows, m, g, metric)[0] for m in methods]
            stds = [agg(rows, m, g, metric)[1] for m in methods]
            off = (gi - (len(GROUPS) - 1) / 2) * width
            ax.bar(x + off, means, width, yerr=stds, capsize=2, color=gc[g],
                   label=f"G={g}", alpha=0.88, error_kw={"linewidth": 0.7})
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[m] for m in methods], fontsize=8, rotation=20, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(f"{env.capitalize()} - {metric}")
        ax.legend(title="Nr. groups", frameon=False, ncol=len(GROUPS), fontsize=8)
        fig.tight_layout()
        _save(fig, f"bar_{env}_{tag(metric)}")


def plot_bar_combined(rows, env, methods):
    names = list(METRICS)
    ncols, gc = 3, gcolors()
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows), squeeze=False)
    fig.suptitle(f"{env.capitalize()} - method comparison", fontsize=13)
    x = np.arange(len(methods))
    width = 0.8 / len(GROUPS)
    for idx, metric in enumerate(names):
        ax = axes[idx // ncols][idx % ncols]
        for gi, g in enumerate(GROUPS):
            means = [agg(rows, m, g, metric)[0] for m in methods]
            stds = [agg(rows, m, g, metric)[1] for m in methods]
            off = (gi - (len(GROUPS) - 1) / 2) * width
            ax.bar(x + off, means, width, yerr=stds, capsize=2, color=gc[g],
                   label=f"G={g}", alpha=0.88, error_kw={"linewidth": 0.7})
        ax.set_title(metric, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[m] for m in methods], fontsize=7, rotation=20, ha="right")
    for idx in range(len(names), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=gc[g], alpha=0.88) for g in GROUPS]
    fig.legend(handles, [f"G={g}" for g in GROUPS], title="Nr. groups",
               loc="lower center", ncol=len(GROUPS), frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    _save(fig, f"bar_{env}")


def plot_groups_line(rows, env, methods):
    names = list(METRICS)
    ncols = 3
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    fig.suptitle(f"{env.capitalize()} - metrics vs nr. groups", fontsize=13)
    for idx, metric in enumerate(names):
        ax = axes[idx // ncols][idx % ncols]
        for m in methods:
            gs = [g for g in GROUPS if agg(rows, m, g, metric)[2] > 0]
            mean = [agg(rows, m, g, metric)[0] for g in gs]
            std = [agg(rows, m, g, metric)[1] for g in gs]
            if not gs:
                continue
            ax.plot(gs, mean, marker="o", color=PALETTE[m], linewidth=1.6, markersize=5, label=LABELS[m])
            ax.fill_between(gs, np.array(mean) - np.array(std), np.array(mean) + np.array(std),
                            alpha=0.12, color=PALETTE[m])
        ax.set_title(metric)
        ax.set_xlabel("Nr. groups")
        ax.set_xticks(GROUPS)
    for idx in range(len(names), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
    handles = [Line2D([0], [0], color=PALETTE[m], marker="o", linewidth=1.6, label=LABELS[m]) for m in methods]
    fig.legend(handles=handles, loc="lower center", ncol=len(methods), frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout()
    _save(fig, f"groups_{env}")


def plot_fixed_g(rows, env, g):
    names = list(METRICS)
    present = [m for m in METHODS if agg(rows, m, g, names[0])[2] > 0]
    if len(present) < 2:
        return
    fig, axes = plt.subplots(1, len(names), figsize=(3.0 * len(names), 3.6), squeeze=False)
    fig.suptitle(f"{env.capitalize()} - methods at G={g}", fontsize=12)
    x = np.arange(len(present))
    for idx, metric in enumerate(names):
        ax = axes[0][idx]
        means = [agg(rows, m, g, metric)[0] for m in present]
        stds = [agg(rows, m, g, metric)[1] for m in present]
        ax.bar(x, means, 0.7, yerr=stds, capsize=3,
               color=[PALETTE[m] for m in present], alpha=0.9, error_kw={"linewidth": 0.8})
        ax.set_title(metric, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[m] for m in present], fontsize=7, rotation=30, ha="right")
    fig.tight_layout()
    _save(fig, f"{env}_methods_g{g}")


def print_numeric_table(rows, env):
    methods = present_methods(rows)
    for g in GROUPS:
        present = [m for m in methods if agg(rows, m, g, "HV (per-dim)")[2] > 0]
        if not present:
            continue
        print(f"\n[{env} G={g}]  (mean +/- std over seeds, recomputed from logs)")
        print("method".ljust(20) + "".join(f"{met[:12]:>16}" for met in METRICS))
        for m in present:
            line = LABELS[m].ljust(20)
            for met in METRICS:
                mu, sd, n = agg(rows, m, g, met)
                line += f"{mu:>9.4f}+-{sd:<5.3f}"
            print(line)


def main():
    for env in ENVS:
        rows = collect(env)
        print_numeric_table(rows, env)
        methods = present_methods(rows)
        print(f"\n=== {env}: {len(rows)} runs, methods={methods} ===")
        for m in methods:
            cov = {g: agg(rows, m, g, "HV (per-dim)")[2] for g in GROUPS}
            print(f"  {LABELS[m]:20s} {cov}")
        if not rows:
            continue
        plot_bar_per_metric(rows, env, methods)
        plot_bar_combined(rows, env, methods)
        plot_groups_line(rows, env, methods)
        for g in GROUPS:
            plot_fixed_g(rows, env, g)
    print(f"\nAll figures -> {SAVEDIR}/")


if __name__ == "__main__":
    main()
