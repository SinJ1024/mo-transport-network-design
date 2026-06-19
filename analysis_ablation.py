"""
Ablation analysis: GCN lambda-scheduler experiments.
Generates training curves, bar charts, and LaTeX three-line tables.

Usage:
    python analysis_ablation.py              # all envs, all groups
    python analysis_ablation.py --env xian   # xian only
    python analysis_ablation.py --no-curves  # skip training curves (fast)
"""

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

import wandb

# ── Constants ─────────────────────────────────────────────────────────────────

PROJECT    = "jingyuan-sun03-tu-delft/cl_ablation"
HIDDEN_DIM = 128
GROUPS     = [3, 5, 7, 10]
CONFIGS    = ["l0", "temporal", "spatial", "spatiotemporal", "pcn", "ncn"]
ENVS       = ["xian", "amsterdam"]

CONFIG_LABELS = {
    "l0":             "GCN-Lorenz",
    "temporal":       "GCN-Temporal",
    "spatial":        "GCN-Spatial",
    "spatiotemporal": "GCN-Spatiotemporal",
    "pcn":            "PCN",
    "ncn":            "NCN (Nash)",
}

PALETTE = {
    "l0":             "#4C72B0",
    "temporal":       "#DD8452",
    "spatial":        "#55A868",
    "spatiotemporal": "#C44E52",
    "pcn":            "#8172B3",
    "ncn":            "#937860",
}

# Final-step metrics for table and bar charts
# key → (display name, higher-is-better)
METRICS = {
    "eval/hypervolume_pdim":       ("HV (per-dim)",  True),
    "eval/eum":                    ("EUM",            True),
    "eval/sen_welfare_max":        ("Sen Welfare",    True),
    "eval/demand_coverage_max":    ("Coverage Rate",  True),
    "eval/served_floor_max":       ("Served Floor",   True),
    "eval/gini_min":               ("Gini",           False),  # min = best front point (lower is better)
    "eval/efficiency_max":         ("Efficiency",     True),
    "eval/spatial_sw_high_median": ("Spatial SW High Median", True),  # no _max key logged; left as median
}

# Metrics shown on training curves
CURVE_METRICS = {
    "eval/hypervolume_pdim":       "HV (per-dim)",
    "eval/eum":                    "EUM",
    "eval/sen_welfare_max":        "Sen Welfare",
    "eval/demand_coverage_max":    "Coverage Rate",
    "eval/spatial_sw_high_median": "Spatial SW High Median",
}
CURVE_SMOOTH = 5   # rolling window (on ~30 eval points; 5 ≈ light smoothing)

SAVEDIR = Path("figures/ablation")
SAVEDIR.mkdir(parents=True, exist_ok=True)

# ── Data loading ──────────────────────────────────────────────────────────────

def classify_run(cfg: dict):
    """Map a run's wandb config to (env, groups, method); method in CONFIGS or None.

    Uses config fields rather than the run name, so it also handles the PCN and NCN
    baselines (which are not named GCN-<env>-g<G>-<config>). PCN is detected by name,
    NCN by criterion=='nash', and the four GCN schemes by spatial_alpha / schedule.
    """
    env = cfg.get("env")
    groups = cfg.get("nr_groups")
    if env is None or groups is None:
        return None, None, None
    name = str(cfg.get("experiment_name") or cfg.get("algo") or "")
    if name.upper().startswith("PCN"):
        return env, int(groups), "pcn"
    if cfg.get("criterion") == "nash":
        return env, int(groups), "ncn"
    if cfg.get("criterion", "lorenz") == "lorenz":
        spatial = float(cfg.get("spatial_alpha") or 0) > 0
        sched = (cfg.get("lambda_schedule") or cfg.get("scheduling_method") or "constant") != "constant"
        return env, int(groups), {
            (False, False): "l0", (False, True): "temporal",
            (True, False): "spatial", (True, True): "spatiotemporal",
        }[(spatial, sched)]
    return None, None, None


def fetch_summary(envs_filter=None) -> pd.DataFrame:
    api = wandb.Api()
    # NOTE: only runs with this hidden_dim and state are pulled. If the PCN/NCN
    # baselines used a different hidden_dim or live in another project, adjust
    # PROJECT / drop the hidden_dim filter so they are included.
    runs = api.runs(PROJECT, filters={"config.hidden_dim": HIDDEN_DIM, "state": "finished"})
    rows = []
    for r in runs:
        env, groups, config = classify_run(r.config)
        if env is None:
            continue
        if envs_filter and env not in envs_filter:
            continue
        row = dict(env=env, groups=groups, config=config,
                   seed=r.config.get("seed"), run_id=r.id)
        for key in METRICS:
            row[key] = r.summary.get(key, np.nan)
        rows.append(row)

    df = pd.DataFrame(rows)
    # keep one run per (env, groups, config, seed) — latest run_id wins
    df = (df.sort_values("run_id")
            .drop_duplicates(subset=["env", "groups", "config", "seed"], keep="last")
            .reset_index(drop=True))
    return df


def fetch_histories(df_summary: pd.DataFrame) -> pd.DataFrame:
    api = wandb.Api()
    keys = ["global_step"] + list(CURVE_METRICS.keys())
    frames = []
    total = len(df_summary)
    for i, (_, row) in enumerate(df_summary.iterrows()):
        if (i + 1) % 20 == 0:
            print(f"  history {i+1}/{total}")
        try:
            r = api.run(f"{PROJECT}/{row['run_id']}")
            h = r.history(keys=keys)
            h = h.dropna(subset=["global_step"])
            h["env"]    = row["env"]
            h["groups"] = row["groups"]
            h["config"] = row["config"]
            h["seed"]   = row["seed"]
            frames.append(h)
        except Exception as e:
            print(f"  skip {row['run_id']}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── Helper ────────────────────────────────────────────────────────────────────

def _bold_best(col_vals: list, higher_better: bool) -> list:
    """Return list of strings with LaTeX \\textbf{} on the best value."""
    nums = []
    for v in col_vals:
        try:
            nums.append(float(v.split("$")[0].strip()))
        except Exception:
            nums.append(np.nan)
    best = np.nanmax(nums) if higher_better else np.nanmin(nums)
    out = []
    for v, n in zip(col_vals, nums):
        out.append(f"\\textbf{{{v}}}" if np.isclose(n, best) else v)
    return out


def _has_data(df, env, metric):
    """True if metric has any non-zero, non-nan values for this env."""
    vals = df[df["env"] == env][metric].dropna()
    return len(vals) > 0 and vals.abs().sum() > 0


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_training_curves(df_hist: pd.DataFrame, env: str):
    """4-panel figure (one panel per group), sublines = configs."""
    df = df_hist[df_hist["env"] == env]
    if df.empty:
        print(f"  No history data for {env}, skipping curves.")
        return

    for metric, label in CURVE_METRICS.items():
        valid = df[metric].dropna()
        if valid.empty or valid.abs().sum() == 0:
            continue

        fig, axes = plt.subplots(1, len(GROUPS), figsize=(4.5 * len(GROUPS), 3.5),
                                 sharey=False)
        fig.suptitle(f"{env.capitalize()} — {label}", fontsize=13, y=1.01)

        for ax, g in zip(axes, GROUPS):
            sub = df[(df["env"] == env) & (df["groups"] == g)]
            any_plotted = False
            for cfg in CONFIGS:
                c = sub[sub["config"] == cfg].dropna(subset=[metric]).copy()
                if c.empty:
                    continue
                # Bin steps to 1000-step intervals so seeds with slightly
                # different eval schedules are grouped together correctly.
                c["step_bin"] = (c["global_step"] // 1000) * 1000
                grp  = c.groupby("step_bin")[metric]
                mean = grp.mean()
                std  = grp.std().fillna(0)
                w = CURVE_SMOOTH
                mean_s = mean.rolling(w, center=True, min_periods=1).mean()
                std_s  = std.rolling(w, center=True, min_periods=1).mean()
                ax.plot(mean_s.index, mean_s.values,
                        label=CONFIG_LABELS[cfg], color=PALETTE[cfg], linewidth=1.6)
                ax.fill_between(mean_s.index,
                                (mean_s - std_s).values, (mean_s + std_s).values,
                                alpha=0.15, color=PALETTE[cfg])
                any_plotted = True

            ax.set_title(f"G = {g}", fontsize=11)
            ax.set_xlabel("Step")
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k"))
            if ax is axes[0]:
                ax.set_ylabel(label)
            if not any_plotted:
                ax.set_visible(False)

        handles = [Line2D([0], [0], color=PALETTE[c], linewidth=1.6,
                          label=CONFIG_LABELS[c]) for c in CONFIGS]
        fig.legend(handles=handles, loc="lower center", ncol=4,
                   bbox_to_anchor=(0.5, -0.08), frameon=False)
        fig.tight_layout()

        tag = metric.replace("eval/", "").replace("/", "_")
        for ext in ("pdf", "png"):
            fig.savefig(SAVEDIR / f"curve_{env}_{tag}.{ext}", bbox_inches="tight")
        print(f"  Saved curve_{env}_{tag}.pdf")
        plt.close(fig)


def plot_bar_per_metric(df: pd.DataFrame, env: str):
    """One figure per metric: x=config, one bar group per nr_groups."""
    sub = df[df["env"] == env]
    if sub.empty:
        return

    for metric, (label, _) in METRICS.items():
        if not _has_data(df, env, metric):
            continue

        # collect mean ± std per (groups, config)
        records = []
        for g in GROUPS:
            for cfg in CONFIGS:
                vals = sub[(sub["groups"] == g) & (sub["config"] == cfg)][metric].dropna()
                if len(vals) == 0:
                    continue
                records.append(dict(groups=g, config=cfg,
                                    mean=vals.mean(), std=vals.std(ddof=0)))
        if not records:
            continue
        tmp = pd.DataFrame(records)

        fig, ax = plt.subplots(figsize=(7, 3.8))
        # only methods that have data for this metric (drops PCN/NCN on cell-level metrics)
        present_cfgs = [c for c in CONFIGS if (tmp["config"] == c).any()]
        x = np.arange(len(present_cfgs))
        width = 0.18
        present_groups = sorted(tmp["groups"].unique())
        offsets = np.linspace(-(len(present_groups) - 1) / 2,
                               (len(present_groups) - 1) / 2,
                               len(present_groups)) * width

        for off, g in zip(offsets, present_groups):
            row = tmp[tmp["groups"] == g]
            means = [row[row["config"] == c]["mean"].values[0]
                     if len(row[row["config"] == c]) else np.nan for c in present_cfgs]
            stds  = [row[row["config"] == c]["std"].values[0]
                     if len(row[row["config"] == c]) else 0  for c in present_cfgs]
            bars = ax.bar(x + off, means, width, yerr=stds,
                          capsize=3, label=f"G={g}", alpha=0.85,
                          error_kw={"linewidth": 0.8})

        ax.set_xticks(x)
        ax.set_xticklabels([CONFIG_LABELS[c] for c in present_cfgs])
        ax.set_ylabel(label)
        ax.set_title(f"{env.capitalize()} — {label}")
        ax.legend(title="Nr. groups", frameon=False, ncol=len(present_groups))
        fig.tight_layout()

        tag = metric.replace("eval/", "").replace("/", "_")
        for ext in ("pdf", "png"):
            fig.savefig(SAVEDIR / f"bar_{env}_{tag}.{ext}", bbox_inches="tight")
        print(f"  Saved bar_{env}_{tag}.pdf")
        plt.close(fig)


def plot_bar_combined(df: pd.DataFrame, env: str):
    """All metrics in one figure: subplots grid, x=config, bars per group."""
    sub = df[df["env"] == env]
    if sub.empty:
        return

    active = [(k, lbl) for k, (lbl, _) in METRICS.items() if _has_data(df, env, k)]
    if not active:
        return

    ncols = 4
    nrows = int(np.ceil(len(active) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    fig.suptitle(f"{env.capitalize()} — All Metrics", fontsize=13)

    n_configs = len(CONFIGS)
    x = np.arange(n_configs)
    width = 0.18
    present_groups = sorted(sub["groups"].unique())
    offsets = np.linspace(-(len(present_groups) - 1) / 2,
                           (len(present_groups) - 1) / 2,
                           len(present_groups)) * width
    group_colors = {g: c for g, c in zip(present_groups,
                    plt.rcParams["axes.prop_cycle"].by_key()["color"])}

    for idx, (metric, label) in enumerate(active):
        ax = axes[idx // ncols][idx % ncols]
        # only methods that actually have data for THIS metric: drops e.g. PCN/NCN
        # from the cell-level panels (served floor, coverage, spatial SW) automatically
        present_cfgs = [c for c in CONFIGS if sub[sub["config"] == c][metric].notna().any()]
        xx = np.arange(len(present_cfgs))
        records = []
        for g in present_groups:
            for cfg in present_cfgs:
                vals = sub[(sub["groups"] == g) & (sub["config"] == cfg)][metric].dropna()
                if len(vals):
                    records.append(dict(groups=g, config=cfg,
                                        mean=vals.mean(), std=vals.std(ddof=0)))
        tmp = pd.DataFrame(records)
        for off, g in zip(offsets, present_groups):
            row = tmp[tmp["groups"] == g] if not tmp.empty else tmp
            means = [row[row["config"] == c]["mean"].values[0]
                     if (not row.empty and len(row[row["config"] == c])) else np.nan for c in present_cfgs]
            stds  = [row[row["config"] == c]["std"].values[0]
                     if (not row.empty and len(row[row["config"] == c])) else 0  for c in present_cfgs]
            ax.bar(xx + off, means, width, yerr=stds, capsize=3,
                   color=group_colors[g], label=f"G={g}", alpha=0.85,
                   error_kw={"linewidth": 0.8})
        ax.set_title(label, fontsize=10)
        ax.set_xticks(xx)
        ax.set_xticklabels([CONFIG_LABELS[c] for c in present_cfgs], fontsize=8, rotation=15, ha="right")

    # hide unused axes
    for idx in range(len(active), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, alpha=0.85, color=group_colors[g])
               for g in present_groups]
    fig.legend(handles, [f"G={g}" for g in present_groups],
               title="Nr. groups", loc="lower center",
               ncol=len(present_groups), frameon=False,
               bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(SAVEDIR / f"bar_{env}.{ext}", bbox_inches="tight")
    print(f"  Saved bar_{env}.pdf")
    plt.close(fig)


def plot_configs_across_groups(df: pd.DataFrame, env: str):
    """Line plot: x=nr_groups, one line per config, per metric."""
    sub = df[df["env"] == env]
    if sub.empty:
        return

    active_metrics = [(k, v) for k, v in METRICS.items() if _has_data(df, env, k)]
    if not active_metrics:
        return

    ncols = 3
    nrows = int(np.ceil(len(active_metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    fig.suptitle(f"{env.capitalize()} — metrics vs. nr. groups", fontsize=13)

    for idx, (metric, (label, _)) in enumerate(active_metrics):
        ax = axes[idx // ncols][idx % ncols]
        for cfg in CONFIGS:
            c = sub[sub["config"] == cfg]
            grp = c.groupby("groups")[metric]
            mean = grp.mean()
            std  = grp.std(ddof=0).fillna(0)
            if mean.empty:
                continue
            ax.plot(mean.index, mean.values,
                    label=CONFIG_LABELS[cfg], color=PALETTE[cfg],
                    marker="o", linewidth=1.6, markersize=5)
            ax.fill_between(mean.index,
                            (mean - std).values, (mean + std).values,
                            alpha=0.12, color=PALETTE[cfg])
        ax.set_title(label)
        ax.set_xlabel("Nr. groups")
        ax.set_xticks(GROUPS)

    # hide unused axes
    for idx in range(len(active_metrics), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    handles = [Line2D([0], [0], color=PALETTE[c], marker="o", linewidth=1.6,
                      label=CONFIG_LABELS[c]) for c in CONFIGS]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.04), frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(SAVEDIR / f"groups_{env}.{ext}", bbox_inches="tight")
    print(f"  Saved groups_{env}.pdf")
    plt.close(fig)


def plot_cross_env(df: pd.DataFrame):
    """Compare xian vs amsterdam for spatiotemporal config across groups."""
    active_metrics = [(k, v) for k, v in METRICS.items()
                      if _has_data(df, "xian", k) and _has_data(df, "amsterdam", k)]
    if not active_metrics:
        return

    ncols = 3
    nrows = int(np.ceil(len(active_metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    fig.suptitle("Spatiotemporal — Xian vs Amsterdam", fontsize=13)

    env_colors = {"xian": "#4C72B0", "amsterdam": "#DD8452"}

    for idx, (metric, (label, _)) in enumerate(active_metrics):
        ax = axes[idx // ncols][idx % ncols]
        for env in ENVS:
            c = df[(df["env"] == env) & (df["config"] == "spatiotemporal")]
            grp = c.groupby("groups")[metric]
            mean = grp.mean()
            std  = grp.std(ddof=0).fillna(0)
            if mean.empty:
                continue
            ax.errorbar(mean.index, mean.values, yerr=std.values,
                        label=env.capitalize(), color=env_colors[env],
                        marker="o", linewidth=1.6, capsize=4, markersize=5)
        ax.set_title(label)
        ax.set_xlabel("Nr. groups")
        ax.set_xticks(GROUPS)

    for idx in range(len(active_metrics), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    handles = [Line2D([0], [0], color=c, marker="o", linewidth=1.6,
                      label=e.capitalize()) for e, c in env_colors.items()]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.04), frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(SAVEDIR / f"cross_env.{ext}", bbox_inches="tight")
    print(f"  Saved cross_env.pdf")
    plt.close(fig)


# ── LaTeX table ───────────────────────────────────────────────────────────────

def _fmt(mean, std):
    if np.isnan(mean):
        return "--"
    return f"{mean:.4f} $\\pm$ {std:.4f}"


def make_latex_table(df: pd.DataFrame, env: str, groups: int) -> str:
    sub = df[(df["env"] == env) & (df["groups"] == groups)]
    if sub.empty:
        return ""

    # only include metrics with actual data
    active = [(k, lbl, hib) for k, (lbl, hib) in METRICS.items()
              if _has_data(sub, env, k)]
    if not active:
        return ""

    col_fmt  = "l" + "c" * len(active)
    col_head = " & ".join(["Method"] + [lbl for _, lbl, _ in active])

    lines = [
        f"% {env.capitalize()}, G={groups}",
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{Ablation on {env.capitalize()}, $G={groups}$ income groups.}}",
        f"\\label{{tab:ablation_{env}_g{groups}}}",
        f"\\begin{{tabular}}{{{col_fmt}}}",
        "\\toprule",
        col_head + " \\\\",
        "\\midrule",
    ]

    # collect raw strings first so we can bold the best
    cell_strings = {(k, cfg): "--" for k, _, _ in active for cfg in CONFIGS}
    for cfg in CONFIGS:
        c = sub[sub["config"] == cfg]
        for key, _, _ in active:
            vals = c[key].dropna()
            if len(vals) > 0:
                cell_strings[(key, cfg)] = _fmt(vals.mean(), vals.std(ddof=0))

    # bold best per column
    for key, _, hib in active:
        col = [cell_strings[(key, cfg)] for cfg in CONFIGS]
        bolded = _bold_best(col, hib)
        for cfg, val in zip(CONFIGS, bolded):
            cell_strings[(key, cfg)] = val

    for cfg in CONFIGS:
        cells = [CONFIG_LABELS[cfg]] + [cell_strings[(k, cfg)] for k, _, _ in active]
        lines.append(" & ".join(cells) + " \\\\")

    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def save_latex_tables(df: pd.DataFrame):
    blocks = []
    for env in ENVS:
        for g in GROUPS:
            t = make_latex_table(df, env, g)
            if t:
                blocks.append(t)
    fpath = SAVEDIR / "tables.tex"
    fpath.write_text("\n\n".join(blocks), encoding="utf-8")
    print(f"  Saved {fpath}")


def print_summary(df: pd.DataFrame):
    for env in ENVS:
        sub = df[df["env"] == env]
        if sub.empty:
            continue
        print(f"\n{'='*70}")
        print(f"  {env.upper()}  (mean across seeds)")
        print(f"{'='*70}")
        cols = {k: lbl for k, (lbl, _) in METRICS.items() if _has_data(df, env, k)}
        pivot = sub.groupby(["groups", "config"])[list(cols.keys())].mean()
        pivot.columns = list(cols.values())
        print(pivot.to_string())


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", nargs="+", default=ENVS)
    parser.add_argument("--no-curves", action="store_true",
                        help="Skip training curves (much faster)")
    args = parser.parse_args()

    print("Fetching summary from wandb …")
    df = fetch_summary(envs_filter=args.env)
    print(f"  {len(df)} runs loaded.\n")

    print("Generating tables …")
    save_latex_tables(df)
    print_summary(df)

    print("\nGenerating bar charts (per metric) …")
    for env in args.env:
        plot_bar_per_metric(df, env)

    print("\nGenerating combined bar charts …")
    for env in args.env:
        plot_bar_combined(df, env)

    print("\nGenerating metrics-vs-groups line plots …")
    for env in args.env:
        plot_configs_across_groups(df, env)

    if set(args.env) == set(ENVS):
        print("\nGenerating cross-env comparison …")
        plot_cross_env(df)

    if not args.no_curves:
        print("\nFetching training histories (takes a few minutes) …")
        df_hist = fetch_histories(df)
        if not df_hist.empty:
            for env in args.env:
                plot_training_curves(df_hist, env)
    else:
        print("\nTraining curves skipped (--no-curves).")

    print(f"\nAll figures → {SAVEDIR}/")


if __name__ == "__main__":
    main()
