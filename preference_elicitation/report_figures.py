import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":  "sans-serif",
    "font.size":    9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

REGRET_THRESHOLD = 0.05

MODEL_ORDER = ["logistic_regression", "bradley_terry", "feature_bt"]
MODEL_LABELS = {
    "logistic_regression": "Logistic Reg.",
    "bradley_terry":       "Bradley-Terry",
    "feature_bt":          "Feature BT",
}
MODEL_COLORS = {
    "logistic_regression": "#2196F3",
    "bradley_terry":       "#FF9800",
    "feature_bt":          "#4CAF50",
}
FAIRNESS_LABELS = {
    "lambda_lcn": "λ-LCN",
    "pcn":        "PCN",
    "lcn":        "LCN",
}
STRATEGY_HATCHES = {"uncertainty": "", "random": "///"}
STRATEGY_ALPHA   = {"uncertainty": 0.85, "random": 0.55}


# ── Data loading ───────────────────────────────────────────────────────────────

def load(paths: list[str]) -> pd.DataFrame:
    rows = []
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        for row in data:
            if "rl_model" not in row and "policy_set" in row:
                row["rl_model"] = os.path.splitext(os.path.basename(row["policy_set"]))[0]
        rows.extend(data)
    df = pd.DataFrame(rows)
    df["top1_hit"] = (df["selected_policy_id"] == df["true_best_policy_id"]).astype(float)
    return df


def at_max_budget(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["preference_model", "rl_model", "user_name", "strategy"]
    max_q = df.groupby(keys)["query_budget"].transform("max")
    return df[df["query_budget"] == max_q]


def min_queries(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Min query_budget where mean regret <= threshold, per group."""
    keys = ["preference_model", "rl_model", "user_name", "strategy"]
    records = []
    for vals, grp in df.groupby(keys):
        by_q = grp.groupby("query_budget")["normalized_regret"].mean()
        reached = by_q[by_q <= threshold]
        q = float(reached.index.min()) if len(reached) else np.nan
        records.append(dict(zip(keys, vals), min_queries=q))
    return pd.DataFrame(records)


def save(fig, path: str):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Figure 1: Learning curves ─────────────────────────────────────────────────

def fig1_learning_curves(df: pd.DataFrame, out_dir: str, threshold: float):
    """
    One line per preference model, averaged over users, fairness definitions
    and strategies.  Shaded band = ±1 std across the three fairness definitions
    (shows whether convergence is consistent regardless of which definition is
    used).
    """
    # Step 1: mean per (pref_model, rl_model, query_budget)
    s1 = (
        df.groupby(["preference_model", "rl_model", "query_budget"])
        ["normalized_regret"].mean()
        .reset_index()
    )
    # Step 2: mean & std across rl_models for each (pref_model, query_budget)
    s2 = (
        s1.groupby(["preference_model", "query_budget"])["normalized_regret"]
        .agg(["mean", "std"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(6, 3.8))

    pref_models = [p for p in MODEL_ORDER if p in df["preference_model"].unique()]
    for pm in pref_models:
        grp = s2[s2["preference_model"] == pm].sort_values("query_budget")
        xs  = grp["query_budget"].values
        ys  = grp["mean"].values
        err = grp["std"].fillna(0).values
        c   = MODEL_COLORS[pm]
        ax.plot(xs, ys, color=c, linewidth=2.0, label=MODEL_LABELS[pm])
        ax.fill_between(xs, np.clip(ys - err, 0, 1), np.clip(ys + err, 0, 1),
                        color=c, alpha=0.12)

    ax.axhline(threshold, color="crimson", linewidth=1.0, linestyle=":",
               label=f"Threshold  ({threshold})")
    ax.set_xlabel("Query budget (number of pairwise comparisons)")
    ax.set_ylabel("Normalised regret  (↓ better)")
    ax.set_title("Does preference elicitation converge?", fontweight="bold", fontsize=10)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=8, framealpha=0.9)
    fig.tight_layout()
    save(fig, os.path.join(out_dir, "fig1_learning_curves.png"))


# ── Figure 2: Query efficiency ────────────────────────────────────────────────

def fig2_efficiency(df: pd.DataFrame, out_dir: str, threshold: float):
    """
    Mean queries to reach the regret threshold, grouped by preference model and
    split by query strategy (uncertainty vs random).  Missing bars (never
    reached) are shown as hatched bars at the plot ceiling with an 'N/R' label.
    """
    eff = min_queries(df, threshold)
    summary = (
        eff.groupby(["preference_model", "strategy"])["min_queries"]
        .mean()
        .reset_index()
    )

    pref_models = [p for p in MODEL_ORDER if p in df["preference_model"].unique()]
    strategies  = sorted(summary["strategy"].unique())
    x = np.arange(len(pref_models))
    width = 0.38

    # Ceiling for N/R bars
    finite = summary["min_queries"].dropna()
    ceiling = finite.max() * 1.15 if len(finite) else 50.0

    fig, ax = plt.subplots(figsize=(6, 3.8))

    for si, strat in enumerate(strategies):
        vals = []
        for pm in pref_models:
            row = summary[(summary["preference_model"] == pm) & (summary["strategy"] == strat)]
            vals.append(row["min_queries"].values[0] if len(row) else np.nan)

        offset = (si - 0.5) * width
        heights = [v if not np.isnan(v) else ceiling for v in vals]
        hatches = [STRATEGY_HATCHES[strat] if not np.isnan(v) else "xxx" for v in vals]

        for xi, (h, hatch) in enumerate(zip(heights, hatches)):
            color = "#4a90d9" if strat == "uncertainty" else "#aec6e8"
            ax.bar(x[xi] + offset, h, width=width * 0.92,
                   color=color, hatch=hatch, edgecolor="white",
                   linewidth=0.5, alpha=STRATEGY_ALPHA[strat],
                   label=strat.capitalize() if xi == 0 else "_nolegend_")
            if np.isnan(vals[xi]):
                ax.text(x[xi] + offset, ceiling * 0.5, "N/R",
                        ha="center", va="center", fontsize=7.5,
                        color="crimson", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(pm, pm) for pm in pref_models], fontsize=8)
    ax.set_ylabel(f"Mean queries to regret ≤ {threshold}  (↓ better)", fontsize=8)
    ax.set_title("Query efficiency per preference model", fontweight="bold", fontsize=10)
    ax.set_ylim(0, ceiling * 1.12)
    ax.legend(fontsize=8, framealpha=0.9)
    fig.tight_layout()
    save(fig, os.path.join(out_dir, "fig2_efficiency.png"))


# ── Figure 3: Fairness definition comparison ──────────────────────────────────

def fig3_fairness(df: pd.DataFrame, out_dir: str):
    """
    Left panel:  elicitability — normalised regret at max budget per fairness
                 definition (bars = preference models).  Lower = easier to
                 elicit good preferences.
    Right panel: true policy quality — mean best-policy utility per fairness
                 definition (averaged over users).  Higher = better policies
                 were produced by that training approach.
    """
    max_df = at_max_budget(df)

    fairness_defs = sorted(df["rl_model"].unique())
    pref_models   = [p for p in MODEL_ORDER if p in df["preference_model"].unique()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # ── Left: regret per fairness definition, bars = preference models ────────
    x     = np.arange(len(fairness_defs))
    width = 0.75 / len(pref_models)

    for pi, pm in enumerate(pref_models):
        vals = []
        for fd in fairness_defs:
            seg = max_df[(max_df["preference_model"] == pm) & (max_df["rl_model"] == fd)]
            vals.append(seg["normalized_regret"].mean() if len(seg) else np.nan)
        offset = (pi - (len(pref_models) - 1) / 2) * width
        ax1.bar(x + offset, vals, width=width * 0.92,
                color=MODEL_COLORS[pm], label=MODEL_LABELS[pm],
                edgecolor="white", linewidth=0.5, alpha=0.85)

    ax1.set_xticks(x)
    ax1.set_xticklabels([FAIRNESS_LABELS.get(fd, fd) for fd in fairness_defs])
    ax1.set_ylabel("Mean normalised regret at max budget  (↓ better)", fontsize=8)
    ax1.set_title("Elicitability per fairness definition", fontweight="bold", fontsize=10)
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=7.5, framealpha=0.9)

    # ── Right: best-policy utility per fairness definition ────────────────────
    # best_utility is constant for (policy_set, user) — average across users
    util = (
        max_df.groupby(["rl_model", "user_name"])["best_utility"]
        .mean()
        .reset_index()
        .groupby("rl_model")["best_utility"]
        .agg(["mean", "std"])
    )

    pal = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]
    means = [util.loc[fd, "mean"] if fd in util.index else np.nan for fd in fairness_defs]
    stds  = [util.loc[fd, "std"]  if fd in util.index else 0       for fd in fairness_defs]

    ax2.bar(x, means, yerr=stds, width=0.5,
            color=[pal[i % len(pal)] for i in range(len(fairness_defs))],
            edgecolor="white", linewidth=0.5, capsize=4, alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels([FAIRNESS_LABELS.get(fd, fd) for fd in fairness_defs])
    ax2.set_ylabel("Mean best-policy utility  (↑ better)", fontsize=8)
    ax2.set_title("True policy quality per fairness definition", fontweight="bold", fontsize=10)

    fig.suptitle(
        "Which fairness definition best supports preference elicitation?",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    save(fig, os.path.join(out_dir, "fig3_fairness.png"))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate three report figures.")
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--output",  default="figures/report/")
    parser.add_argument("--regret-threshold", type=float, default=REGRET_THRESHOLD)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    threshold = args.regret_threshold

    df = load(args.results)
    print(
        f"Loaded {len(df):,} rows | "
        f"models: {sorted(df['preference_model'].unique())} | "
        f"fairness: {sorted(df['rl_model'].unique())}"
    )

    print("\nFig 1 — learning curves ...")
    fig1_learning_curves(df, args.output, threshold)

    print("Fig 2 — query efficiency ...")
    fig2_efficiency(df, args.output, threshold)

    print("Fig 3 — fairness definition comparison ...")
    fig3_fairness(df, args.output)

    print(f"\nDone. Figures saved to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
