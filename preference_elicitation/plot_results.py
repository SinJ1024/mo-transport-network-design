import argparse
import json
import os
from collections import defaultdict
from itertools import product

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

MODEL_COLORS = {
    "logistic_regression": "#2196F3",
    "bradley_terry":       "#FF9800",
    "feature_bt":          "#4CAF50",
}
MODEL_LABELS = {
    "logistic_regression": "Logistic Reg.",
    "bradley_terry":       "Bradley-Terry",
    "feature_bt":          "Feature BT",
}
STRATEGY_STYLES = {
    "random":      "--",
    "uncertainty": "-",
}
REGRET_THRESHOLD = 0.05



def _rl_model_from_policy_set(policy_set: str) -> str:
    """Return parent directory name for per-seed paths, stem for top-level files."""
    dirname = os.path.dirname(policy_set)
    if dirname:
        return os.path.basename(dirname)
    return os.path.splitext(os.path.basename(policy_set))[0]


def load_results(paths: list[str]) -> list[dict]:
    rows = []
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        for row in data:
            if "rl_model" not in row and "policy_set" in row:
                row["rl_model"] = _rl_model_from_policy_set(row["policy_set"])
        rows.extend(data)
    return rows


def group_by(rows, *keys):
    """Nest rows into a dict keyed by successive key values."""
    out = defaultdict(list)
    for r in rows:
        k = tuple(r[k] for k in keys)
        out[k].append(r)
    return dict(out)


def pivot_learning_curve(rows, x_key="query_budget", y_key="normalized_regret"):
    """Return (x_array, y_array) averaged over rows sharing the same x value."""
    by_x = defaultdict(list)
    for r in rows:
        by_x[r[x_key]].append(r[y_key])
    xs = sorted(by_x)
    ys = [np.mean(by_x[x]) for x in xs]
    return np.asarray(xs), np.asarray(ys)


def min_queries_at_threshold(rows, threshold=REGRET_THRESHOLD):
    """Return smallest query_budget where mean normalised regret <= threshold."""
    by_x = defaultdict(list)
    for r in rows:
        by_x[r["query_budget"]].append(r["normalized_regret"])
    for x in sorted(by_x):
        if np.mean(by_x[x]) <= threshold:
            return x
    return None



def plot_learning_curves(rows, out_dir):
    user_names = sorted({r["user_name"] for r in rows})
    rl_models  = sorted({r["rl_model"]  for r in rows})
    pref_models = sorted({r["preference_model"] for r in rows})
    strategies  = sorted({r["strategy"]         for r in rows})

    n_users = len(user_names)
    n_rl    = len(rl_models)
    fig, axes = plt.subplots(
        n_users, n_rl,
        figsize=(5 * n_rl, 3.5 * n_users),
        sharey=True, sharex=False,
        squeeze=False,
    )

    for row_i, user in enumerate(user_names):
        for col_j, rl in enumerate(rl_models):
            ax = axes[row_i][col_j]
            subset = [r for r in rows if r["user_name"] == user and r["rl_model"] == rl]

            for pm, st in product(pref_models, strategies):
                seg = [r for r in subset if r["preference_model"] == pm and r["strategy"] == st]
                if not seg:
                    continue
                xs, ys = pivot_learning_curve(seg)
                ax.plot(
                    xs, ys,
                    color=MODEL_COLORS.get(pm, "grey"),
                    linestyle=STRATEGY_STYLES.get(st, "-"),
                    linewidth=1.6,
                    label=f"{MODEL_LABELS.get(pm, pm)} ({st})",
                )

            ax.axhline(REGRET_THRESHOLD, color="red", linewidth=0.8, linestyle=":", alpha=0.7)
            ax.set_title(f"{user}\n[{rl}]", fontsize=8)
            ax.set_xlabel("Query budget", fontsize=7)
            ax.set_ylabel("Norm. regret", fontsize=7)
            ax.tick_params(labelsize=7)
            ax.set_ylim(-0.02, 1.05)

    # Shared legend
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Learning curves: normalised regret vs query budget", fontsize=11, y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, "learning_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")



def plot_efficiency(rows, out_dir):
    user_names  = sorted({r["user_name"]         for r in rows})
    pref_models = sorted({r["preference_model"]  for r in rows})
    rl_models   = sorted({r["rl_model"]          for r in rows})
    strategies  = sorted({r["strategy"]          for r in rows})

    n_rl = len(rl_models)
    fig, axes = plt.subplots(1, n_rl, figsize=(6 * n_rl, 4.5), sharey=False, squeeze=False)

    bar_width = 0.8 / (len(pref_models) * len(strategies))

    for col_j, rl in enumerate(rl_models):
        ax = axes[0][col_j]
        x_positions = np.arange(len(user_names))

        offset = 0
        for pm in pref_models:
            for st in strategies:
                heights = []
                for user in user_names:
                    seg = [r for r in rows
                           if r["user_name"] == user
                           and r["rl_model"] == rl
                           and r["preference_model"] == pm
                           and r["strategy"] == st]
                    mq = min_queries_at_threshold(seg) if seg else None
                    heights.append(mq if mq is not None else np.nan)

                positions = x_positions + offset * bar_width
                color = MODEL_COLORS.get(pm, "grey")
                hatch = "" if st == "uncertainty" else "///"
                ax.bar(
                    positions, heights,
                    width=bar_width * 0.9,
                    color=color, alpha=0.85, hatch=hatch,
                    label=f"{MODEL_LABELS.get(pm, pm)} ({st})",
                    edgecolor="white", linewidth=0.4,
                )
                offset += 1

        ax.set_xticks(x_positions + 0.4 - bar_width / 2)
        ax.set_xticklabels(user_names, rotation=25, ha="right", fontsize=7)
        ax.set_ylabel(f"Min queries (regret ≤ {REGRET_THRESHOLD})", fontsize=8)
        ax.set_title(f"Efficiency — {rl}", fontsize=9)
        ax.tick_params(labelsize=7)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Query efficiency per fairness definition", fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, "efficiency.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_ranking_quality(rows, out_dir):
    """
    At the maximum query budget for each combination, show:
    - Kendall-τ (higher = better)
    - Mean rank displacement (closer to 0 = better, positive = under-rating)
    """
    user_names  = sorted({r["user_name"]        for r in rows})
    pref_models = sorted({r["preference_model"] for r in rows})
    rl_models   = sorted({r["rl_model"]         for r in rows})

    n_rl = len(rl_models)
    fig, axes = plt.subplots(2, n_rl, figsize=(6 * n_rl, 7), squeeze=False)

    bar_width = 0.8 / len(pref_models)

    for col_j, rl in enumerate(rl_models):
        ax_tau  = axes[0][col_j]
        ax_bias = axes[1][col_j]
        x_positions = np.arange(len(user_names))

        for pm_i, pm in enumerate(pref_models):
            tau_vals  = []
            bias_vals = []
            for user in user_names:
                seg = [r for r in rows
                       if r["user_name"] == user
                       and r["rl_model"] == rl
                       and r["preference_model"] == pm]
                if not seg:
                    tau_vals.append(np.nan)
                    bias_vals.append(np.nan)
                    continue
                # Use results at the maximum query budget
                max_q = max(r["query_budget"] for r in seg)
                at_max = [r for r in seg if r["query_budget"] == max_q]
                tau_vals.append(np.nanmean([r["kendall_tau"] for r in at_max]))
                bias_vals.append(np.nanmean([r["mean_rank_displacement"] for r in at_max]))

            pos = x_positions + pm_i * bar_width
            color = MODEL_COLORS.get(pm, "grey")
            ax_tau.bar(pos, tau_vals, width=bar_width * 0.9, color=color, alpha=0.85,
                       label=MODEL_LABELS.get(pm, pm), edgecolor="white", linewidth=0.4)
            ax_bias.bar(pos, bias_vals, width=bar_width * 0.9, color=color, alpha=0.85,
                        label=MODEL_LABELS.get(pm, pm), edgecolor="white", linewidth=0.4)

        for ax, ylabel, title_tag in [
            (ax_tau,  "Kendall-τ (↑ better)",       "Ranking correlation"),
            (ax_bias, "Mean rank displacement (→0)", "Systematic bias"),
        ]:
            ax.set_xticks(x_positions + bar_width * (len(pref_models) - 1) / 2)
            ax.set_xticklabels(user_names, rotation=25, ha="right", fontsize=7)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.set_title(f"{title_tag} — {rl}", fontsize=9)
            ax.tick_params(labelsize=7)
            ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.5)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Ranking quality at maximum query budget", fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, "ranking_quality.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_heatmap(rows, out_dir):
    """
    Heatmap: rows = preference models, columns = users (fairness definitions).
    Colour = mean normalised regret at max budget.  Separate panel per rl_model.
    """
    pref_models = sorted({r["preference_model"] for r in rows})
    user_names  = sorted({r["user_name"]        for r in rows})
    rl_models   = sorted({r["rl_model"]         for r in rows})

    metrics = [
        ("normalized_regret",    "Norm. Regret\n(↓ better)"),
        ("kendall_tau",          "Kendall-τ\n(↑ better)"),
        ("top5_ranking_mismatch","Top-5 Mismatch\n(↓ better)"),
        ("mean_rank_displacement","Rank Bias\n(→0 better)"),
    ]

    n_rl = len(rl_models)
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_rl * n_metrics, 1,
                             figsize=(max(8, len(user_names) * 1.2), 3.5 * n_rl * n_metrics),
                             squeeze=False)

    ax_idx = 0
    for rl in rl_models:
        for metric_key, metric_label in metrics:
            ax = axes[ax_idx][0]
            matrix = np.full((len(pref_models), len(user_names)), np.nan)
            for pi, pm in enumerate(pref_models):
                for ui, user in enumerate(user_names):
                    seg = [r for r in rows
                           if r["rl_model"] == rl
                           and r["preference_model"] == pm
                           and r["user_name"] == user]
                    if not seg:
                        continue
                    max_q = max(r["query_budget"] for r in seg)
                    at_max = [r for r in seg if r["query_budget"] == max_q]
                    matrix[pi, ui] = np.nanmean([r[metric_key] for r in at_max])

            invert = metric_key in ("normalized_regret", "top5_ranking_mismatch")
            cmap = "RdYlGn" if not invert else "RdYlGn_r"
            vmin, vmax = np.nanmin(matrix), np.nanmax(matrix)
            if metric_key == "kendall_tau":
                vmin, vmax = -1, 1

            im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_xticks(range(len(user_names)))
            ax.set_xticklabels(user_names, rotation=30, ha="right", fontsize=7)
            ax.set_yticks(range(len(pref_models)))
            ax.set_yticklabels([MODEL_LABELS.get(p, p) for p in pref_models], fontsize=8)
            ax.set_title(f"{metric_label}  [{rl}]", fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

            for pi in range(len(pref_models)):
                for ui in range(len(user_names)):
                    val = matrix[pi, ui]
                    if not np.isnan(val):
                        ax.text(ui, pi, f"{val:.2f}", ha="center", va="center",
                                fontsize=6.5, color="black")
            ax_idx += 1

    fig.suptitle("Performance heatmap: preference model × fairness definition", fontsize=12)
    fig.tight_layout()
    path = os.path.join(out_dir, "heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_top5_mismatch(rows, out_dir):
    user_names  = sorted({r["user_name"]        for r in rows})
    pref_models = sorted({r["preference_model"] for r in rows})
    rl_models   = sorted({r["rl_model"]         for r in rows})
    strategies  = sorted({r["strategy"]         for r in rows})

    n_users = len(user_names)
    n_rl    = len(rl_models)
    fig, axes = plt.subplots(n_users, n_rl,
                             figsize=(5 * n_rl, 3.5 * n_users),
                             sharey=True, squeeze=False)

    for row_i, user in enumerate(user_names):
        for col_j, rl in enumerate(rl_models):
            ax = axes[row_i][col_j]
            subset = [r for r in rows if r["user_name"] == user and r["rl_model"] == rl]

            for pm, st in product(pref_models, strategies):
                seg = [r for r in subset if r["preference_model"] == pm and r["strategy"] == st]
                if not seg:
                    continue
                xs, ys = pivot_learning_curve(seg, y_key="top5_ranking_mismatch")
                ax.plot(xs, ys,
                        color=MODEL_COLORS.get(pm, "grey"),
                        linestyle=STRATEGY_STYLES.get(st, "-"),
                        linewidth=1.6,
                        label=f"{MODEL_LABELS.get(pm, pm)} ({st})")

            ax.set_title(f"{user}\n[{rl}]", fontsize=8)
            ax.set_xlabel("Query budget", fontsize=7)
            ax.set_ylabel("Top-5 mismatch", fontsize=7)
            ax.tick_params(labelsize=7)
            ax.set_ylim(-0.1, 5.5)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Top-5 ranking mismatch vs query budget", fontsize=11, y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, "top5_mismatch.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")



def main():
    global REGRET_THRESHOLD
    parser = argparse.ArgumentParser(description="Plot preference elicitation evaluation results.")
    parser.add_argument(
        "--results", nargs="+", required=True,
        help="One or more result JSON files (e.g. results/wandb_evaluation_lambda_lcn.json).",
    )
    parser.add_argument(
        "--output", default="figures/",
        help="Directory to write figures into.",
    )
    parser.add_argument(
        "--regret-threshold", type=float, default=REGRET_THRESHOLD,
        help="Regret threshold for efficiency metric.",
    )
    parser.add_argument(
        "--users", nargs="+", default=None,
        help="Only plot these user types (e.g. --users efficiency_focused fairness_focused).",
    )
    args = parser.parse_args()

    REGRET_THRESHOLD = args.regret_threshold

    os.makedirs(args.output, exist_ok=True)
    rows = load_results(args.results)
    if args.users:
        rows = [r for r in rows if r.get("user_name") in args.users]
    print(f"Loaded {len(rows)} result rows from {len(args.results)} file(s).")

    print("Plotting learning curves ...")
    plot_learning_curves(rows, args.output)

    print("Plotting efficiency ...")
    plot_efficiency(rows, args.output)

    print("Plotting ranking quality ...")
    plot_ranking_quality(rows, args.output)

    print("Plotting heatmap ...")
    plot_heatmap(rows, args.output)

    print("Plotting top-5 mismatch ...")
    plot_top5_mismatch(rows, args.output)

    print(f"\nAll figures saved to {args.output}")


if __name__ == "__main__":
    main()
