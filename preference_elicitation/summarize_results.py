import argparse
import json
import os

import numpy as np
import pandas as pd

REGRET_THRESHOLD = 0.05

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "bradley_terry":       "Bradley-Terry",
    "feature_bt":          "Feature BT",
}



def _rl_model_from_policy_set(policy_set: str) -> str:
    """Return the model label from a policy_set path.

    For aggregate files (e.g. 'gcn.json') returns the stem ('gcn').
    For per-seed files (e.g. 'gcn/front_table_11.json') returns the parent
    directory name ('gcn') so all seeds are grouped under the same label.
    """
    dirname = os.path.dirname(policy_set)
    if dirname:
        return os.path.basename(dirname)
    return os.path.splitext(os.path.basename(policy_set))[0]


def load_results(paths: list[str]) -> pd.DataFrame:
    rows = []
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        for row in data:
            if "rl_model" not in row and "policy_set" in row:
                row["rl_model"] = _rl_model_from_policy_set(row["policy_set"])
            row["top1_hit"] = int(row["selected_policy_id"] == row["true_best_policy_id"])
        rows.extend(data)
    return pd.DataFrame(rows)


def at_max_budget(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows at the maximum query_budget for each group."""
    group_keys = ["preference_model", "rl_model", "user_name", "strategy"]
    max_q = df.groupby(group_keys)["query_budget"].transform("max")
    return df[df["query_budget"] == max_q]


def compute_min_queries(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    For each (preference_model, rl_model, user_name, strategy) group find the
    smallest query_budget where the mean normalised regret <= threshold.
    Returns a DataFrame with a `min_queries` column (NaN if never reached).
    """
    group_keys = ["preference_model", "rl_model", "user_name", "strategy"]
    records = []
    for keys, group in df.groupby(group_keys):
        by_q = group.groupby("query_budget")["normalized_regret"].mean()
        reached = by_q[by_q <= threshold]
        min_q = float(reached.index.min()) if len(reached) > 0 else np.nan
        records.append(dict(zip(group_keys, keys), min_queries=min_q))
    return pd.DataFrame(records)



def print_table(title: str, df: pd.DataFrame):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    print(df.to_string(float_format=lambda x: f"{x:.3f}"))
    print()



def main():
    parser = argparse.ArgumentParser(
        description="Generate report-quality summary statistics."
    )
    parser.add_argument(
        "--results", nargs="+", required=True,
        help="One or more result JSON files.",
    )
    parser.add_argument(
        "--output", default="stats/",
        help="Directory to write CSV outputs into.",
    )
    parser.add_argument(
        "--regret-threshold", type=float, default=REGRET_THRESHOLD,
        help="Normalised-regret threshold for query efficiency metric.",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    threshold = args.regret_threshold

    df = load_results(args.results)
    print(
        f"\nLoaded {len(df):,} rows"
        f" | preference models: {sorted(df['preference_model'].unique())}"
        f" | fairness defs: {sorted(df['rl_model'].unique())}"
    )

    max_df = at_max_budget(df)
    eff_df  = compute_min_queries(df, threshold)

    t1 = (
        max_df
        .groupby("preference_model")
        .agg(
            top1_accuracy        =("top1_hit",               "mean"),
            mean_regret          =("normalized_regret",       "mean"),
            mean_kendall_tau     =("kendall_tau",             "mean"),
            mean_rank_bias       =("mean_rank_displacement",  "mean"),
            mean_top5_mismatch   =("top5_ranking_mismatch",   "mean"),
            mean_policy_distance =("policy_distance",         "mean"),
            mean_selected_rank   =("selected_rank",           "mean"),
        )
        .round(3)
    )
    eff_model = (
        eff_df.groupby("preference_model")["min_queries"]
        .agg(mean_queries=("mean"), median_queries=("median"))
        .round(1)
    )
    t1 = t1.join(eff_model)
    t1.index = t1.index.map(lambda x: MODEL_LABELS.get(x, x))
    print_table(
        f"Q1+Q2 | Model performance (avg over all users & fairness defs, "
        f"threshold={threshold})",
        t1,
    )
    t1.to_csv(os.path.join(args.output, "model_summary.csv"))

    best_util = (
        max_df.groupby(["rl_model", "user_name"])["best_utility"]
        .mean()
        .reset_index()
        .groupby("rl_model")["best_utility"]
        .agg(mean_best_utility=("mean"), std_best_utility=("std"))
        .round(4)
    )
    elicit = (
        max_df.groupby("rl_model")
        .agg(
            mean_regret_at_max   =("normalized_regret",      "mean"),
            top1_accuracy        =("top1_hit",               "mean"),
            mean_kendall_tau     =("kendall_tau",             "mean"),
            mean_top5_mismatch   =("top5_ranking_mismatch",   "mean"),
            mean_policy_distance =("policy_distance",         "mean"),
            mean_selected_rank   =("selected_rank",           "mean"),
        )
        .round(3)
    )
    eff_fairness = (
        eff_df.groupby("rl_model")["min_queries"]
        .agg(mean_queries=("mean"), median_queries=("median"))
        .round(1)
    )
    t2 = best_util.join(elicit).join(eff_fairness)
    print_table("Q3+Q4 | Fairness definition comparison", t2)
    t2.to_csv(os.path.join(args.output, "fairness_summary.csv"))

    for metric, label, q in [
        ("normalized_regret",      "Regret at max budget (↓ better)",        "Q1"),
        ("top1_hit",               "Top-1 accuracy (↑ better)",              "Q1"),
        ("kendall_tau",            "Kendall-τ at max budget (↑ better)",     "Q3"),
        ("mean_rank_displacement", "Mean rank bias at max budget (→0)",      "Q3"),
        ("top5_ranking_mismatch",  "Top-5 mismatch at max budget (↓ better)","Q3"),
        ("policy_distance",        "Avg policy distance at max budget",      "Q4"),
        ("selected_rank",          "Avg selected rank at max budget (↓ better)", "Q4"),
    ]:
        t = (
            max_df
            .pivot_table(
                index="preference_model",
                columns="rl_model",
                values=metric,
                aggfunc="mean",
            )
            .round(3)
        )
        t.index = t.index.map(lambda x: MODEL_LABELS.get(x, x))
        print_table(f"{q} | {label}  [model × fairness definition]", t)
        t.to_csv(os.path.join(args.output, f"cross_{metric}.csv"))

    t4 = (
        max_df.groupby(["preference_model", "strategy"])
        .agg(
            mean_regret          =("normalized_regret",    "mean"),
            top1_accuracy        =("top1_hit",             "mean"),
            mean_top5_mismatch   =("top5_ranking_mismatch", "mean"),
            mean_policy_distance =("policy_distance",       "mean"),
            mean_selected_rank   =("selected_rank",         "mean"),
        )
        .round(3)
    )
    eff_strat = (
        eff_df.groupby(["preference_model", "strategy"])["min_queries"]
        .mean()
        .round(1)
        .rename("mean_queries_to_threshold")
    )
    t4 = t4.join(eff_strat)
    t4.index = t4.index.set_levels(
        [t4.index.levels[0].map(lambda x: MODEL_LABELS.get(x, x)), t4.index.levels[1]]
    )
    print_table("Q2 | Uncertainty vs Random query strategy", t4)
    t4.to_csv(os.path.join(args.output, "strategy_comparison.csv"))

    t5 = (
        max_df.groupby("user_name")
        .agg(
            mean_regret          =("normalized_regret",    "mean"),
            top1_accuracy        =("top1_hit",             "mean"),
            mean_best_utility    =("best_utility",          "mean"),
            mean_selected_utility=("selected_utility",      "mean"),
            mean_top5_mismatch   =("top5_ranking_mismatch", "mean"),
            mean_policy_distance =("policy_distance",       "mean"),
            mean_selected_rank   =("selected_rank",         "mean"),
        )
        .round(3)
    )
    print_table("Supplementary | Performance by user type (avg over all models & fairness defs)", t5)
    t5.to_csv(os.path.join(args.output, "user_summary.csv"))

    print(f"All CSV tables saved to:  {os.path.abspath(args.output)}\n")


if __name__ == "__main__":
    main()
