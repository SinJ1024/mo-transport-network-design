"""
Run all preference models on every policy set found under policy_sets/.

Scans:
  policy_sets/*.json          — aggregate sets (all seeds merged)
  policy_sets/*/*.json        — per-seed sets

For each JSON file the script runs Logistic Regression, Bradley-Terry and
Feature BT, then saves:
  results/<relative_path>.json   — per policy-set results
  results/all_results.json       — every row combined

Dependencies: only the three model runner files, users.py, query_strategies.py
"""

import argparse
import dataclasses
import glob
import json
import os

import numpy as np

from users import make_default_users
from linear_regression_runner import (
    load_policies,
    preprocess_policies,
    run_query_budget_search as _run_lr,
)
from bradley_terry_runner import run_budget_search as _run_bt
from feature_bt_runner import run_budget_search as _run_fbt


# ── Helpers ────────────────────────────────────────────────────────────────────

def _to_dicts(results, preference_model: str, policy_set_path: str) -> list:
    rows = []
    for r in results:
        row = dataclasses.asdict(r)
        row["preference_model"] = preference_model
        row["policy_set"] = policy_set_path
    return rows


def _find_policy_sets(root: str) -> list[str]:
    """Return per-seed JSON files first, then top-level aggregate files."""
    seeds = sorted(glob.glob(os.path.join(root, "*", "*.json")))
    top   = sorted(glob.glob(os.path.join(root, "*.json")))
    return seeds + top


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run all preference models on all local policy sets."
    )
    parser.add_argument("--policy-sets-dir", default="policy_sets",
                        help="Root directory containing policy set JSON files.")
    parser.add_argument("--output-dir", default="results",
                        help="Directory to write result JSON files into.")
    parser.add_argument(
        "--preference-models", nargs="+",
        default=["logistic_regression", "bradley_terry", "feature_bt"],
        choices=["logistic_regression", "bradley_terry", "feature_bt"],
    )
    parser.add_argument("--strategies", nargs="+", default=["random", "uncertainty"])
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--query-step", type=int, default=1)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--regret-threshold", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--disable-budget-search", action="store_true")
    parser.add_argument("--fixed-query-budget", type=int, default=None)
    parser.add_argument(
        "--filter", default=None,
        help="Only process policy sets whose relative path starts with this prefix (e.g. 'lcn').",
    )
    args = parser.parse_args()

    policy_set_files = _find_policy_sets(args.policy_sets_dir)
    if args.filter:
        policy_set_files = [
            f for f in policy_set_files
            if os.path.relpath(f, args.policy_sets_dir).startswith(args.filter)
        ]
    if not policy_set_files:
        print(f"No policy set JSON files found under {args.policy_sets_dir!r}. Exiting.")
        return

    print(f"Found {len(policy_set_files)} policy set(s).\n")
    os.makedirs(args.output_dir, exist_ok=True)

    all_rows = []

    for ps_path in policy_set_files:
        rel = os.path.relpath(ps_path, args.policy_sets_dir)
        print(f"\n{'='*60}")
        print(f"Policy set: {rel}")
        print(f"{'='*60}")

        raw = load_policies(ps_path)
        policies = preprocess_policies(raw)
        print(f"  Policies after preprocessing: {len(policies)}")

        if len(policies) < 2:
            print("  Skipping: fewer than 2 valid policies.")
            continue

        reward_dim = len(policies[0]["reward_vector"])
        users = make_default_users(reward_dim)

        max_queries = args.max_queries or max(1, len(policies) // 2)

        shared = dict(
            policies=policies,
            users=users,
            strategies=args.strategies,
            max_queries=max_queries,
            query_step=args.query_step,
            seeds=args.seeds,
            regret_threshold=args.regret_threshold,
            disable_budget_search=args.disable_budget_search,
            fixed_query_budget=args.fixed_query_budget,
        )

        set_rows = []

        if "logistic_regression" in args.preference_models:
            print("\n  [Logistic Regression]")
            for r in _run_lr(**shared):
                row = dataclasses.asdict(r)
                row["preference_model"] = "logistic_regression"
                row["policy_set"] = rel
                set_rows.append(row)

        if "bradley_terry" in args.preference_models:
            print("\n  [Bradley-Terry]")
            for r in _run_bt(**shared, l2=args.l2):
                row = dataclasses.asdict(r)
                row["preference_model"] = "bradley_terry"
                row["policy_set"] = rel
                set_rows.append(row)

        if "feature_bt" in args.preference_models:
            print("\n  [Feature Bradley-Terry]")
            for r in _run_fbt(**shared, l2=args.l2, ridge_alpha=args.ridge_alpha):
                row = dataclasses.asdict(r)
                row["preference_model"] = "feature_bt"
                row["policy_set"] = rel
                set_rows.append(row)

        # Save per-policy-set results
        out_path = os.path.join(args.output_dir, rel.replace(os.sep, "_"))
        with open(out_path, "w") as f:
            json.dump(set_rows, f, indent=2)
        print(f"\n  Saved {len(set_rows)} rows → {out_path}")

        all_rows.extend(set_rows)

    # Save combined results
    combined_path = os.path.join(args.output_dir, "all_results.json")
    with open(combined_path, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\nSaved {len(all_rows)} total rows → {combined_path}")


if __name__ == "__main__":
    main()
