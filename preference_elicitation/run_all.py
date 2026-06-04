"""
End-to-end pipeline: prepare policy sets → run all preference models → plot.

Usage
-----
    python run_all.py
    python run_all.py --skip-prepare   # if policy_sets/ already exists
    python run_all.py --skip-eval      # only re-plot existing results
"""

import argparse
import glob
import subprocess
import sys
import os


def run(cmd: list[str], label: str):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\nERROR: '{label}' exited with code {result.returncode}. Aborting.")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run the full preference elicitation pipeline.")
    parser.add_argument("--skip-prepare", action="store_true",
                        help="Skip policy-set preparation (reuse existing policy_sets/).")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip model evaluation (only re-plot existing results).")
    parser.add_argument(
        "--preference-models", nargs="+",
        default=["logistic_regression", "bradley_terry", "feature_bt"],
        choices=["logistic_regression", "bradley_terry", "feature_bt"],
    )
    parser.add_argument("--output-dir", default="results",
                        help="Directory for result JSON files.")
    parser.add_argument("--figures-dir", default="figures",
                        help="Directory for output figures.")
    parser.add_argument("--regret-threshold", type=float, default=0.05)
    args = parser.parse_args()

    py = sys.executable
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    # ── Step 1: Prepare policy sets ───────────────────────────────────────────
    if not args.skip_prepare:
        run([py, "prepare_policy_sets.py"], "Prepare policy sets from local folders")
    else:
        print("\n[Step 1] Skipping policy-set preparation.")

    # ── Step 2: Run preference elicitation for each model set ─────────────────
    combined_output = os.path.join(args.output_dir, "all_results.json")

    if not args.skip_eval:
        run(
            [
                py, "run_evaluation.py",
                "--policy-sets-dir", "policy_sets",
                "--preference-models", *args.preference_models,
                "--output-dir", args.output_dir,
            ],
            "Run preference elicitation (all models × all policy sets)",
        )
    else:
        print("\n[Step 2] Skipping preference elicitation evaluation.")

    # ── Step 3: Plot ──────────────────────────────────────────────────────────
    # Collect whichever per-model result files exist
    per_model_files = sorted(glob.glob(os.path.join(args.output_dir, "*.json")))
    per_model_files = [f for f in per_model_files if os.path.basename(f) != "all_results.json"]
    if not per_model_files:
        per_model_files = [combined_output] if os.path.exists(combined_output) else []

    if per_model_files:
        run(
            [
                py, "plot_results.py",
                "--results", *per_model_files,
                "--output", args.figures_dir,
                "--regret-threshold", str(args.regret_threshold),
            ],
            "Generate plots",
        )
    else:
        print("\n[Step 3] No result files found — skipping plotting.")

    print(f"\n{'='*60}")
    print("  Pipeline complete.")
    print(f"  Results : {args.output_dir}/")
    print(f"  Figures : {args.figures_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
