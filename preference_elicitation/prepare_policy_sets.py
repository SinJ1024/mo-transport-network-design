"""
Convert local wandb table JSON files into policy-set JSON files that match
the format expected by the preference-elicitation models.

Output format per policy (fields not present in the source data are omitted):
    {
        "policy_id": <int>,
        "reward_vector": [float, ...],
        "lcn_lambda": <float or null>,
        "states": []
    }

For each configured folder the script writes:
  • policy_sets/<model>/<seed_file>.json  — one file per source JSON (per seed)
  • policy_sets/<model>.json              — all seeds merged (aggregate)

Folder → lcn_lambda mapping is configured in FOLDER_CONFIG.
"""

import glob
import json
import os
import re


FOLDER_CONFIG = [
    ("lambda_lcn",           0.5),
    ("pcn",                  None),
    ("lcn",                  None),
    ("gcn",                  None),
    ("lambda scheduler lcn", None),
]


def _sanitize(filename: str) -> str:
    """Turn a raw filename into a safe slug, e.g. 'front.table (1).json' → 'front_table_1'."""
    name = os.path.splitext(filename)[0]         
    name = re.sub(r"[^\w]+", "_", name)           
    name = name.strip("_")
    return name


def _parse_table(filepath: str, lcn_lambda, id_offset: int = 0) -> list:
    """Parse one wandb table JSON; return policies with policy_ids starting at id_offset."""
    with open(filepath, "r") as f:
        table = json.load(f)

    columns = table["columns"]
    data = table["data"]

    obj_indices = [i for i, c in enumerate(columns) if c.startswith("objective")]
    if not obj_indices:
        obj_indices = list(range(len(columns)))

    policies = []
    for pid, row in enumerate(data):
        reward_vector = [
            float(row[i]) if row[i] is not None else 0.0
            for i in obj_indices
        ]
        policies.append({
            "policy_id": id_offset + pid,
            "reward_vector": reward_vector,
            "lcn_lambda": lcn_lambda,
            "states": [],
        })
    return policies


def _save(policies: list, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(policies, f, indent=2)
    print(f"    Saved {len(policies):>4} policies → {path}")


def main():
    for folder_path, lcn_lambda in FOLDER_CONFIG:
        model_name = os.path.basename(os.path.normpath(folder_path))
        per_seed_dir = os.path.join("policy_sets", model_name)
        aggregate_path = os.path.join("policy_sets", f"{model_name}.json")

        print(f"\nProcessing {folder_path!r}  (lcn_lambda={lcn_lambda})")

        json_files = sorted(glob.glob(os.path.join(folder_path, "*.json")))
        if not json_files:
            print(f"  WARNING: no JSON files found — skipping.")
            continue

        all_policies = []
        global_id = 0

        for filepath in json_files:
            slug = _sanitize(os.path.basename(filepath))
            policies = _parse_table(filepath, lcn_lambda, id_offset=global_id)

            per_seed_policies = [
                {**p, "policy_id": i} for i, p in enumerate(policies)
            ]
            _save(per_seed_policies, os.path.join(per_seed_dir, f"{slug}.json"))

            all_policies.extend(policies)
            global_id += len(policies)

        _save(all_policies, aggregate_path)


if __name__ == "__main__":
    main()
