import argparse
import json
import os
import random
from dataclasses import dataclass

import numpy as np
from scipy.stats import kendalltau
import choix

from users import make_default_users
from query_strategies import policy_features, random_query, uncertainty_query
from linear_regression_runner import preprocess_policies, load_policies


@dataclass
class FBTResult:
    user_name: str
    strategy: str
    query_budget: int
    selected_policy_id: int
    true_best_policy_id: int
    regret: float
    normalized_regret: float
    selected_rank: float
    selected_utility: float
    best_utility: float
    top5_ranking_mismatch: float
    policy_distance: float
    kendall_tau: float
    mean_rank_displacement: float


class FeatureBTModel:
    """
    Two-step feature-based Bradley-Terry model built on choix.

    Step 1 — BT scoring (choix):
        choix.opt_pairwise estimates a latent score for every policy from
        pairwise comparisons.  This is the standard item-based BT fit.

    Step 2 — Feature regression (ridge):
        A ridge regression maps φ(policy) → BT score for all policies that
        appeared in at least one comparison.  The learned weight vector w then
        lets us score *any* policy (including never-compared ones) via
            score(p) = w · φ(p)

    This combines choix's principled BT optimisation with the parameter
    efficiency of a feature-based model.
    """

    def __init__(
        self,
        n_policies: int,
        n_features: int,
        l2: float = 1e-3,
        ridge_alpha: float = 1.0,
        method: str = "BFGS",
        max_iter: int = 500,
        tol: float = 1e-5,
    ):
        self.n_policies = n_policies
        self.n_features = n_features
        self.l2 = l2
        self.ridge_alpha = ridge_alpha
        self.method = method
        self.max_iter = max_iter
        self.tol = tol

        self.weights = np.zeros(n_features, dtype=float)
        self._item_scores = np.zeros(n_policies, dtype=float)
        self.is_fitted = False

    @staticmethod
    def _to_choix_data(comparisons):
        return [(i, j) if pref == 1 else (j, i) for i, j, pref in comparisons]

    def fit(self, comparisons, policies):
        """
        comparisons : list of (i, j, pref)
            pref = 1  →  policy i preferred over policy j
            pref = 0  →  policy j preferred over policy i
        policies    : list of policy dicts
        """
        if len(comparisons) == 0:
            return

        pairwise_data = self._to_choix_data(comparisons)

        initial = np.array([self.score(p) for p in policies])

        item_scores = choix.opt_pairwise(
            n_items=self.n_policies,
            data=pairwise_data,
            alpha=self.l2,
            method=self.method,
            initial_params=initial,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        item_scores = item_scores - np.mean(item_scores)
        self._item_scores = item_scores

        observed = sorted({idx for i, j, _ in comparisons for idx in (i, j)})
        Phi = np.array([policy_features(policies[k]) for k in observed])  # (m, d)
        s = item_scores[observed]                                          # (m,)

        A = Phi.T @ Phi + self.ridge_alpha * np.eye(self.n_features)
        self.weights = np.linalg.solve(A, Phi.T @ s)

        self.is_fitted = True

    def score(self, policy) -> float:
        return float(np.dot(self.weights, policy_features(policy)))

    def select_best(self, policies) -> int:
        scores = np.array([self.score(p) for p in policies])
        return int(np.argmax(scores))

    def predict_preference_probability(self, policy_a, policy_b) -> float:
        if not self.is_fitted:
            return 0.5
        delta = policy_features(policy_a) - policy_features(policy_b)
        logit = float(np.dot(self.weights, delta))
        return float(1.0 / (1.0 + np.exp(-np.clip(logit, -50, 50))))


def evaluate_selected_policy(policies, selected_idx, user, predicted_scores=None):
    utilities = np.asarray([user.utility(p) for p in policies], dtype=float)

    best_idx = int(np.argmax(utilities))
    best_utility = float(utilities[best_idx])
    selected_utility = float(utilities[selected_idx])

    regret = best_utility - selected_utility
    utility_range = float(np.max(utilities) - np.min(utilities))
    normalized_regret = 0.0 if utility_range == 0 else regret / utility_range

    ranked_indices = list(np.argsort(-utilities))
    selected_rank = ranked_indices.index(selected_idx) + 1

    selected_rv = np.asarray(policies[selected_idx]["reward_vector"], dtype=float)
    best_rv = np.asarray(policies[best_idx]["reward_vector"], dtype=float)
    policy_distance = float(np.linalg.norm(selected_rv - best_rv))

    if predicted_scores is not None:
        predicted_scores_arr = np.asarray(predicted_scores, dtype=float)
        k = min(5, len(policies))
        true_top_k = set(np.argsort(-utilities)[:k].tolist())
        pred_top_k = set(np.argsort(-predicted_scores_arr)[:k].tolist())
        top5_ranking_mismatch = float(k - len(true_top_k & pred_top_k))
        true_ranks = np.argsort(np.argsort(-utilities))
        pred_ranks = np.argsort(np.argsort(-predicted_scores_arr))
        tau, _ = kendalltau(true_ranks, pred_ranks)
        kendall_tau = float(tau)
        mean_rank_displacement = float(np.mean(pred_ranks.astype(float) - true_ranks.astype(float)))
    else:
        top5_ranking_mismatch = float(min(5, len(policies)))
        kendall_tau = 0.0
        mean_rank_displacement = 0.0

    return {
        "true_best_policy_id": policies[best_idx]["policy_id"],
        "best_utility": best_utility,
        "selected_utility": selected_utility,
        "regret": regret,
        "normalized_regret": normalized_regret,
        "selected_rank": selected_rank,
        "top5_ranking_mismatch": top5_ranking_mismatch,
        "policy_distance": policy_distance,
        "kendall_tau": kendall_tau,
        "mean_rank_displacement": mean_rank_displacement,
    }


def run_fbt_elicitation(policies, user, strategy_name, query_budget, seed=0, l2=1e-3, ridge_alpha=1.0):
    random.seed(seed)
    np.random.seed(seed)

    n_features = len(policy_features(policies[0]))
    model = FeatureBTModel(
        n_policies=len(policies),
        n_features=n_features,
        l2=l2,
        ridge_alpha=ridge_alpha,
    )
    comparisons = []
    asked_pairs = set()

    for _ in range(query_budget):
        if strategy_name == "random":
            pair = random_query(policies, model=model, asked_pairs=asked_pairs)
        elif strategy_name == "uncertainty":
            pair = uncertainty_query(policies, model=model, asked_pairs=asked_pairs)
        else:
            raise ValueError(f"Unknown query strategy: {strategy_name}")

        if pair is None:
            break

        i, j = pair
        asked_pairs.add(tuple(sorted((i, j))))

        pref = user.prefer(policies[i], policies[j])
        comparisons.append((i, j, pref))

        model.fit(comparisons, policies)

    if model.is_fitted:
        selected_idx = model.select_best(policies)
        predicted_scores = [model.score(p) for p in policies]
    else:
        selected_idx = random.randrange(len(policies))
        predicted_scores = None

    eval_result = evaluate_selected_policy(policies, selected_idx, user, predicted_scores)

    return FBTResult(
        user_name=user.name,
        strategy=strategy_name,
        query_budget=query_budget,
        selected_policy_id=policies[selected_idx]["policy_id"],
        true_best_policy_id=eval_result["true_best_policy_id"],
        regret=eval_result["regret"],
        normalized_regret=eval_result["normalized_regret"],
        selected_rank=eval_result["selected_rank"],
        selected_utility=eval_result["selected_utility"],
        best_utility=eval_result["best_utility"],
        top5_ranking_mismatch=eval_result["top5_ranking_mismatch"],
        policy_distance=eval_result["policy_distance"],
        kendall_tau=eval_result["kendall_tau"],
        mean_rank_displacement=eval_result["mean_rank_displacement"],
    )


def average_results(results):
    first = results[0]
    return FBTResult(
        user_name=first.user_name,
        strategy=first.strategy,
        query_budget=first.query_budget,
        selected_policy_id=first.selected_policy_id,
        true_best_policy_id=first.true_best_policy_id,
        regret=float(np.mean([r.regret for r in results])),
        normalized_regret=float(np.mean([r.normalized_regret for r in results])),
        selected_rank=float(np.mean([r.selected_rank for r in results])),
        selected_utility=float(np.mean([r.selected_utility for r in results])),
        best_utility=float(np.mean([r.best_utility for r in results])),
        top5_ranking_mismatch=float(np.mean([r.top5_ranking_mismatch for r in results])),
        policy_distance=float(np.mean([r.policy_distance for r in results])),
        kendall_tau=float(np.mean([r.kendall_tau for r in results])),
        mean_rank_displacement=float(np.mean([r.mean_rank_displacement for r in results])),
    )


def find_min_queries(results, regret_threshold):
    sorted_results = sorted(results, key=lambda r: r.query_budget)
    for r in sorted_results:
        if r.normalized_regret <= regret_threshold:
            return r.query_budget
    return None


def run_budget_search(
    policies,
    users,
    strategies,
    max_queries=None,
    query_step=1,
    seeds=(0,),
    regret_threshold=0.05,
    disable_budget_search=False,
    fixed_query_budget=None,
    l2=1e-3,
    ridge_alpha=1.0,
):
    if max_queries is None:
        max_queries = max(1, len(policies) // 2)

    if disable_budget_search:
        if fixed_query_budget is None:
            fixed_query_budget = max_queries
        query_budgets = [fixed_query_budget]
    else:
        query_budgets = list(range(1, max_queries + 1, query_step))

    all_results = []

    for user in users:
        for strategy in strategies:
            user_strategy_results = []

            for q in query_budgets:
                seed_results = []

                for seed in seeds:
                    result = run_fbt_elicitation(
                        policies=policies,
                        user=user,
                        strategy_name=strategy,
                        query_budget=q,
                        seed=seed,
                        l2=l2,
                        ridge_alpha=ridge_alpha,
                    )
                    seed_results.append(result)

                avg_result = average_results(seed_results)
                user_strategy_results.append(avg_result)
                all_results.append(avg_result)

            min_q = find_min_queries(
                user_strategy_results,
                regret_threshold=regret_threshold,
            )

            print(
                f"FBT | User={user.name:20s} | Strategy={strategy:12s} | "
                f"min queries @ regret<={regret_threshold}: {min_q}"
            )

    return all_results


def save_results(results, output_path):
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    rows = []
    for r in results:
        rows.append(
            {
                "model": "feature_bradley_terry",
                "user_name": r.user_name,
                "strategy": r.strategy,
                "query_budget": r.query_budget,
                "selected_policy_id": r.selected_policy_id,
                "true_best_policy_id": r.true_best_policy_id,
                "regret": r.regret,
                "normalized_regret": r.normalized_regret,
                "selected_rank": r.selected_rank,
                "selected_utility": r.selected_utility,
                "best_utility": r.best_utility,
                "top5_ranking_mismatch": r.top5_ranking_mismatch,
                "policy_distance": r.policy_distance,
                "kendall_tau": r.kendall_tau,
            }
        )

    with open(output_path, "w") as f:
        json.dump(rows, f, indent=2)

    print(f"Saved feature-BT results to {output_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--policy-set", type=str, required=True)
    parser.add_argument("--output", type=str, default="results/feature_bt_results.json")
    parser.add_argument("--strategies", nargs="+", default=["random", "uncertainty"])
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--query-step", type=int, default=1)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--regret-threshold", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--ridge-alpha", type=float, default=1.0,
                        help="Ridge regularisation for the feature regression step.")

    parser.add_argument(
        "--disable-budget-search",
        action="store_true",
        help="Run only one fixed query budget instead of scanning many budgets.",
    )
    parser.add_argument(
        "--fixed-query-budget",
        type=int,
        default=None,
        help="Used only when --disable-budget-search is active.",
    )

    args = parser.parse_args()

    policies = preprocess_policies(load_policies(args.policy_set))
    print(f"Using {len(policies)} preprocessed policies.")

    if len(policies) < 2:
        raise ValueError("Need at least two policies for preference elicitation.")

    reward_dim = len(policies[0]["reward_vector"])
    users = make_default_users(reward_dim)

    max_queries = args.max_queries
    if max_queries is None:
        max_queries = max(1, len(policies) // 2)

    results = run_budget_search(
        policies=policies,
        users=users,
        strategies=args.strategies,
        max_queries=max_queries,
        query_step=args.query_step,
        seeds=args.seeds,
        regret_threshold=args.regret_threshold,
        disable_budget_search=args.disable_budget_search,
        fixed_query_budget=args.fixed_query_budget,
        l2=args.l2,
        ridge_alpha=args.ridge_alpha,
    )

    save_results(results, args.output)


if __name__ == "__main__":
    main()
