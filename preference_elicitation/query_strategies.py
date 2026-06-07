import random
import numpy as np


def policy_features(policy: dict) -> np.ndarray:
    rewards = np.asarray(policy["reward_vector"], dtype=float)

    total_reward = float(np.sum(rewards))
    min_reward = float(np.min(rewards))
    max_reward = float(np.max(rewards))
    mean_reward = float(np.mean(rewards))

    if total_reward == 0:
        gini = 0.0
    else:
        diff_sum = np.abs(rewards[:, None] - rewards[None, :]).sum()
        gini = float(diff_sum / (2 * len(rewards) * total_reward))

    sen = total_reward * (1.0 - gini)

    return np.concatenate(
        [
            rewards,
            np.array([total_reward, min_reward, max_reward, mean_reward, gini, sen]),
        ]
    )


def random_query(policies, model=None, asked_pairs=None):
    n = len(policies)
    asked_pairs = asked_pairs or set()

    possible = [
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if (i, j) not in asked_pairs
    ]

    if not possible:
        return None

    return random.choice(possible)


def uncertainty_query(policies, model, asked_pairs=None, sample_size=500):
    """
    Chooses the pair closest to 50/50 predicted preference.
    Falls back to random querying if the model is not fitted.
    """
    asked_pairs = asked_pairs or set()
    n = len(policies)

    possible = [
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if (i, j) not in asked_pairs
    ]

    if not possible:
        return None

    if model is None or not getattr(model, "is_fitted", False):
        return random.choice(possible)

    if len(possible) > sample_size:
        possible = random.sample(possible, sample_size)

    best_pair = None
    best_uncertainty = float("inf")

    for i, j in possible:
        prob_a = model.predict_preference_probability(policies[i], policies[j])
        uncertainty = abs(prob_a - 0.5)

        if uncertainty < best_uncertainty:
            best_uncertainty = uncertainty
            best_pair = (i, j)

    return best_pair