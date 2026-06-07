import numpy as np


def gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if np.sum(x) == 0:
        return 0.0
    diff_sum = np.abs(x[:, None] - x[None, :]).sum()
    return float(diff_sum / (2 * len(x) * np.sum(x)))


def sen_welfare(x: np.ndarray) -> float:
    return float(np.sum(x) * (1.0 - gini(x)))


class SimulatedUser:
    """
    Simulated decision-maker with a hidden utility function.
    Used to answer pairwise preference queries and evaluate final selected policies.
    """

    def __init__(self, name, weights=None, fairness_weight=0.0, noise=0.0):
        self.name = name
        self.weights = None if weights is None else np.asarray(weights, dtype=float)
        self.fairness_weight = fairness_weight
        self.noise = noise

    def utility(self, policy: dict) -> float:
        rewards = np.asarray(policy["reward_vector"], dtype=float)

        if self.weights is None:
            weights = np.ones(len(rewards)) / len(rewards)
        else:
            weights = self.weights / np.sum(self.weights)

        base_utility = float(np.dot(weights, rewards))
        inequality_penalty = gini(rewards)

        return base_utility - self.fairness_weight * inequality_penalty

    def prefer(self, policy_a: dict, policy_b: dict) -> int:
        """
        Returns:
            1 if A is preferred,
            0 if B is preferred.
        """
        ua = self.utility(policy_a)
        ub = self.utility(policy_b)

        if self.noise > 0:
            ua += np.random.normal(0, self.noise)
            ub += np.random.normal(0, self.noise)

        return 1 if ua >= ub else 0


def make_default_users(reward_dim: int):
    equal = np.ones(reward_dim) / reward_dim

    users = [
        SimulatedUser(
            name="efficiency_focused",
            weights=equal,
            fairness_weight=0.0,
            noise=0.0,
        ),
        SimulatedUser(
            name="fairness_focused",
            weights=equal,
            fairness_weight=0.5,
            noise=0.0,
        ),
    ]

    chosen = np.random.choice(reward_dim, size=2, replace=False)
    for i in chosen:
        w = np.ones(reward_dim) * 0.1
        w[i] = 1.0
        users.append(
            SimulatedUser(
                name=f"group_{i + 1}_priority",
                weights=w,
                fairness_weight=0.1,
                noise=0.0,
            )
        )

    return users