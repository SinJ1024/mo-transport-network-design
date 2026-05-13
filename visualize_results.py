"""Visualize trained LCN policies: Pareto front, city grid routes, and per-group reward bars.

Usage:
    # Evaluate a saved model and plot (requires same env config as training)
    python visualize_results.py --env dilemma --nr_stations 9 \
        --model_path results/lcn_dilemma_.../LCN_model_0.pt \
        --lcn_lambda 0.5 --distance_ref interpolate2

    # Compare multiple runs (provide multiple model paths)
    python visualize_results.py --env dilemma --nr_stations 9 \
        --model_path results/run1/LCN_model_0.pt results/run2/LCN_model_0.pt \
        --labels "baseline" "temporal" \
        --lcn_lambda 0.5 --distance_ref interpolate2

    # Just plot from a wandb CSV export
    python visualize_results.py --from_csv wandb_export.csv
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_agent_and_evaluate(args, model_path):
    """Load a saved LCN model and run evaluation."""
    import mo_gymnasium as mo_gym
    import torch as th

    from morl_baselines.multi_policy.lcn.lcn import LCNTNDP, LCNTNDPModel
    from motndp.city import City
    from motndp.constraints import MetroConstraints
    import envs

    env_configs = {
        'dilemma': dict(
            city_path=Path("./envs/mo-tndp/cities/dilemma_5x5"),
            gym_env='motndp_dilemma-v0', groups_file="groups.txt",
            ignore_existing_lines=True,
            scaling_factor=np.array([1, 1, 0.1]),
            ref_point=np.array([0, 0]),
            max_return=np.array([1, 1]),
        ),
        'margins': dict(
            city_path=Path("./envs/mo-tndp/cities/margins_5x5"),
            gym_env='motndp_margins-v0', groups_file="groups.txt",
            ignore_existing_lines=True,
            scaling_factor=np.array([1, 1, 0.1]),
            ref_point=np.array([0, 0]),
            max_return=np.array([1, 1]),
        ),
        'xian': dict(
            city_path=Path("./envs/mo-tndp/cities/xian"),
            gym_env='motndp_xian-v0',
            groups_file=f"price_groups_{args.nr_groups}.txt",
            ignore_existing_lines=True,
            scaling_factor=np.array([100] * args.nr_groups + [0.01]),
            ref_point=np.array([0] * args.nr_groups),
            max_return=np.array([1] * args.nr_groups),
        ),
        'amsterdam': dict(
            city_path=Path("./envs/mo-tndp/cities/amsterdam"),
            gym_env='motndp_amsterdam-v0',
            groups_file=f"price_groups_{args.nr_groups}.txt",
            ignore_existing_lines=True,
            scaling_factor=np.array([100] * args.nr_groups + [0.01]),
            ref_point=np.array([0] * args.nr_groups),
            max_return=np.array([1] * args.nr_groups),
        ),
    }

    cfg = env_configs[args.env]
    city = City(cfg['city_path'], groups_file=cfg['groups_file'],
                ignore_existing_lines=cfg['ignore_existing_lines'])

    env = mo_gym.make(cfg['gym_env'], city=city,
                      constraints=MetroConstraints(city),
                      nr_stations=args.nr_stations, chained_reward=True)

    agent = LCNTNDP(
        env, scaling_factor=cfg['scaling_factor'],
        learning_rate=args.lr, batch_size=256,
        hidden_dim=args.hidden_dim, nr_layers=args.nr_layers,
        distance_ref=args.distance_ref, lcn_lambda=args.lcn_lambda,
        log=False, model_class=LCNTNDPModel,
    )

    agent.model = th.load(model_path, map_location=agent.device, weights_only=False)
    agent.model.eval()

    # Need to fill ER buffer minimally so evaluate() works
    agent.experience_replay = []
    starting_loc = (args.starting_loc_x, args.starting_loc_y) if args.starting_loc_x is not None else None
    from morl_baselines.multi_policy.lcn.lcn import Transition
    for ep_i in range(args.num_eval_policies + 5):
        transitions = []
        obs, info = env.reset(options={'loc': starting_loc})
        done = False
        while not done:
            action = env.action_space.sample(mask=info['action_mask'])
            n_obs, reward, terminated, truncated, info = env.step(action)
            transitions.append(Transition(obs, action, info['action_mask'],
                                          np.float32(reward).copy(), n_obs, terminated))
            done = terminated or truncated
            obs = n_obs
        agent._add_episode(transitions, max_size=100, step=ep_i)

    e_returns, _, _, e_states, e_cell_sat = agent.evaluate(
        env, cfg['max_return'], n=args.num_eval_policies, starting_loc=starting_loc)

    return e_returns, e_states, e_cell_sat, city, cfg


# ── Plot functions ──────────────────────────────────────────────────────

def plot_pareto_front(all_returns, labels=None, save_path=None):
    """Scatter plot of Pareto front approximations."""
    n_obj = all_returns[0].shape[1]
    if n_obj < 2:
        print("Need at least 2 objectives for Pareto front plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    colors = plt.cm.tab10.colors

    for i, returns in enumerate(all_returns):
        label = labels[i] if labels else f"Run {i+1}"
        ax.scatter(returns[:, 0], returns[:, 1],
                   c=[colors[i % len(colors)]], marker=markers[i % len(markers)],
                   alpha=0.7, s=60, edgecolors='k', linewidths=0.5, label=label)

    ax.set_xlabel("Objective 1", fontsize=12)
    ax.set_ylabel("Objective 2", fontsize=12)
    ax.set_title("Pareto Front Approximation", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def plot_pareto_front_multi_obj(all_returns, labels=None, save_path=None):
    """Parallel coordinates plot for >2 objectives."""
    n_obj = all_returns[0].shape[1]
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10.colors

    for i, returns in enumerate(all_returns):
        label = labels[i] if labels else f"Run {i+1}"
        for j, r in enumerate(returns):
            ax.plot(range(n_obj), r, c=colors[i % len(colors)],
                    alpha=0.3, linewidth=1,
                    label=label if j == 0 else None)

    ax.set_xticks(range(n_obj))
    ax.set_xticklabels([f"Obj {k+1}" for k in range(n_obj)], fontsize=11)
    ax.set_ylabel("Return", fontsize=12)
    ax.set_title("Parallel Coordinates — Pareto Front", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def plot_routes_on_grid(city, all_states, labels=None, save_path=None):
    """Plot transit routes on the city grid with OD demand heatmap."""
    agg_od = city.agg_od_mx()
    n_runs = len(all_states)
    n_routes = min(5, len(all_states[0]))
    fig, axes = plt.subplots(1, n_runs, figsize=(6 * n_runs, 6), squeeze=False)
    colors = plt.cm.Set1.colors

    for run_i in range(n_runs):
        ax = axes[0, run_i]
        im = ax.imshow(agg_od, cmap='YlOrRd', alpha=0.6, origin='upper')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='OD demand')

        for route_i in range(n_routes):
            states = np.array(all_states[run_i][route_i])
            ax.plot(states[:, 1], states[:, 0], '-o',
                    color=colors[route_i % len(colors)],
                    markersize=5, linewidth=2, alpha=0.8,
                    label=f"Route {route_i+1}")
            ax.plot(states[0, 1], states[0, 0], '*',
                    color=colors[route_i % len(colors)],
                    markersize=15, markeredgecolor='k')

        title = labels[run_i] if labels else f"Run {run_i+1}"
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=8, loc='lower right')
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

    plt.suptitle("Routes on City Grid (star = start)", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def plot_group_rewards(all_returns, labels=None, save_path=None):
    """Bar chart comparing mean per-group rewards across runs."""
    n_obj = all_returns[0].shape[1]
    n_runs = len(all_returns)
    x = np.arange(n_obj)
    width = 0.8 / n_runs
    colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, returns in enumerate(all_returns):
        means = returns.mean(axis=0)
        stds = returns.std(axis=0)
        label = labels[i] if labels else f"Run {i+1}"
        ax.bar(x + i * width - 0.4 + width / 2, means, width,
               yerr=stds, label=label, color=colors[i % len(colors)],
               edgecolor='k', linewidth=0.5, capsize=3, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Group {k+1}" for k in range(n_obj)], fontsize=11)
    ax.set_ylabel("Mean Return", fontsize=12)
    ax.set_title("Per-Group Reward Comparison", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def plot_fairness_summary(all_returns, labels=None, save_path=None):
    """Radar/bar chart of fairness metrics: Gini, Sen Welfare, Nash Welfare."""
    from morl_baselines.common.performance_indicators import gini

    metrics_names = ["Efficiency\n(sum)", "Sen Welfare", "Nash Welfare", "1 - Gini\n(equality)"]
    n_runs = len(all_returns)
    colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(metrics_names))
    width = 0.8 / n_runs

    for i, returns in enumerate(all_returns):
        gi = gini(returns)
        utils_sum = np.sum(returns, axis=1)
        nash = np.prod(returns, axis=1)
        sen = utils_sum * (1 - gi)

        vals = [np.mean(utils_sum), np.mean(sen), np.mean(nash), np.mean(1 - gi)]
        label = labels[i] if labels else f"Run {i+1}"
        ax.bar(x + i * width - 0.4 + width / 2, vals, width,
               label=label, color=colors[i % len(colors)],
               edgecolor='k', linewidth=0.5, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=11)
    ax.set_title("Fairness Metrics Comparison", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def plot_cell_satisfaction_heatmap(city, cell_satisfaction, labels=None, save_path=None):
    """Heatmap of per-cell OD satisfaction rates."""
    n_runs = len(cell_satisfaction)
    fig, axes = plt.subplots(1, n_runs, figsize=(6 * n_runs, 5), squeeze=False)

    for i in range(n_runs):
        ax = axes[0, i]
        if len(cell_satisfaction[i]) == 0:
            ax.text(0.5, 0.5, "No data", ha='center', va='center')
            continue
        mean_sat = np.mean(cell_satisfaction[i], axis=0)
        sat_grid = mean_sat.reshape(city.grid_x_size, city.grid_y_size)
        im = ax.imshow(sat_grid, cmap='RdYlGn', vmin=0, vmax=max(0.01, sat_grid.max()),
                       origin='upper')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Satisfaction rate')
        title = labels[i] if labels else f"Run {i+1}"
        ax.set_title(title, fontsize=13)

    plt.suptitle("Mean Cell Satisfaction Rate", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def plot_from_csv(csv_path, save_dir=None):
    """Plot training curves from a wandb CSV export."""
    df = pd.read_csv(csv_path)
    print(f"Columns: {list(df.columns)}")

    metric_groups = {
        "Hypervolume": [c for c in df.columns if 'hypervolume' in c.lower()],
        "Loss": [c for c in df.columns if 'loss' in c.lower()],
        "Gini": [c for c in df.columns if 'gini' in c.lower()],
        "Sen Welfare": [c for c in df.columns if 'sen_welfare' in c.lower()],
        "Lambda": [c for c in df.columns if 'lcn_lambda' in c.lower()],
        "Max-Min Floor": [c for c in df.columns if 'maxmin_floor' in c.lower()],
        "Spatial SW": [c for c in df.columns if 'spatial_sw' in c.lower()],
    }

    step_col = 'global_step' if 'global_step' in df.columns else df.columns[0]

    for group_name, cols in metric_groups.items():
        cols = [c for c in cols if c in df.columns]
        if not cols:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        for c in cols:
            data = df[[step_col, c]].dropna()
            ax.plot(data[step_col], data[c], label=c, linewidth=1.5)
        ax.set_xlabel("Global Step", fontsize=11)
        ax.set_ylabel(group_name, fontsize=11)
        ax.set_title(group_name, fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_dir:
            path = Path(save_dir) / f"{group_name.lower().replace(' ', '_')}.png"
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved: {path}")
        plt.show()


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize LCN training results")
    parser.add_argument('--env', default='dilemma', type=str,
                        choices=['dilemma', 'margins', 'xian', 'amsterdam'])
    parser.add_argument('--nr_groups', default=5, type=int)
    parser.add_argument('--nr_stations', type=int, default=9)
    parser.add_argument('--starting_loc_x', default=None, type=int)
    parser.add_argument('--starting_loc_y', default=None, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--hidden_dim', default=64, type=int)
    parser.add_argument('--nr_layers', default=1, type=int)
    parser.add_argument('--distance_ref', default='interpolate2', type=str)
    parser.add_argument('--lcn_lambda', default=0.5, type=float)
    parser.add_argument('--num_eval_policies', default=10, type=int)
    parser.add_argument('--model_path', nargs='+', type=str,
                        help='Path(s) to saved .pt model files')
    parser.add_argument('--labels', nargs='+', type=str, default=None,
                        help='Labels for each model (for legend)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save figures (default: show only)')
    parser.add_argument('--from_csv', type=str, default=None,
                        help='Plot training curves from wandb CSV export')

    args = parser.parse_args()

    if args.from_csv:
        plot_from_csv(args.from_csv, save_dir=args.save_dir)
        return

    if not args.model_path:
        parser.error("Provide --model_path or --from_csv")

    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    all_returns = []
    all_states = []
    all_cell_sat = []
    city = None

    for mp in args.model_path:
        print(f"Loading and evaluating: {mp}")
        e_returns, e_states, e_cell_sat, city, cfg = load_agent_and_evaluate(args, mp)
        all_returns.append(e_returns)
        all_states.append(e_states)
        all_cell_sat.append(e_cell_sat)

    labels = args.labels or [Path(p).parent.name for p in args.model_path]
    save = lambda name: str(Path(args.save_dir) / name) if args.save_dir else None

    n_obj = all_returns[0].shape[1]
    if n_obj == 2:
        plot_pareto_front(all_returns, labels, save("pareto_front.png"))
    else:
        plot_pareto_front_multi_obj(all_returns, labels, save("pareto_parallel.png"))

    plot_group_rewards(all_returns, labels, save("group_rewards.png"))
    plot_fairness_summary(all_returns, labels, save("fairness_metrics.png"))
    plot_routes_on_grid(city, all_states, labels, save("routes.png"))

    if any(len(cs) > 0 for cs in all_cell_sat):
        plot_cell_satisfaction_heatmap(city, all_cell_sat, labels, save("cell_satisfaction.png"))


if __name__ == "__main__":
    main()
