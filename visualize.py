"""Unified visualizer for GCN and LCN trained models.

Usage examples:

  # GCN – routes + pareto + fairness
  python visualize.py --algo gcn --env xian --nr_groups 10 --nr_stations 20 \\
      --starting_loc_x 9 --starting_loc_y 19 \\
      --criterion lorenz --distance_ref interpolate2 --gcn_lambda 0.0 \\
      --spatial_alpha 1.0 --include_demand_context \\
      --model_path results/gcn_xian_.../GCN_model_29.pt \\
      --save_dir viz_out

  # LCN – compare two runs
  python visualize.py --algo lcn --env xian --nr_groups 3 --nr_stations 20 \\
      --starting_loc_x 19 --starting_loc_y 9 --lcn_lambda 0.5 \\
      --model_path results/run1/LCN_model_0.pt results/run2/LCN_model_0.pt \\
      --labels "baseline" "spatial" --save_dir viz_out

  # Select specific plots
  python visualize.py --algo gcn ... --plots routes rewards

  # Plot training curves from wandb CSV
  python visualize.py --from_csv wandb_export.csv --save_dir viz_out
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── Environment config ───────────────────────────────────────────────────────

def get_env_config(args):
    if args.env in ("dilemma", "margins"):
        city_name = "dilemma_5x5" if args.env == "dilemma" else "margins_5x5"
        return dict(
            city_path=Path(f"./envs/mo-tndp/cities/{city_name}"),
            gym_env=f"motndp_{args.env}-v0",
            groups_file="groups.txt",
            ignore_existing_lines=True,
            scaling_factor=np.array([1, 1, 0.1]),
            ref_point=np.array([0, 0]),
            max_return=np.array([1, 1]),
        )
    else:
        return dict(
            city_path=Path(f"./envs/mo-tndp/cities/{args.env}"),
            gym_env=f"motndp_{args.env}-v0",
            groups_file=f"price_groups_{args.nr_groups}.txt",
            ignore_existing_lines=True,
            scaling_factor=np.array([100] * args.nr_groups + [0.01]),
            ref_point=np.array([0] * args.nr_groups),
            max_return=np.array([1] * args.nr_groups),
        )


def make_city_and_env(args, cfg):
    import mo_gymnasium as mo_gym
    import envs
    from motndp.city import City
    from motndp.constraints import MetroConstraints

    city = City(cfg["city_path"], groups_file=cfg["groups_file"],
                ignore_existing_lines=cfg["ignore_existing_lines"])
    env = mo_gym.make(cfg["gym_env"], city=city,
                      constraints=MetroConstraints(city),
                      nr_stations=args.nr_stations,
                      chained_reward=True,
                      include_demand_context=args.include_demand_context)
    return city, env


# ── Experience replay filler (shared by both algos) ──────────────────────────

def fill_er(agent, env, starting_loc, n_episodes, max_buffer_size, Transition):
    print(f"  Filling experience replay ({n_episodes} random episodes)...")
    for i in range(n_episodes):
        obs, info = env.reset(options={"loc": starting_loc})
        transitions, done = [], False
        while not done:
            action = env.action_space.sample(mask=info["action_mask"])
            n_obs, reward, terminated, truncated, info = env.step(action)
            transitions.append(Transition(obs, action, info["action_mask"],
                                          np.float32(reward).copy(), n_obs, terminated))
            done = terminated or truncated
            obs = n_obs
        agent._add_episode(transitions, max_size=max_buffer_size, step=i)


# ── LCN loader ───────────────────────────────────────────────────────────────

def load_and_eval_lcn(args, model_path, cfg):
    import torch as th
    from morl_baselines.multi_policy.lcn.lcn import LCNTNDP, LCNTNDPModel, Transition

    city, env = make_city_and_env(args, cfg)
    starting_loc = (args.starting_loc_x, args.starting_loc_y) \
        if args.starting_loc_x is not None else None

    agent = LCNTNDP(env, scaling_factor=cfg["scaling_factor"],
                    learning_rate=args.lr, batch_size=256,
                    hidden_dim=args.hidden_dim, nr_layers=args.nr_layers,
                    distance_ref=args.distance_ref, lcn_lambda=args.lcn_lambda,
                    log=False, model_class=LCNTNDPModel)

    agent.model = th.load(model_path, map_location=agent.device, weights_only=False)
    agent.model.eval()
    agent.cd_threshold = 0.2
    agent.experience_replay = []

    fill_er(agent, env, starting_loc,
            n_episodes=args.num_eval_policies + 5,
            max_buffer_size=100, Transition=Transition)

    e_returns, _, _, e_states, e_cell_sat = agent.evaluate(
        env, cfg["max_return"], n=args.num_eval_policies, starting_loc=starting_loc)
    return e_returns, e_states, e_cell_sat, city


# ── GCN loader ───────────────────────────────────────────────────────────────

def load_and_eval_gcn(args, model_path, cfg):
    import torch
    from morl_baselines.multi_policy.gcn.gcn import GCN, Transition
    from morl_baselines.multi_policy.gcn.gcn_model_classes import DefaultGCNModel
    from morl_baselines.multi_policy.gcn.fairness_funcs import (
        get_non_pareto_dominated, get_nash_dominated,
        pareto_l2, lorenz_l2, nash_l2,
    )

    city, env = make_city_and_env(args, cfg)
    starting_loc = (args.starting_loc_x, args.starting_loc_y) \
        if args.starting_loc_x is not None else None

    if args.criterion == "pareto":
        dominance_func, l2_func, l2_params = get_non_pareto_dominated, pareto_l2, {}
    elif args.criterion == "lorenz":
        dominance_func, l2_func = get_non_pareto_dominated, lorenz_l2
        l2_params = {"distance_ref": args.distance_ref,
                     "lcn_lambda": args.gcn_lambda,
                     "spatial_alpha": args.spatial_alpha}
    else:  # nash
        dominance_func, l2_func = get_nash_dominated, nash_l2
        l2_params = {"mode": args.nash_mode, "shift": args.nash_shift}

    agent = GCN(env, scaling_factor=cfg["scaling_factor"],
                learning_rate=args.lr, batch_size=256,
                hidden_dim=args.hidden_dim, nr_layers=args.nr_layers,
                dominance_func=dominance_func, l2_func=l2_func, l2_params=l2_params,
                log=False, seed=args.seed, model_class=DefaultGCNModel)

    agent.model = torch.load(model_path, map_location=agent.device, weights_only=False)
    agent.model.eval()

    fill_er(agent, env, starting_loc,
            n_episodes=max(60, args.num_eval_policies + 5),
            max_buffer_size=100, Transition=Transition)

    e_returns, _, _, e_states, e_cell_sat = agent.evaluate(
        env, cfg["max_return"], n=args.num_eval_policies, starting_loc=starting_loc)
    return e_returns, e_states, e_cell_sat, city


# ── Plot functions ───────────────────────────────────────────────────────────

def plot_routes_on_grid(city, all_states, labels=None, save_path=None):
    agg_od = city.agg_od_mx()
    n_runs = len(all_states)
    n_routes = min(5, len(all_states[0]))
    fig, axes = plt.subplots(1, n_runs, figsize=(6 * n_runs, 6), squeeze=False)
    colors = plt.cm.Set1.colors
    linestyles = ["-", "--", ":", "-.", (0, (5, 1))]

    for run_i in range(n_runs):
        ax = axes[0, run_i]
        im = ax.imshow(agg_od, cmap="YlOrRd", alpha=0.6, origin="upper")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="OD demand")
        for route_i in range(n_routes):
            states = np.array(all_states[run_i][route_i])
            ax.plot(states[:, 1], states[:, 0],
                    color=colors[route_i % len(colors)],
                    linestyle=linestyles[route_i % len(linestyles)],
                    marker="o", markersize=5, linewidth=2.5, alpha=0.65,
                    label=f"Route {route_i + 1}")
            ax.plot(states[0, 1], states[0, 0], "*",
                    color=colors[route_i % len(colors)],
                    markersize=15, markeredgecolor="k",
                    zorder=n_routes - route_i)
        ax.set_title(labels[run_i] if labels else f"Run {run_i + 1}", fontsize=13)
        ax.legend(fontsize=8, loc="lower right")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

    plt.suptitle("Routes on City Grid (star = start)", fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_pareto_front(all_returns, labels=None, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 6))
    markers = ["o", "s", "^", "D", "v", "<", ">", "p"]
    colors = plt.cm.tab10.colors
    for i, returns in enumerate(all_returns):
        label = labels[i] if labels else f"Run {i + 1}"
        ax.scatter(returns[:, 0], returns[:, 1],
                   c=[colors[i % len(colors)]], marker=markers[i % len(markers)],
                   alpha=0.7, s=60, edgecolors="k", linewidths=0.5, label=label)
    ax.set_xlabel("Objective 1", fontsize=12)
    ax.set_ylabel("Objective 2", fontsize=12)
    ax.set_title("Pareto Front Approximation", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_pareto_front_multi_obj(all_returns, labels=None, save_path=None):
    n_obj = all_returns[0].shape[1]
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10.colors
    for i, returns in enumerate(all_returns):
        label = labels[i] if labels else f"Run {i + 1}"
        for j, r in enumerate(returns):
            ax.plot(range(n_obj), r, c=colors[i % len(colors)],
                    alpha=0.3, linewidth=1, label=label if j == 0 else None)
    ax.set_xticks(range(n_obj))
    ax.set_xticklabels([f"Obj {k + 1}" for k in range(n_obj)], fontsize=11)
    ax.set_ylabel("Return", fontsize=12)
    ax.set_title("Parallel Coordinates — Pareto Front", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_group_rewards(all_returns, labels=None, save_path=None):
    n_obj = all_returns[0].shape[1]
    n_runs = len(all_returns)
    x = np.arange(n_obj)
    width = 0.8 / n_runs
    colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(max(8, n_obj), 5))
    for i, returns in enumerate(all_returns):
        means, stds = returns.mean(axis=0), returns.std(axis=0)
        label = labels[i] if labels else f"Run {i + 1}"
        ax.bar(x + i * width - 0.4 + width / 2, means, width,
               yerr=stds, label=label, color=colors[i % len(colors)],
               edgecolor="k", linewidth=0.5, capsize=3, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Group {k + 1}" for k in range(n_obj)], fontsize=11)
    ax.set_ylabel("Mean Return", fontsize=12)
    ax.set_title("Per-Group Reward Comparison", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_fairness_summary(all_returns, labels=None, save_path=None):
    from morl_baselines.common.performance_indicators import gini

    metrics_names = ["Efficiency\n(sum)", "Sen Welfare", "Nash Welfare\n(geom)", "1 - Gini\n(equality)"]
    n_runs = len(all_returns)
    colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(metrics_names))
    width = 0.8 / n_runs
    for i, returns in enumerate(all_returns):
        gi = gini(returns)
        utils_sum = np.sum(returns, axis=1)
        sen = utils_sum * (1 - gi)
        # Geometric-mean Nash welfare: the raw product underflows to ~0 at G=10,
        # matching evaluation.py's nash_welfare_geom.
        nash = np.exp(np.mean(np.log(np.clip(returns, 1e-8, None)), axis=1))
        vals = [np.mean(utils_sum), np.mean(sen), np.mean(nash), np.mean(1 - gi)]
        label = labels[i] if labels else f"Run {i + 1}"
        ax.bar(x + i * width - 0.4 + width / 2, vals, width,
               label=label, color=colors[i % len(colors)],
               edgecolor="k", linewidth=0.5, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=11)
    ax.set_title("Fairness Metrics Comparison", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_cell_satisfaction_heatmap(city, cell_satisfaction, labels=None, save_path=None):
    n_runs = len(cell_satisfaction)
    fig, axes = plt.subplots(1, n_runs, figsize=(6 * n_runs, 5), squeeze=False)
    for i in range(n_runs):
        ax = axes[0, i]
        if len(cell_satisfaction[i]) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            continue
        mean_sat = np.mean(cell_satisfaction[i], axis=0)
        sat_grid = mean_sat.reshape(city.grid_x_size, city.grid_y_size)
        im = ax.imshow(sat_grid, cmap="RdYlGn", vmin=0,
                       vmax=max(0.01, sat_grid.max()), origin="upper")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Satisfaction rate")
        ax.set_title(labels[i] if labels else f"Run {i + 1}", fontsize=13)
    plt.suptitle("Mean Cell Satisfaction Rate", fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def plot_from_csv(csv_path, save_dir=None):
    df = pd.read_csv(csv_path)
    print(f"Columns: {list(df.columns)}")
    metric_groups = {
        "Hypervolume": [c for c in df.columns if "hypervolume" in c.lower() and "pdim" not in c.lower()],
        "Hypervolume (per-dim)": [c for c in df.columns if "hypervolume_pdim" in c.lower()],
        "EUM": [c for c in df.columns if "eum" in c.lower()],
        "Cardinality": [c for c in df.columns if "cardinality" in c.lower()],
        "Loss": [c for c in df.columns if "loss" in c.lower()],
        "Gini": [c for c in df.columns if "gini" in c.lower()],
        "Efficiency": [c for c in df.columns if "efficiency" in c.lower()],
        "Sen Welfare": [c for c in df.columns if "sen_welfare" in c.lower()],
        "Nash Welfare (geom)": [c for c in df.columns if "nash_welfare_geom" in c.lower()],
        "Lambda": [c for c in df.columns if "lcn_lambda" in c.lower()],
        "Spatial SW": [c for c in df.columns if "spatial_sw" in c.lower()],
        "Served Floor": [c for c in df.columns if "served_floor" in c.lower()],
        "Demand Coverage": [c for c in df.columns if "demand_coverage" in c.lower()],
        "Price Equity": [c for c in df.columns if "price_equity" in c.lower()],
    }
    step_col = "global_step" if "global_step" in df.columns else df.columns[0]
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
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved: {path}")
        plt.show()


# ── Auto-detect algo from model filename ─────────────────────────────────────

def detect_algo(model_paths):
    for p in model_paths:
        name = Path(p).name.upper()
        if name.startswith("GCN"):
            return "gcn"
        if name.startswith("LCN") or name.startswith("PCN"):
            return "lcn"
    return None


# ── Main ─────────────────────────────────────────────────────────────────────

PLOT_CHOICES = ["routes", "pareto", "rewards", "fairness", "satisfaction"]

def main():
    parser = argparse.ArgumentParser(
        description="Visualize GCN / LCN trained models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Algorithm & mode
    parser.add_argument("--algo", choices=["gcn", "lcn"], default=None,
                        help="Algorithm (auto-detected from model filename if omitted)")
    parser.add_argument("--from_csv", type=str, default=None,
                        help="Plot training curves from a wandb CSV export")

    # ── Environment
    parser.add_argument("--env", default="xian",
                        choices=["dilemma", "margins", "xian", "amsterdam"])
    parser.add_argument("--nr_groups", default=5, type=int)
    parser.add_argument("--nr_stations", default=20, type=int)
    parser.add_argument("--starting_loc_x", default=None, type=int)
    parser.add_argument("--starting_loc_y", default=None, type=int)
    parser.add_argument("--include_demand_context", action="store_true", default=False,
                        help="Augment observation with OD demand context (GCN)")

    # ── Model(s)
    parser.add_argument("--model_path", nargs="+", type=str,
                        help="Path(s) to .pt model files (multiple = comparison plot)")
    parser.add_argument("--labels", nargs="+", type=str, default=None,
                        help="Legend labels for each model path")

    # ── Shared hyperparams
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--hidden_dim", default=128, type=int)
    parser.add_argument("--nr_layers", default=1, type=int)
    parser.add_argument("--distance_ref", default="interpolate2", type=str,
                        choices=["nondominated", "optimal_max", "nondominated_mean",
                                 "interpolate", "interpolate2", "interpolate3"])
    parser.add_argument("--num_eval_policies", default=5, type=int)
    parser.add_argument("--seed", default=42, type=int)

    # ── LCN-specific
    parser.add_argument("--lcn_lambda", default=0.5, type=float,
                        help="Lambda for LCN (0=full Lorenz, 1=full Pareto)")

    # ── GCN-specific
    parser.add_argument("--criterion", default="lorenz",
                        choices=["pareto", "lorenz", "nash"])
    parser.add_argument("--gcn_lambda", default=0.0, type=float)
    parser.add_argument("--spatial_alpha", default=0.0, type=float)
    parser.add_argument("--nash_mode", default="pareto_sized",
                        choices=["pareto_filter", "pareto_sized"])
    parser.add_argument("--nash_shift", default=0.0, type=float)

    # ── Output
    parser.add_argument("--plots", nargs="+", default=["all"],
                        choices=PLOT_CHOICES + ["all"],
                        help="Which plots to generate (default: all)")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory to save figures (default: show only)")

    args = parser.parse_args()

    # CSV mode – independent of algo
    if args.from_csv:
        if args.save_dir:
            Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        plot_from_csv(args.from_csv, save_dir=args.save_dir)
        return

    if not args.model_path:
        parser.error("Provide --model_path or --from_csv")

    # Resolve algo
    if args.algo is None:
        args.algo = detect_algo(args.model_path)
        if args.algo is None:
            parser.error("Cannot auto-detect --algo from model filename. Please pass --algo gcn or --algo lcn")
        print(f"Auto-detected algo: {args.algo}")

    # Resolve plots
    want = set(PLOT_CHOICES) if "all" in args.plots else set(args.plots)

    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    save = lambda name: str(Path(args.save_dir) / name) if args.save_dir else None

    import numpy as np
    import torch
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = get_env_config(args)
    loader = load_and_eval_gcn if args.algo == "gcn" else load_and_eval_lcn

    all_returns, all_states, all_cell_sat = [], [], []
    city = None
    for mp in args.model_path:
        print(f"\nLoading: {mp}")
        e_returns, e_states, e_cell_sat, city = loader(args, mp, cfg)
        print(f"  → {len(e_returns)} policies evaluated, {e_returns.shape[1]} objectives")
        all_returns.append(e_returns)
        all_states.append(e_states)
        all_cell_sat.append(e_cell_sat)

    labels = args.labels or [Path(p).parent.name for p in args.model_path]
    n_obj = all_returns[0].shape[1]

    if "routes" in want:
        plot_routes_on_grid(city, all_states, labels, save("routes.png"))

    if "pareto" in want:
        if n_obj == 2:
            plot_pareto_front(all_returns, labels, save("pareto_front.png"))
        else:
            plot_pareto_front_multi_obj(all_returns, labels, save("pareto_parallel.png"))

    if "rewards" in want:
        plot_group_rewards(all_returns, labels, save("group_rewards.png"))

    if "fairness" in want:
        plot_fairness_summary(all_returns, labels, save("fairness_metrics.png"))

    if "satisfaction" in want and any(len(cs) > 0 for cs in all_cell_sat):
        plot_cell_satisfaction_heatmap(city, all_cell_sat, labels, save("cell_satisfaction.png"))

    print("\nDone.")


if __name__ == "__main__":
    main()
