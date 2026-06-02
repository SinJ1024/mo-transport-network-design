import datetime
from pathlib import Path
import random
import mo_gymnasium as mo_gym
from motndp.city import City
from motndp.constraints import MetroConstraints
import numpy as np
import torch
import envs
import argparse
from morl_baselines.multi_policy.gcn.gcn import GCN
from morl_baselines.multi_policy.gcn.gcn_model_classes import DefaultGCNModel
from morl_baselines.multi_policy.gcn.fairness_funcs import (
    get_non_pareto_dominated, get_nash_dominated, 
    pareto_l2, lorenz_l2, nash_l2
)
from morl_baselines.multi_policy.gcn.hyperparam_scheduler import HyperparamScheduler

def main(args):
    def make_env(gym_env):
        if gym_env == 'deep-sea-treasure-concave-v0':
            return mo_gym.make(gym_env)
 
        city = City(
            args.city_path,
            groups_file=args.groups_file,
            ignore_existing_lines=args.ignore_existing_lines
        )
 
        env = mo_gym.make(args.gym_env,
                        city=city,
                        constraints=MetroConstraints(city),
                        nr_stations=args.nr_stations,
                        chained_reward=True,
                        include_demand_context=args.include_demand_context,)

        return env

    env = make_env(args.gym_env)
 
    agent = GCN(
        env,
        scaling_factor=args.scaling_factor,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        project_name=args.project_name,
        experiment_name=args.experiment_name,
        log=not args.no_log,
        seed=args.seed,
        nr_layers=args.nr_layers,
        hidden_dim=args.hidden_dim,
        dominance_func=args.dominance_func,
        l2_func=args.l2_func,
        l2_params=args.l2_params,
        hyperparam_scheduler=args.scheduler,
        cd_threshold=args.cd_threshold,
        model_class=DefaultGCNModel
    )
 
    if args.starting_loc is None:
        print('NOTE: Training is running with random starting locations.')

    save_dir = Path(f"./results/gcn_{args.env}_{datetime.datetime.today().strftime('%Y%m%d_%H_%M_%S.%f')}")
    agent.train(
        total_timesteps=args.timesteps,
        eval_env=make_env(args.gym_env),
        ref_point=args.ref_point,
        num_er_episodes=args.num_er_episodes,
        num_step_episodes=args.num_step_episodes,
        max_buffer_size=args.max_buffer_size,
        num_model_updates=args.num_model_updates,
        starting_loc=args.starting_loc,
        nr_stations=args.nr_stations,
        max_return=args.max_return,
        save_dir=save_dir,
        pf_plot_limits=args.pf_plot_limits,
        n_policies=args.num_policies,

        # known_pareto_front=env.unwrapped.pareto_front(gamma=1.0),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MO GCN - TNDP")
    # Acceptable values: 'dilemma', 'margins', 'amsterdam', 'dst'
    parser.add_argument('--env', default='xian', type=str)
    # For amsterdam environment we have different groups files (different nr of objectives)
    parser.add_argument('--nr_groups', default=5, type=int)
    # Starting location of the agent
    parser.add_argument('--starting_loc_x', default=None, type=int)
    parser.add_argument('--starting_loc_y', default=None, type=int)
    # Episode horizon -- used as a proxy of both the budget and the number of stations (stations are not really costed)
    parser.add_argument('--nr_stations', type=int, required=True)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--num_er_episodes', default=50, type=int)
    parser.add_argument('--num_step_episodes', default=10, type=int)
    parser.add_argument('--num_model_updates', default=10, type=int)
    parser.add_argument('--num_policies', default=10, type=int)
    parser.add_argument('--max_buffer_size', default=50, type=int)
    parser.add_argument('--nr_layers', default=1, type=int)
    parser.add_argument('--hidden_dim', default=64, type=int)
    parser.add_argument('--timesteps', default=2000, type=int)
    parser.add_argument('--no_log', action='store_true', default=False)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--cd_threshold', default=0.2, type=float, help='controls the threshold for crowdedness distance.')
    parser.add_argument('--distance_ref', default='nondominated', type=str, choices=['nondominated', 'optimal_max', 'nondominated_mean', 'interpolate', 'interpolate2', 'interpolate3'], help='controls the reference point for calculating the distance of every solution to the optimal point.')
    parser.add_argument('--gcn_lambda', default=None, type=float, help='value between 0 and 1. Controls the size of the front to explore. lambda -> 1: full pareto front. lambda -> 0 full lorenz front.')
    parser.add_argument('--lambda_schedule', default='constant', type=str, choices=['constant', 'linear', 'cosine', 'step'], help='temporal schedule for lambda curriculum.')
    parser.add_argument('--lambda_start', default=1.0, type=float, help='initial lambda value for curriculum.')
    parser.add_argument('--lambda_end', default=None, type=float, help='target lambda value for curriculum (defaults to --gcn_lambda).')
    parser.add_argument('--lambda_warmup_fraction', default=0.0, type=float, help='fraction of training to keep lambda at lambda_start.')
    parser.add_argument('--lambda_freeze_fraction', default=0.1, type=float, help='fraction of training at end to keep lambda at lambda_end.')
    parser.add_argument('--spatial_alpha', default=0.0, type=float, help='spatial scaling factor for per-episode effective lambda. 0 disables spatial component.')
    parser.add_argument('--include_demand_context', action='store_true', default=False, help='augment observation with normalized OD demand context.')
    parser.add_argument('--criterion', default='lorenz',  choices=['pareto', 'lorenz', 'nash'], type=str)
    # NSW params:
    parser.add_argument('--nash_mode', default='pareto_sized', choices=['pareto_filter', 'pareto_sized'])
    parser.add_argument('--nash_top_k', default=None, type=int)
    parser.add_argument('--nash_shift', default=0.0, type=float)

    args = parser.parse_args()
    print(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.criterion == 'pareto':
        args.dominance_func, args.l2_func = get_non_pareto_dominated, pareto_l2
        args.l2_params = {}
    elif args.criterion == "lorenz":
        args.dominance_func, args.l2_func = get_non_pareto_dominated, lorenz_l2
        args.l2_params = {
            'distance_ref': args.distance_ref,
            'lcn_lambda': args.gcn_lambda,
            'spatial_alpha': args.spatial_alpha
        }
    elif args.criterion == 'nash':
        args.dominance_func, args.l2_func = get_nash_dominated, nash_l2
        args.l2_params = {
            'mode': args.nash_mode,
            'shift': args.nash_shift
        }
        if args.nash_top_k is not None:
            l2_params['top_k'] = args.nash_top_k

    args.scheduler = None
    if args.lambda_schedule != 'constant':
        args.scheduler = HyperparamScheduler(
            schedule_type=args.lambda_schedule,
            target_key='lcn_lambda',
            start_val=args.lambda_start,
            end_val=args.lambda_end if args.lambda_end is not None else args.gcn_lambda,
            total_timesteps=args.timesteps,
            warmup_fraction=args.lambda_warmup_fraction,
            freeze_fraction=args.lambda_freeze_fraction
        )

    # Some values are hardcoded for each environment (this is flexible, but we don't want to have to pass 100 arguments to the script)
    if args.env == 'amsterdam':
        args.city_path = Path(f"./envs/mo-tndp/cities/amsterdam")
        args.gym_env = 'motndp_amsterdam-v0'
        args.project_name = "MORL-TNDP"
        args.groups_file = f"price_groups_{args.nr_groups}.txt"
        args.ignore_existing_lines = True
        args.experiment_name = "GCN-Amsterdam"
        args.scaling_factor = np.array([100] * args.nr_groups + [0.01])
        args.ref_point = np.array([0] * args.nr_groups)
        args.max_return=np.array([1] * args.nr_groups)
        args.pf_plot_limits = None
    elif args.env == 'xian':
        args.city_path = Path(f"./envs/mo-tndp/cities/xian")
        args.gym_env = 'motndp_xian-v0'
        args.project_name = "MORL-TNDP"
        args.groups_file = f"price_groups_{args.nr_groups}.txt"
        args.ignore_existing_lines = True
        args.experiment_name = "GCN-Xian"
        args.scaling_factor = np.array([100] * args.nr_groups + [0.01])
        args.ref_point = np.array([0] * args.nr_groups)
        args.max_return=np.array([1] * args.nr_groups)
        args.pf_plot_limits = None

    if args.starting_loc_x is not None and args.starting_loc_y is not None:
        args.starting_loc = (args.starting_loc_x, args.starting_loc_y)
    else:
        args.starting_loc = None

    main(args)
