#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=ablation_xian
#SBATCH --partition=all6000
#SBATCH --account=all6000users
#SBATCH --gres=gpu:1
#SBATCH --mem=62G
#SBATCH --cpus-per-task=12
#SBATCH --time=8:00:00

# Usage: sbatch ablation_xian.sh <seed> <experiment>
# Experiments: baseline_lorenz, baseline_pareto, temporal, spatial, spatiotemporal, context, full
#
# Example (single):
#   sbatch ablation_xian.sh 42 temporal
#
# Example (all experiments, all seeds):
#   for EXP in baseline_lorenz baseline_pareto temporal spatial spatiotemporal context full; do
#     for SEED in 42 123 456 789 1024; do
#       sbatch ablation_xian.sh $SEED $EXP
#     done
#   done

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Error: Missing required arguments"
    echo "Usage: $0 <seed> <experiment>"
    echo "Experiments: baseline_lorenz, baseline_pareto, temporal, spatial, spatiotemporal, context, full"
    exit 1
fi

SEED=$1
EXP=$2

PYTHON="/home/dmichai/anaconda3/envs/mo-nw-design/bin/python"

# Shared hyperparameters (from best xian config with interpolate3)
BASE="$PYTHON train_lcn.py --env=xian --nr_groups=3 --nr_stations=20 \
  --starting_loc_x=9 --starting_loc_y=19 \
  --timesteps=30000 --batch_size=128 --hidden_dim=128 --lr=0.1 \
  --max_buffer_size=50 --nr_layers=1 \
  --num_er_episodes=50 --num_model_updates=5 --num_step_episodes=10 \
  --distance_ref=interpolate3 --seed=$SEED"

case $EXP in
  baseline_lorenz)
    # Pure Lorenz dominance (lambda=0)
    CMD="$BASE --lcn_lambda=0.0"
    ;;
  baseline_pareto)
    # Pure Pareto dominance (lambda=1)
    CMD="$BASE --lcn_lambda=1.0"
    ;;
  temporal)
    # Temporal curriculum only: cosine from 1.0 -> 0.0
    CMD="$BASE --lcn_lambda=0.5 \
      --lambda_schedule=cosine --lambda_start=1.0 --lambda_end=0.0 \
      --lambda_warmup_fraction=0.1 --lambda_freeze_fraction=0.1"
    ;;
  spatial)
    # Spatial lambda only: fixed lambda with spatial modifier
    CMD="$BASE --lcn_lambda=0.5 --spatial_alpha=0.5"
    ;;
  spatiotemporal)
    # Spatiotemporal: cosine schedule + spatial modifier
    CMD="$BASE --lcn_lambda=0.5 \
      --lambda_schedule=cosine --lambda_start=1.0 --lambda_end=0.0 \
      --lambda_warmup_fraction=0.1 --lambda_freeze_fraction=0.1 \
      --spatial_alpha=0.5"
    ;;
  context)
    # Demand context observation only
    CMD="$BASE --lcn_lambda=0.0 --include_demand_context"
    ;;
  full)
    # All innovations combined
    CMD="$BASE --lcn_lambda=0.5 \
      --lambda_schedule=cosine --lambda_start=1.0 --lambda_end=0.0 \
      --lambda_warmup_fraction=0.1 --lambda_freeze_fraction=0.1 \
      --spatial_alpha=0.5 --include_demand_context"
    ;;
  *)
    echo "Error: Unknown experiment '$EXP'"
    echo "Choose from: baseline_lorenz, baseline_pareto, temporal, spatial, spatiotemporal, context, full"
    exit 1
    ;;
esac

echo "Running experiment: $EXP (seed=$SEED)"
echo "Command: $CMD"
$CMD
