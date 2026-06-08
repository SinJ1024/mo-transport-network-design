#!/usr/bin/env bash
# =============================================================================
# My part: distance_ref study (nondominated vs interpolate2 vs interpolate3)
#          for spatiotemporal lambda-modulation + spatial fairness metrics.
# =============================================================================
# Overnight-friendly: launch ONCE, resumable (.done markers), each run self-saves
# (model checkpoints + log + wandb metrics).
#
# Matrix (15 GCN configs x 5 seeds = 75 runs, + 5 PCN = 80):
#   Baselines : nd (pure Lorenz = LCN), pareto, nash
#   interp2   : l0 l05 l1 spatial curriculum spatiotemporal  (lambda family)
#   interp3   : l0 l05 l1 spatial curriculum spatiotemporal  (lambda family)
#   PCN       : separate algorithm (front-quality metrics only, no cell-level)
#
# NOTE: nondominated IGNORES lambda, so it is only run as the pure-Lorenz
#       baseline; the lambda mechanisms (spatial/curriculum/...) only make sense
#       under interpolate2 / interpolate3.
#
# Est. time on a 12-core CPU at PAR=12: ~3 h for all 80 runs.
#
# Usage:
#   CUDA_VISIBLE_DEVICES="" PAR=12 nohup bash jobs/exp_my_part.sh > my_part.out 2>&1 &
# =============================================================================
set -u
cd "$(dirname "$0")/.."   # repo root

export WANDB_ENTITY=johnario-tu-delft
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTORCH_NUM_THREADS=1

PAR=${PAR:-12}
PROJECT=${PROJECT:-cl_ablation}   # wandb project (separate from Bobi's experiments)
RUN_I3=${RUN_I3:-1}               # set 0 to skip the interpolate3 configs
RUN_PCN=${RUN_PCN:-1}             # set 0 to skip the PCN baseline
SEEDS=(42 123 456 789 1011)
LOGDIR=logs/my_part
mkdir -p "$LOGDIR"

# Shared base (matches the report's B.7 shared config; nr_groups=10).
BASE="--env=xian --starting_loc_x=9 --starting_loc_y=19 --nr_stations=20 --nr_groups=10 \
--timesteps=30000 --batch_size=256 --hidden_dim=128 --lr=0.01 --max_buffer_size=100 \
--nr_layers=1 --num_er_episodes=100 --num_model_updates=5 --num_step_episodes=10 \
--project_name=$PROJECT --wandb_entity=johnario-tu-delft"

# --- GCN configs: baselines + interpolate2 always; interpolate3 optional ------
NAMES=( nd pareto nash i2_l0 i2_l05 i2_l1 i2_spatial i2_curriculum i2_spatiotemporal )
FLAGS=(
  "--criterion=lorenz --distance_ref=nondominated"
  "--criterion=pareto"
  "--criterion=nash"
  "--criterion=lorenz --distance_ref=interpolate2 --gcn_lambda=0.0"
  "--criterion=lorenz --distance_ref=interpolate2 --gcn_lambda=0.5"
  "--criterion=lorenz --distance_ref=interpolate2 --gcn_lambda=1.0"
  "--criterion=lorenz --distance_ref=interpolate2 --gcn_lambda=0.0 --spatial_alpha=0.5"
  "--criterion=lorenz --distance_ref=interpolate2 --gcn_lambda=0.0 --lambda_schedule=cosine --lambda_start=1.0 --lambda_end=0.0"
  "--criterion=lorenz --distance_ref=interpolate2 --gcn_lambda=0.0 --spatial_alpha=0.5 --lambda_schedule=cosine --lambda_start=1.0 --lambda_end=0.0"
)
if [ "$RUN_I3" = "1" ]; then
  NAMES+=( i3_l0 i3_l05 i3_l1 i3_spatial i3_curriculum i3_spatiotemporal )
  FLAGS+=(
    "--criterion=lorenz --distance_ref=interpolate3 --gcn_lambda=0.0"
    "--criterion=lorenz --distance_ref=interpolate3 --gcn_lambda=0.5"
    "--criterion=lorenz --distance_ref=interpolate3 --gcn_lambda=1.0"
    "--criterion=lorenz --distance_ref=interpolate3 --gcn_lambda=0.0 --spatial_alpha=0.5"
    "--criterion=lorenz --distance_ref=interpolate3 --gcn_lambda=0.0 --lambda_schedule=cosine --lambda_start=1.0 --lambda_end=0.0"
    "--criterion=lorenz --distance_ref=interpolate3 --gcn_lambda=0.0 --spatial_alpha=0.5 --lambda_schedule=cosine --lambda_start=1.0 --lambda_end=0.0"
  )
fi

run_gcn() {
  local name="$1" flags="$2" seed="$3" tag="$1_$3" log
  log="$LOGDIR/${name}_${seed}.log"
  if [ -f "$LOGDIR/${name}_${seed}.done" ]; then echo "[skip] ${name}_${seed}"; return 0; fi
  echo "[$(date +%H:%M:%S)] start ${name}_${seed}"
  if python train_gcn.py $BASE $flags --seed="$seed" --experiment_name="GCN-Xian-$name" > "$log" 2>&1; then
    touch "$LOGDIR/${name}_${seed}.done"; echo "[$(date +%H:%M:%S)] done  ${name}_${seed}"
  else
    echo "[$(date +%H:%M:%S)] FAIL  ${name}_${seed}  (see $log)"
  fi
}

# --- PCN baseline (separate script; front-quality metrics only, NO cell-level) -
PCN_BASE="--env=xian --nr_groups=10 --nr_stations=20 --starting_loc_x=9 --starting_loc_y=19 \
--timesteps=30000 --batch_size=256 --hidden_dim=128 --lr=0.01 --max_buffer_size=100 \
--nr_layers=1 --num_er_episodes=100 --num_step_episodes=10 --num_model_updates=5 \
--num_policies=10 --project_name=$PROJECT"
run_pcn() {
  local seed="$1" log="$LOGDIR/pcn_${seed}.log"
  if [ -f "$LOGDIR/pcn_${seed}.done" ]; then echo "[skip] pcn_${seed}"; return 0; fi
  echo "[$(date +%H:%M:%S)] start pcn_${seed}"
  if python train_pcn.py $PCN_BASE --seed="$seed" > "$log" 2>&1; then
    touch "$LOGDIR/pcn_${seed}.done"; echo "[$(date +%H:%M:%S)] done  pcn_${seed}"
  else
    echo "[$(date +%H:%M:%S)] FAIL  pcn_${seed}  (see $log)"
  fi
}

# --- launch all jobs through one bounded pool (PAR at a time) ------------------
echo "Launching $(( ${#NAMES[@]} * ${#SEEDS[@]} + ${#SEEDS[@]} )) runs (15 GCN configs + PCN) x ${#SEEDS[@]} seeds, PAR=$PAR"
i=0
for ci in "${!NAMES[@]}"; do
  for s in "${SEEDS[@]}"; do
    run_gcn "${NAMES[$ci]}" "${FLAGS[$ci]}" "$s" &
    i=$((i + 1)); [ $((i % PAR)) -eq 0 ] && wait
  done
done
# PCN last (lowest priority; skip with RUN_PCN=0)
if [ "$RUN_PCN" = "1" ]; then
  for s in "${SEEDS[@]}"; do
    run_pcn "$s" &
    i=$((i + 1)); [ $((i % PAR)) -eq 0 ] && wait
  done
fi
wait

# --- summary ------------------------------------------------------------------
echo "==================== SUMMARY ===================="
total=$(( ${#NAMES[@]} * ${#SEEDS[@]} + ${#SEEDS[@]} ))
done_n=$(ls "$LOGDIR"/*.done 2>/dev/null | wc -l | tr -d ' ')
echo "completed: ${done_n} / ${total} runs"
for ci in "${!NAMES[@]}"; do for s in "${SEEDS[@]}"; do
  t="${NAMES[$ci]}_${s}"; [ -f "$LOGDIR/${t}.done" ] || echo "  MISSING/FAILED: $t"
done; done
for s in "${SEEDS[@]}"; do [ -f "$LOGDIR/pcn_${s}.done" ] || echo "  MISSING/FAILED: pcn_${s}"; done
echo "logs: $LOGDIR/   |   metrics: wandb johnario-tu-delft / RiDM"
echo "================================================="
