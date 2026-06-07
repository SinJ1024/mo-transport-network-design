#!/usr/bin/env bash
# =============================================================================
# My part: Spatiotemporal lambda-modulation + spatial fairness metrics
# =============================================================================
# Overnight-friendly: launch ONCE, walk away, wake up to results.
#   - Resumable: each finished run drops a .done marker; re-running the script
#     skips completed runs, so a crash/sleep/Ctrl-C just continues where it left.
#   - Self-saving: every run writes its own log, saves model checkpoints
#     (train_gcn.py -> ./results/gcn_xian_<ts>/), and streams the NEW metrics
#     (served_floor, demand_coverage, spatial_sw_*, price_equity_*,
#      nash_welfare_geom_*) to wandb johnario-tu-delft / project RiDM.
#   - Thread-capped so each run uses ~1 core (macOS Accelerate ignores OMP).
#
# Usage (pick one):
#   PAR=4 bash jobs/exp_my_part.sh
#   nohup PAR=4 bash jobs/exp_my_part.sh > my_part.out 2>&1 &     # survives logout
#
# PAR = how many runs in parallel (default 4). On an 8-core machine that you are
# NOT otherwise using, 4-6 is fine; drop to 2 if you are running other jobs.
#
# Config matrix below = 8 settings x 5 seeds = 40 runs (~12-20h at PAR=4).
# Comment out lines in NAMES/FLAGS to trim.
# =============================================================================
set -u
cd "$(dirname "$0")/.."   # repo root

# --- wandb + single-thread caps (VECLIB is the one macOS numpy actually obeys) -
export WANDB_ENTITY=johnario-tu-delft
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTORCH_NUM_THREADS=1

PAR=${PAR:-4}
SEEDS=(42 123 456 789 1011)
LOGDIR=logs/my_part
mkdir -p "$LOGDIR"

# Shared base config (matches the agreed BASE; nr_groups=10).
BASE="--env=xian --starting_loc_x=9 --starting_loc_y=19 --nr_stations=20 --nr_groups=10 \
--timesteps=30000 --batch_size=256 --hidden_dim=128 --lr=0.01 --max_buffer_size=100 \
--nr_layers=1 --num_er_episodes=100 --num_model_updates=5 --num_step_episodes=10 \
--project_name=RiDM --wandb_entity=johnario-tu-delft"

# --- experiment matrix --------------------------------------------------------
# Fixed-lambda points (l0/l05/l1) anchor the HV / front-coverage comparison;
# spatial / curriculum / spatiotemporal isolate the two mechanisms;
# pareto / nash give cross-criterion reference points for the spatial-metric table.
NAMES=( fixed_l0 fixed_l05 fixed_l1 spatial curriculum spatiotemporal pareto nash )
FLAGS=(
  "--criterion=lorenz --distance_ref=interpolate3 --gcn_lambda=0.0"
  "--criterion=lorenz --distance_ref=interpolate3 --gcn_lambda=0.5"
  "--criterion=lorenz --distance_ref=interpolate3 --gcn_lambda=1.0"
  "--criterion=lorenz --distance_ref=interpolate3 --gcn_lambda=0.0 --spatial_alpha=0.5"
  "--criterion=lorenz --distance_ref=interpolate3 --gcn_lambda=0.0 --lambda_schedule=cosine --lambda_start=1.0 --lambda_end=0.0"
  "--criterion=lorenz --distance_ref=interpolate3 --gcn_lambda=0.0 --spatial_alpha=0.5 --lambda_schedule=cosine --lambda_start=1.0 --lambda_end=0.0"
  "--criterion=pareto"
  "--criterion=nash"
)

# --- one run (resumable + self-saving) ---------------------------------------
run_one() {
  local name="$1" flags="$2" seed="$3"
  local tag="${name}_${seed}"
  local log="$LOGDIR/${tag}.log"
  if [ -f "$LOGDIR/${tag}.done" ]; then
    echo "[skip] $tag (already done)"
    return 0
  fi
  echo "[$(date +%H:%M:%S)] start $tag"
  if python train_gcn.py $BASE $flags --seed="$seed" --experiment_name="GCN-Xian-$name" > "$log" 2>&1; then
    touch "$LOGDIR/${tag}.done"
    echo "[$(date +%H:%M:%S)] done  $tag"
  else
    echo "[$(date +%H:%M:%S)] FAIL  $tag  (see $log)"
  fi
}

# --- launch with bounded concurrency (bash 3.2 batch pool; no wait -n needed) --
echo "Launching up to $(( ${#NAMES[@]} * ${#SEEDS[@]} )) runs, PAR=$PAR"
i=0
for ci in "${!NAMES[@]}"; do
  for s in "${SEEDS[@]}"; do
    run_one "${NAMES[$ci]}" "${FLAGS[$ci]}" "$s" &
    i=$((i + 1))
    if [ $((i % PAR)) -eq 0 ]; then wait; fi
  done
done
wait

# --- summary ------------------------------------------------------------------
echo "==================== SUMMARY ===================="
done_n=$(ls "$LOGDIR"/*.done 2>/dev/null | wc -l | tr -d ' ')
echo "completed: ${done_n} / $(( ${#NAMES[@]} * ${#SEEDS[@]} )) runs"
for ci in "${!NAMES[@]}"; do
  for s in "${SEEDS[@]}"; do
    t="${NAMES[$ci]}_${s}"
    [ -f "$LOGDIR/${t}.done" ] || echo "  MISSING/FAILED: $t  (rerun this script to retry)"
  done
done
echo "logs: $LOGDIR/   |   metrics: wandb johnario-tu-delft / RiDM"
echo "================================================="
