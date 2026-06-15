#!/usr/bin/env bash
# GCN lambda-scheduler ablation, tuned for a single workstation (no SLURM).
# Runs groups x configs x seeds in a thread-capped parallel pool, resumable via .done markers.
#
# Usage:
#   ./jobs/gcn_ablation_desktop.sh                          # run with the knobs below
#   PAR=12 GROUPS_LIST="3 5" ./jobs/gcn_ablation_desktop.sh  # override per-run
#
# Each run is one `python train_gcn.py ...`. Logs go to logs/gcn_ablation/.

set -u

# ===================== editable knobs =====================
ENV=${ENV:-xian}
GROUPS_LIST=${GROUPS_LIST:-"3 5 7 10"}    # nr_groups values to sweep
NR_STATIONS=${NR_STATIONS:-20}            # amsterdam: use 10
STARTING_LOC_X=${STARTING_LOC_X:-9}
STARTING_LOC_Y=${STARTING_LOC_Y:-19}
HIDDEN_DIM=${HIDDEN_DIM:-128}
TIMESTEPS=${TIMESTEPS:-30000}
SEEDS=${SEEDS:-"42 123 456 789 1024"}
CONFIGS=${CONFIGS:-"l0 temporal spatial spatiotemporal"}   # add: pareto context full
PROJECT=${PROJECT:-cl_ablation}
ENTITY=${ENTITY:-}                 # set to your wandb entity/team, e.g. johnario-tu-delft; empty = default
PAR=${PAR:-10}                     # concurrent runs (7900X has 16 cores; each run is thread-capped to 1)
# ==========================================================

# Keep numpy/BLAS single-threaded so PAR parallel runs do not oversubscribe cores
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
# Eval-cost knobs (read by gcn.py / evaluation.py): fewer/cheaper evals, no front-table upload
export GCN_N_EVALS=${GCN_N_EVALS:-30} GCN_N_POINTS_PF=${GCN_N_POINTS_PF:-20} WANDB_LOG_FRONT=${WANDB_LOG_FRONT:-0}

LOGDIR=logs/gcn_ablation
mkdir -p "$LOGDIR"

base_cmd() {
  local CFG=$1 G=$2
  echo "python train_gcn.py --env=$ENV --nr_groups=$G --nr_stations=$NR_STATIONS \
    --starting_loc_x=$STARTING_LOC_X --starting_loc_y=$STARTING_LOC_Y --timesteps=$TIMESTEPS \
    --batch_size=256 --hidden_dim=$HIDDEN_DIM --lr=0.01 --max_buffer_size=200 \
    --nr_layers=1 --num_er_episodes=100 --num_model_updates=10 --num_step_episodes=10 \
    --criterion=lorenz --distance_ref=interpolate2 \
    --project_name=$PROJECT --experiment_name=GCN-${ENV}-g${G}-${CFG} ${ENTITY:+--wandb_entity=$ENTITY}"
}

# Per-config lambda-scheduler flags. l0 = fixed Lorenz baseline (lambda=0, no schedule, no spatial).
cfg_flags() {
  case "$1" in
    l0)             echo "--gcn_lambda=0.0" ;;
    pareto)         echo "--gcn_lambda=1.0" ;;
    temporal)       echo "--gcn_lambda=0.0 --lambda_schedule=cosine --lambda_start=1.0 --lambda_end=0.0 --lambda_warmup_fraction=0.1 --lambda_freeze_fraction=0.1" ;;
    spatial)        echo "--gcn_lambda=0.0 --spatial_alpha=0.5" ;;
    spatiotemporal) echo "--gcn_lambda=0.0 --lambda_schedule=cosine --lambda_start=1.0 --lambda_end=0.0 --lambda_warmup_fraction=0.1 --lambda_freeze_fraction=0.1 --spatial_alpha=0.5" ;;
    context)        echo "--gcn_lambda=0.0 --include_demand_context" ;;
    full)           echo "--gcn_lambda=0.0 --lambda_schedule=cosine --lambda_start=1.0 --lambda_end=0.0 --lambda_warmup_fraction=0.1 --lambda_freeze_fraction=0.1 --spatial_alpha=0.5 --include_demand_context" ;;
    *) echo "BAD_CONFIG_$1" ;;
  esac
}

echo "ENV=$ENV  GROUPS_LIST=$GROUPS_LIST  NR_STATIONS=$NR_STATIONS  HIDDEN_DIM=$HIDDEN_DIM  PAR=$PAR"
echo "CONFIGS=$CONFIGS"
echo "SEEDS=$SEEDS"
echo "project=$PROJECT  entity=${ENTITY:-<default>}"
echo

running=0
for G in $GROUPS_LIST; do
  for CFG in $CONFIGS; do
    FLAGS=$(cfg_flags "$CFG")
    case "$FLAGS" in BAD_CONFIG_*) echo "Unknown config: $CFG"; exit 1 ;; esac
    for SEED in $SEEDS; do
      TAG="${ENV}_g${G}_${CFG}_s${SEED}"
      if [ -f "$LOGDIR/$TAG.done" ]; then echo "skip  $TAG (already done)"; continue; fi
      echo "start $TAG"
      (
        if $(base_cmd "$CFG" "$G") $FLAGS --seed="$SEED" > "$LOGDIR/$TAG.log" 2>&1; then
          touch "$LOGDIR/$TAG.done"; echo "OK    $TAG"
        else
          echo "FAIL  $TAG  (see $LOGDIR/$TAG.log)"
        fi
      ) &
      running=$((running + 1))
      if [ "$running" -ge "$PAR" ]; then
        wait -n 2>/dev/null || wait
        running=$((running - 1))
      fi
    done
  done
done
wait
echo
echo "All runs finished. Logs: $LOGDIR/  (re-run this script to resume any FAIL/missing)."
