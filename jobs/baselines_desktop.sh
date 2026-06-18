#!/usr/bin/env bash
# Baseline runner for LCN, PCN, and NSW (GCN with Nash criterion).
# Matches the hyperparameter budget used in the GCN ablation study.
# Runs algorithms x groups x seeds in a thread-capped parallel pool; resumable via .done markers.
#
# Usage:
#   ENV=xian  NR_STATIONS=20 ./jobs/baselines_desktop.sh   # Xian
#   ENV=amsterdam NR_STATIONS=10 STARTING_LOC_X=5 STARTING_LOC_Y=9 ./jobs/baselines_desktop.sh
#
# Run both envs sequentially (Xian then Amsterdam):
#   ./jobs/baselines_desktop.sh && \
#   ENV=amsterdam NR_STATIONS=10 STARTING_LOC_X=5 STARTING_LOC_Y=9 ./jobs/baselines_desktop.sh

set -u

# ===================== editable knobs =====================
ENV=${ENV:-xian}
GROUPS_LIST=${GROUPS_LIST:-"3 5 7 10"}
NR_STATIONS=${NR_STATIONS:-20}           # amsterdam: use 10
STARTING_LOC_X=${STARTING_LOC_X:-9}
STARTING_LOC_Y=${STARTING_LOC_Y:-19}
HIDDEN_DIM=${HIDDEN_DIM:-128}
TIMESTEPS=${TIMESTEPS:-30000}
SEEDS=${SEEDS:-"42 123 456 789 1024"}
ALGOS=${ALGOS:-"lcn pcn nsw"}           # which baselines to run
PROJECT=${PROJECT:-cl_ablation}
ENTITY=${ENTITY:-}                       # wandb entity, e.g. johnario-tu-delft; empty = default
PAR=${PAR:-10}                           # concurrent runs
# ==========================================================

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export GCN_N_EVALS=${GCN_N_EVALS:-30} GCN_N_POINTS_PF=${GCN_N_POINTS_PF:-20} WANDB_LOG_FRONT=${WANDB_LOG_FRONT:-0}

LOGDIR=logs/baselines
mkdir -p "$LOGDIR"

# Build the training command for one (algo, G, seed) combination.
# Prints the full command to stdout (no execution).
make_cmd() {
  local ALGO=$1 G=$2 SEED=$3
  local EXPNAME="${ALGO^^}-${ENV}-g${G}"   # e.g. LCN-xian-g5
  local ENTITY_FLAG="${ENTITY:+--wandb_entity=$ENTITY}"

  local BASE_FLAGS="--env=$ENV --nr_groups=$G --nr_stations=$NR_STATIONS \
    --starting_loc_x=$STARTING_LOC_X --starting_loc_y=$STARTING_LOC_Y \
    --timesteps=$TIMESTEPS --batch_size=256 --hidden_dim=$HIDDEN_DIM --lr=0.01 \
    --max_buffer_size=200 --nr_layers=1 --num_er_episodes=100 \
    --num_model_updates=10 --num_step_episodes=10 \
    --project_name=$PROJECT --experiment_name=$EXPNAME $ENTITY_FLAG --seed=$SEED"

  case "$ALGO" in
    lcn)
      echo "python train_lcn.py $BASE_FLAGS --distance_ref=interpolate2 --lcn_lambda=0.0"
      ;;
    pcn)
      echo "python train_pcn.py $BASE_FLAGS"
      ;;
    nsw)
      # NSW = GCN with Nash Social Welfare criterion
      echo "python train_gcn.py $BASE_FLAGS --criterion=nash --distance_ref=interpolate2"
      ;;
    *)
      echo "UNKNOWN_ALGO_$ALGO"
      ;;
  esac
}

echo "ENV=$ENV  GROUPS_LIST=$GROUPS_LIST  NR_STATIONS=$NR_STATIONS  HIDDEN_DIM=$HIDDEN_DIM  PAR=$PAR"
echo "ALGOS=$ALGOS"
echo "SEEDS=$SEEDS"
echo "project=$PROJECT  entity=${ENTITY:-<default>}"
echo

running=0
for ALGO in $ALGOS; do
  for G in $GROUPS_LIST; do
    for SEED in $SEEDS; do
      TAG="${ENV}_${ALGO}_g${G}_s${SEED}"
      if [ -f "$LOGDIR/$TAG.done" ]; then echo "skip  $TAG (already done)"; continue; fi

      CMD=$(make_cmd "$ALGO" "$G" "$SEED")
      case "$CMD" in UNKNOWN_ALGO_*)
        echo "Unknown algo: $ALGO"; exit 1 ;; esac

      echo "start $TAG"
      (
        if $CMD > "$LOGDIR/$TAG.log" 2>&1; then
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
