#!/bin/bash

set -u

source .venv/bin/activate

ENV=${ENV:-xian}
NR_GROUPS=${NR_GROUPS:-5}          # smaller G = faster + avoids HV=0 degeneracy (10 was the problem)
NR_STATIONS=${NR_STATIONS:-20}
HIDDEN_DIM=${HIDDEN_DIM:-64}       # smaller net = faster (was 128)
TIMESTEPS=${TIMESTEPS:-30000}
SEEDS=${SEEDS:-"42 123 456 789 1024"}
CONFIGS=${CONFIGS:-"sized filter"}   # add: pareto context full
PROJECT=${PROJECT:-cl_ablation}
ENTITY=${ENTITY:-johnario-tu-delft}                 # set to your wandb entity/team, e.g. johnario-tu-delft; empty = default
PAR=${PAR:-10}

export GCN_N_EVALS=${GCN_N_EVALS:-30} GCN_N_POINTS_PF=${GCN_N_POINTS_PF:-20} WANDB_LOG_FRONT=${WANDB_LOG_FRONT:-0}

LOGDIR=logs/gcn_nash
mkdir -p "$LOGDIR"

base_cmd() {
	echo "python train_gcn.py \
	--env=$ENV \
	--nr_groups=$NR_GROUPS \
	--nr_stations=$NR_STATIONS \
	--starting_loc_x=9 \
	--starting_loc_y=9 \
	--timesteps=$TIMESTEPS \
	--batch_size=256 \
	--hidden_dim=$HIDDEN_DIM \
	--lr=0.01 \
	--max_buffer_size=100 \
	--nr_layers=1 \
	--num_er_episodes=100 \
	--num_model_updates=5 \
	--num_step_episodes=10 \
	--criterion=nash \
	--project_name=$PROJECT \
	--experiment_name=GCN-${ENV}-${1} ${ENTITY:+--wandb_entity=$ENTITY}"
}

cfg_flags() {
	case "$1" in
		sized)		echo "--nash_mode=pareto_sized" ;;
		filter)		echo "--nash_mode=pareto_filter --nash_top_k=30" ;;
		*)			echo "BAD_CONFIG_$1" ;;
	esac
}

echo "ENV=$ENV  NR_GROUPS=$NR_GROUPS  HIDDEN_DIM=$HIDDEN_DIM  PAR=$PAR"
echo "CONFIGS=$CONFIGS"
echo "SEEDS=$SEEDS"
echo "project=$PROJECT  entity=${ENTITY:-<default>}"
echo

running=0
for CFG in $CONFIGS; do
  FLAGS=$(cfg_flags "$CFG")
  case "$FLAGS" in BAD_CONFIG_*) echo "Unknown config: $CFG"; exit 1 ;; esac
  for SEED in $SEEDS; do
    TAG="${ENV}_g${NR_GROUPS}_${CFG}_s${SEED}"
    if [ -f "$LOGDIR/$TAG.done" ]; then echo "skip  $TAG (already done)"; continue; fi
    echo "start $TAG"
    (
      if $(base_cmd "$CFG") $FLAGS --seed="$SEED" > "$LOGDIR/$TAG.log" 2>&1; then
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
wait
echo
echo "All runs finished. Logs: $LOGDIR/  (re-run this script to resume any FAIL/missing)."
