#!/usr/bin/env bash
# GCN ablations on Xi'an  ->  wandb entity johnario-tu-delft / project RiDM
#
# Part A (my innovations): isolate spatial-lambda + temporal curriculum
#        criterion=lorenz, distance_ref=interpolate3, base lambda=0
#          baseline       : plain LCN (lambda=0)
#          spatial        : + demand-aware spatial lambda (alpha=0.5)
#          curriculum     : + temporal lambda anneal (cosine 1 -> 0)
#          spatiotemporal : + both
# Part B (GCN generality): pluggable dominance criteria
#          pareto         : Pareto front
#          nash           : Nash-welfare front
#        (lorenz lambda=0 == baseline above, reused)
#
# 6 configs x 5 seeds = 30 runs. Parallelism via PAR (default 4).
# Usage:  PAR=4 bash jobs/ablation_gcn_xian.sh
set -u

cd "$(dirname "$0")/.."   # repo root

export WANDB_ENTITY=johnario-tu-delft
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1   # macOS Accelerate/vecLib (numpy) ignores OMP_NUM_THREADS
export PYTORCH_NUM_THREADS=1

PAR=${PAR:-4}
SEEDS=(42 123 456 789 1011)

BASE="--env=xian --starting_loc_x=9 --starting_loc_y=19 --nr_stations=20 --nr_groups=10 \
--timesteps=30000 --batch_size=256 --hidden_dim=128 --lr=0.01 --max_buffer_size=100 \
--nr_layers=1 --num_er_episodes=100 --num_model_updates=5 --num_step_episodes=10 \
--project_name=RiDM --wandb_entity=johnario-tu-delft"

NAMES=(baseline spatial curriculum spatiotemporal pareto nash)
FLAGS=(
  "--criterion=lorenz --distance_ref=interpolate3 --gcn_lambda=0.0"
  "--criterion=lorenz --distance_ref=interpolate3 --gcn_lambda=0.0 --spatial_alpha=0.5"
  "--criterion=lorenz --distance_ref=interpolate3 --gcn_lambda=0.0 --lambda_schedule=cosine --lambda_start=1.0 --lambda_end=0.0"
  "--criterion=lorenz --distance_ref=interpolate3 --gcn_lambda=0.0 --spatial_alpha=0.5 --lambda_schedule=cosine --lambda_start=1.0 --lambda_end=0.0"
  "--criterion=pareto"
  "--criterion=nash"
)

mkdir -p logs/gcn_ablation
CMDFILE="$(mktemp)"
for i in "${!NAMES[@]}"; do
  name="${NAMES[$i]}"
  flags="${FLAGS[$i]}"
  for s in "${SEEDS[@]}"; do
    echo "python train_gcn.py $BASE $flags --seed=$s --experiment_name=GCN-Xian-$name > logs/gcn_ablation/${name}_${s}.log 2>&1" >> "$CMDFILE"
  done
done

echo "Launching $(wc -l < "$CMDFILE") runs with PAR=$PAR"
cat -n "$CMDFILE"
echo "----------------------------------------------------------------"

# Concurrency via a batch job-pool (works on bash 3.2 -- no `wait -n` needed):
# launch PAR runs at a time, wait for the batch, then launch the next.
i=0
while IFS= read -r cmd; do
  [ -z "$cmd" ] && continue
  i=$((i + 1))
  echo "[$(date +%H:%M:%S)] start run $i"
  bash -c "$cmd" &
  if [ $((i % PAR)) -eq 0 ]; then
    wait
    echo "[$(date +%H:%M:%S)] batch of $PAR finished ($i done)"
  fi
done < "$CMDFILE"
wait
echo "==== ALL GCN ABLATION RUNS DONE ($i runs) ===="
rm -f "$CMDFILE"
