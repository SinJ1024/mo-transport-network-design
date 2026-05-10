#!/bin/bash
# Launch all ablation experiments across seeds.
#
# Usage:
#   ./jobs/run_all_ablations.sh              # submit all to SLURM
#   ./jobs/run_all_ablations.sh --local      # run locally (sequential, for testing)
#   ./jobs/run_all_ablations.sh --dry-run    # print commands without executing

SEEDS=(42 123 456 789 1024)
EXPERIMENTS=(baseline_lorenz baseline_pareto temporal spatial spatiotemporal context full)
ENVS=(xian amsterdam)

MODE="slurm"
if [ "$1" = "--local" ]; then
    MODE="local"
elif [ "$1" = "--dry-run" ]; then
    MODE="dry"
fi

TOTAL=0
for ENV in "${ENVS[@]}"; do
    SCRIPT="jobs/ablation_${ENV}.sh"
    for EXP in "${EXPERIMENTS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            TOTAL=$((TOTAL + 1))
            if [ "$MODE" = "slurm" ]; then
                echo "[$TOTAL] sbatch $SCRIPT $SEED $EXP"
                sbatch "$SCRIPT" "$SEED" "$EXP"
            elif [ "$MODE" = "local" ]; then
                echo "[$TOTAL] bash $SCRIPT $SEED $EXP"
                bash "$SCRIPT" "$SEED" "$EXP"
            else
                echo "[$TOTAL] [dry-run] sbatch $SCRIPT $SEED $EXP"
            fi
        done
    done
done

echo ""
echo "Total jobs: $TOTAL"
