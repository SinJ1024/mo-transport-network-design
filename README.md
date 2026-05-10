# mo-transport-network-design
We introduce Lorenz Conditioned Networks (LCN), a novel multi-policy algorithm for addressing fairness in Multi-Objective Reinforcement Learning (MORL). Based on Lorenz optimality, LCN learns policies that ensure a fair distribution of rewards among different objectives. We extend LCN to introduce $\lambda$-LCN, based on a relaxation of Lorenz optimality that offers flexibility in determining fairness preferences. Finally, we address the lack of real-world MORL benchmarks, by introducing a large-scale, multi-objective environment for real-world transportation network design. Experiments in Xi'an and Amsterdam demonstrate LCN's ability to learn fair policies and scalability in high-dimensional state-action and reward spaces.


# Setup
Create the conda environment:
```
conda env create -f environment.yml
```

Install the mo-tndp environment:
```
git submodule update --init --recursive
cd envs/mo-tndp
pip install -e .
```

Install the morl-baselines repo:
```
cd morl-baselines
pip install -e .
```

Install the [deep-sea-treasure](https://github.com/imec-idlab/deep-sea-treasure) environment:
```
python3 -m pip install  deep_sea_treasure
```

To run GPI-PD: Install [pycddlib](https://pycddlib.readthedocs.io/en/latest/quickstart.html#installation)
First install gmp:
```
brew install gmp
```
Then, to install the package using pip:
```
CFLAGS=-I`brew --prefix gmp`/include LDFLAGS=-L`brew --prefix gmp`/lib pip install pip install pycddlib
```

On Mac M1, if you get an error while installing the morl_baselines, do `pip install osqp==0.6.1` and try again ([source](https://stackoverflow.com/questions/65920955/failed-building-wheel-for-qdldl-when-installing-cvxpy))

On a linux cluster without sudo permissions, use the following istructions.
Firstly, cd to your home directory.
```
wget https://ftp.gnu.org/gnu/gmp/gmp-6.3.0.tar.bz2
tar -xjf gmp-6.3.0.tar.bz2
cd gmp-6.3.0/
./configure --prefix=/home/YOUR_USER_NAME/opt/
```
Then, configure the lib and include paths:
```
export LD_LIBRARY_PATH=/home/YOUR_USER_NAME/opt/lib:$LD_LIBRARY_PATH
export C_INCLUDE_PATH=/home/YOUR_USER_NAME/opt/include:$C_INCLUDE_PATH
```

And finally, install pycddlib:
```
CFLAGS=-I/home/dmichai/opt/include LDFLAGS=-L/home/dmichai/opt/lib pip install pycddlib
```

# Reproducing the Experiments
All commands to reproduce the experiments can be found [here](https://aware-night-ab1.notion.site/Project-B-MO-LCN-Experiment-Tracker-b4d21ab160eb458a9cff9ab9314606a7)

## Ablation Experiments (Spatiotemporal Lambda + Demand Context)

We extend the original LCN with three innovations:
1. **Spatiotemporal Lambda Curriculum** — transitions lambda from Pareto exploration to Lorenz fairness over training, with a spatial modifier based on per-route demand context.
2. **OD Demand Context Observation** — augments the agent's observation with normalized aggregated OD demand.
3. **New Fairness Metrics** — Max-Min Satisfaction Floor (Rawlsian) and Spatial Sen Welfare (evaluation only).

### New CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--lambda_schedule` | `constant` | Temporal schedule: `constant`, `linear`, `cosine`, `step` |
| `--lambda_start` | `1.0` | Initial lambda (Pareto exploration) |
| `--lambda_end` | `None` | Target lambda (defaults to `--lcn_lambda`) |
| `--lambda_warmup_fraction` | `0.0` | Fraction of training to keep lambda at start value |
| `--lambda_freeze_fraction` | `0.1` | Fraction of training at end to freeze lambda |
| `--spatial_alpha` | `0.0` | Spatial scaling for per-episode effective lambda (0 = disabled) |
| `--include_demand_context` | `False` | Augment observation with normalized OD demand vector |

### Quick Smoke Test (dilemma, ~2 min)

```bash
# Baseline (original behavior, no new flags)
python train_lcn.py --env dilemma --nr_stations 9 --timesteps 2000 \
  --lcn_lambda 0.5 --distance_ref interpolate2 --seed 42

# All innovations enabled
python train_lcn.py --env dilemma --nr_stations 9 --timesteps 2000 \
  --lcn_lambda 0.5 --distance_ref interpolate2 --seed 42 \
  --lambda_schedule cosine --lambda_start 1.0 --lambda_end 0.0 \
  --spatial_alpha 0.5 --include_demand_context
```

### Ablation Experiment Matrix

| Experiment | lambda | schedule | spatial_alpha | demand_context | Purpose |
|-----------|--------|----------|---------------|----------------|---------|
| `baseline_lorenz` | 0.0 | constant | 0 | No | Pure Lorenz baseline |
| `baseline_pareto` | 1.0 | constant | 0 | No | Pure Pareto baseline |
| `temporal` | 0.5 | cosine 1.0->0.0 | 0 | No | Temporal curriculum only |
| `spatial` | 0.5 | constant | 0.5 | No | Spatial lambda only |
| `spatiotemporal` | 0.5 | cosine 1.0->0.0 | 0.5 | No | Combined lambda |
| `context` | 0.0 | constant | 0 | Yes | Demand observation only |
| `full` | 0.5 | cosine 1.0->0.0 | 0.5 | Yes | All innovations |

### Running on SLURM Cluster

Single experiment:
```bash
sbatch jobs/ablation_xian.sh 42 temporal
sbatch jobs/ablation_amsterdam.sh 42 spatiotemporal
```

All experiments across 5 seeds:
```bash
for EXP in baseline_lorenz baseline_pareto temporal spatial spatiotemporal context full; do
  for SEED in 42 123 456 789 1024; do
    sbatch jobs/ablation_xian.sh $SEED $EXP
    sbatch jobs/ablation_amsterdam.sh $SEED $EXP
  done
done
```

Or use the convenience script:
```bash
./jobs/run_all_ablations.sh            # submit all to SLURM (70 jobs total)
./jobs/run_all_ablations.sh --dry-run  # preview commands without submitting
```

### Running Locally

```bash
# Single experiment
bash jobs/ablation_xian.sh 42 temporal

# All experiments sequentially
./jobs/run_all_ablations.sh --local
```

### Key Metrics to Compare (wandb)

| Metric | Key | Meaning |
|--------|-----|---------|
| Hypervolume | `eval/hypervolume` | Pareto front quality |
| Sen Welfare | `eval/sen_welfare_max` | Efficiency x Equality |
| Gini | `eval/gini_min` | Inequality (lower = fairer) |
| Max-Min Floor | `eval/maxmin_floor_max` | Worst-case cell satisfaction |
| Spatial SW Ratio | `eval/spatial_sw_ratio` | High/low demand region balance |
| Lambda Curve | `train/lcn_lambda` | Lambda schedule progression |
