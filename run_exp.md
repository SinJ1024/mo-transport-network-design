# Experiment Guide: Spatiotemporal LCN for MO-TNDP

本文档涵盖环境安装、实验运行方法、参数调整说明，以及实验结果的评估方式。

---

## 1. 环境安装

### 1.1 创建 Conda 环境

```bash
conda env create -f environment.yml
conda activate mo-nw-design
```

> `environment.yml` 基于 Python 3.11.3，包含 gymnasium、matplotlib、cvxpy 等基础依赖。

### 1.2 安装项目子模块

```bash
git submodule update --init --recursive
```

### 1.3 安装 mo-tndp 环境

```bash
cd envs/mo-tndp
pip install -e .
cd ../..
```

### 1.4 安装 morl-baselines

```bash
cd morl-baselines
pip install -e .
cd ..
```

### 1.5 安装其他依赖

```bash
pip install mo-gymnasium
pip install deep_sea_treasure   # 仅 DST 实验需要
```

**Mac M1 用户**：如果安装 morl-baselines 出错，先执行 `pip install osqp==0.6.1` 再重试。

### 1.6 验证安装

```bash
python -c "import morl_baselines; import motndp; import mo_gymnasium; print('All packages OK')"
```

---

## 2. 快速开始

### 2.1 Smoke Test（dilemma 环境，约 2 分钟）

```bash
# 原始基线（不使用任何新功能）
python train_lcn.py --env dilemma --nr_stations 9 --timesteps 2000 \
  --lcn_lambda 0.5 --distance_ref interpolate2 --seed 42 --no_log

# 启用全部新功能
python train_lcn.py --env dilemma --nr_stations 9 --timesteps 2000 \
  --lcn_lambda 0.5 --distance_ref interpolate2 --seed 42 --no_log \
  --lambda_schedule cosine --lambda_start 1.0 --lambda_end 0.0 \
  --spatial_alpha 0.5 --include_demand_context
```

如果两条命令都能跑完并输出 training step 信息，说明安装成功。

### 2.2 带 wandb 日志的运行

去掉 `--no_log` 即可自动记录到 wandb：

```bash
python train_lcn.py --env xian --nr_groups 3 --nr_stations 20 \
  --starting_loc_x 9 --starting_loc_y 19 \
  --timesteps 30000 --batch_size 128 --hidden_dim 128 --lr 0.1 \
  --distance_ref interpolate3 --lcn_lambda 0.5 \
  --lambda_schedule cosine --lambda_start 1.0 --lambda_end 0.0 \
  --seed 42
```

---

## 3. 全部参数说明

### 3.1 环境与任务参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--env` | `dilemma` | 环境名称：`dilemma`, `margins`, `xian`, `amsterdam`, `dst` |
| `--nr_groups` | `5` | 社会经济分组数（amsterdam/xian 有效，决定目标维度） |
| `--nr_stations` | **必填** | 每条线路放置的站点数（即 episode 步数） |
| `--starting_loc_x` | `None` | 起始位置 x 坐标（None = 随机） |
| `--starting_loc_y` | `None` | 起始位置 y 坐标（None = 随机） |

**各环境预设值**（在 `train_lcn.py` 中硬编码，无需手动设置）：

| 环境 | grid 大小 | 推荐 nr_stations | 推荐 starting_loc |
|------|-----------|------------------|-------------------|
| dilemma | 5x5 (25 cells) | 9 | 随机 |
| margins | 5x5 (25 cells) | 9 | 随机 |
| xian | 29x29 (841 cells) | 20 | (9, 19) |
| amsterdam | 35x47 (1645 cells) | 10 | (9, 19) |

> **调整建议**：`nr_stations` 是控制线路长度的核心参数。值越大，线路越长，搜索空间越大。对于新城市环境，建议从 grid 对角线长度的 30%-50% 开始尝试。

### 3.2 训练超参数

| 参数 | 默认值 | 说明 | 调整建议 |
|------|--------|------|----------|
| `--timesteps` | `2000` | 总训练步数 | xian/amsterdam 建议 30000+ |
| `--lr` | `0.01` | 学习率 | xian 用 0.1 效果更好 |
| `--batch_size` | `256` | 批大小 | 128 或 256 |
| `--hidden_dim` | `64` | 隐藏层维度 | 大环境（amsterdam）建议 128 |
| `--nr_layers` | `1` | 网络层数 | 1-2 层，更多层未见显著提升 |
| `--num_er_episodes` | `50` | 初始随机填充 ER buffer 的 episode 数 | 50-100 |
| `--num_step_episodes` | `10` | 每轮训练的采样 episode 数 | 10 |
| `--num_model_updates` | `10` | 每轮训练的模型更新次数 | 5-10 |
| `--max_buffer_size` | `50` | ER buffer 最大容量 | 50-100 |
| `--num_policies` | `10` | 评估时生成的策略数 | 10 |
| `--seed` | `42` | 随机种子 | 消融实验用 42, 123, 456, 789, 1024 |

> **调整位置**：这些参数在 `train_lcn.py` 的 argparse 部分定义（第 98-131 行）。

### 3.3 LCN 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--distance_ref` | `nondominated` | ER buffer 距离计算方式 |
| `--lcn_lambda` | `None` | lambda 值，控制 Pareto vs Lorenz 平衡 |
| `--cd_threshold` | `0.2` | 拥挤距离阈值 |

**`distance_ref` 选项详解**：

| 选项 | 含义 | 使用场景 |
|------|------|----------|
| `nondominated` | 到非支配前沿的距离 | 基础 LCN |
| `optimal_max` | 到最优点的距离 | |
| `nondominated_mean` | 到非支配前沿均值的距离 | |
| `interpolate` | Gini 过滤 + 非支配距离 | 需要设置 `lcn_lambda` |
| `interpolate2` | lambda 加权混合 Pareto 和 Lorenz 向量 | 推荐，需要设置 `lcn_lambda` |
| `interpolate3` | 类似 interpolate2 但归一化方式不同 | 推荐用于 xian/amsterdam |

**`lcn_lambda` 含义**：

- `lambda = 1.0`：纯 Pareto 支配（最大化效率，不考虑公平）
- `lambda = 0.0`：纯 Lorenz 支配（最大化公平性）
- `lambda = 0.5`：Pareto 和 Lorenz 的平衡
- `lambda = None`：不使用 lambda-Lorenz 机制（普通 LCN）

> **调整位置**：`lcn_lambda` 在 `morl_baselines/multi_policy/lcn/lcn.py` 的 `_nlargest()` 方法中使用（约第 330-420 行），控制 ER buffer 中 episode 的排序方式。

---

## 4. 新增功能参数（三项创新）

### 4.1 创新一：时空 Lambda 课程学习

控制 lambda 从 Pareto 探索（高 lambda）逐步过渡到 Lorenz 公平（低 lambda）。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lambda_schedule` | `constant` | 时间调度类型：`constant`, `linear`, `cosine`, `step` |
| `--lambda_start` | `1.0` | 课程开始时的 lambda 值 |
| `--lambda_end` | `None` | 课程结束时的 lambda 值（默认等于 `--lcn_lambda`） |
| `--lambda_warmup_fraction` | `0.0` | 训练初期保持 lambda_start 的比例（0-1） |
| `--lambda_freeze_fraction` | `0.1` | 训练末期冻结 lambda_end 的比例（0-1） |
| `--spatial_alpha` | `0.0` | 空间缩放因子（0 = 禁用空间组件） |

**调度类型可视化**（假设 lambda_start=1.0, lambda_end=0.0）：

```
linear:  1.0 ----____----____----____ 0.0    （直线下降）
cosine:  1.0 ----__------____------__ 0.0    （余弦平滑，两端变化慢）
step:    1.0 ---|0.67---|0.33---|---- 0.0    （在 25%, 50%, 75% 处阶梯跳变）
constant: 保持 lcn_lambda 不变（原始行为）
```

**warmup 和 freeze 示意**：

```
|--warmup--|--------active--------|--freeze--|
  lambda=1.0   (按 schedule 变化)    lambda=0.0
```

**空间 alpha 机制**：

当 `spatial_alpha > 0` 时，每个 episode 的有效 lambda 变为：

```
lambda_effective = max(lambda_base(t), alpha * C(route))
```

其中 `C(route)` 是该 episode 路线经过区域的平均归一化 OD 需求。高需求区域的路线保留更多 Pareto 灵活性，低需求区域更严格执行 Lorenz 公平。

> **调整位置**：
> - 调度器实现：`morl_baselines/multi_policy/lcn/lambda_scheduler.py`
> - 空间上下文计算：`lcn.py` 的 `_compute_route_contexts()` 方法
> - 有效 lambda 应用：`lcn.py` 的 `_nlargest()` 方法中 `interpolate`/`interpolate2`/`interpolate3` 分支
> - 训练循环中的调度更新：`lcn.py` 的 `train()` 方法（`while self.global_step < total_timesteps` 循环顶部）

**推荐配置**：

```bash
# 余弦课程 + 空间调节
--lambda_schedule cosine --lambda_start 1.0 --lambda_end 0.0 \
--lambda_warmup_fraction 0.1 --lambda_freeze_fraction 0.1 \
--spatial_alpha 0.5
```

### 4.2 创新二：OD 需求上下文观测

将归一化的聚合 OD 需求向量拼接到 agent 的观测空间中。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--include_demand_context` | `False` | 启用后观测维度从 `2*grid_size` 变为 `3*grid_size` |

**效果**：

| 环境 | 原始观测维度 | 启用后观测维度 |
|------|-------------|---------------|
| dilemma | 50 (2x25) | 75 (3x25) |
| xian | 1682 (2x841) | 2523 (3x841) |
| amsterdam | 3290 (2x1645) | 4935 (3x1645) |

观测向量组成：`[已覆盖站点 (grid_size) | 当前位置 one-hot (grid_size) | OD 需求上下文 (grid_size)]`

> **调整位置**：
> - 观测空间定义：`envs/mo-tndp/motndp/motndp.py` 的 `__init__()` 和 `get_agent_location()` 方法
> - 需求上下文通过 `city.agg_od_mx()` 计算，min-max 归一化到 [0, 1]
>
> **注意**：对于大环境（amsterdam），观测维度增大可能需要增加 `--hidden_dim`（建议 128）。

### 4.3 创新三：新公平性指标（仅评估）

这些指标**不影响训练过程**，仅在 evaluation checkpoint 时记录到 wandb。

| wandb 指标 | 含义 |
|-----------|------|
| `eval/maxmin_floor_median` | Max-Min Satisfaction Floor 中位数（Rawlsian 公平） |
| `eval/maxmin_floor_max` | Max-Min Satisfaction Floor 最大值 |
| `eval/spatial_sw_high_median` | 高需求区域 Sen Welfare 中位数 |
| `eval/spatial_sw_low_median` | 低需求区域 Sen Welfare 中位数 |
| `eval/spatial_sw_ratio` | 低/高需求区 SW 比值（越接近 1 越均衡） |

**Max-Min Satisfaction Floor**：所有有需求的网格单元中，满足率最低的那个值。衡量最弱势区域是否被照顾到。

**Spatial Sen Welfare**：将网格单元按 OD 需求中位数分为高需求区和低需求区，分别计算 Sen Welfare（效率 x (1-Gini)）。`spatial_sw_ratio` 反映两个区域的服务均衡性。

> **调整位置**：
> - 指标函数：`morl_baselines/common/performance_indicators.py`（`max_min_satisfaction_floor()`, `spatial_sen_welfare()`）
> - 每格满足率计算：`envs/mo-tndp/motndp/city.py` 的 `compute_cell_satisfaction()` 方法
> - 日志记录：`morl_baselines/common/evaluation.py` 的 `log_all_multi_policy_metrics()` 函数
> - 训练循环中的调用：`lcn.py` 的 `train()` 方法中 evaluation checkpoint 部分（约第 760 行）

---

## 5. 消融实验

### 5.1 实验矩阵

| 实验名 | lcn_lambda | schedule | spatial_alpha | demand_context | 目的 |
|--------|-----------|----------|---------------|----------------|------|
| `baseline_lorenz` | 0.0 | constant | 0 | No | 纯 Lorenz 基线 |
| `baseline_pareto` | 1.0 | constant | 0 | No | 纯 Pareto 基线 |
| `temporal` | 0.5 | cosine 1.0->0.0 | 0 | No | 仅时间课程 |
| `spatial` | 0.5 | constant | 0.5 | No | 仅空间 lambda |
| `spatiotemporal` | 0.5 | cosine 1.0->0.0 | 0.5 | No | 组合 lambda |
| `context` | 0.0 | constant | 0 | Yes | 仅需求观测 |
| `full` | 0.5 | cosine 1.0->0.0 | 0.5 | Yes | 全部创新 |

### 5.2 在 SLURM 集群上运行

**提交单个实验**：

```bash
sbatch jobs/ablation_xian.sh 42 temporal
sbatch jobs/ablation_amsterdam.sh 42 spatiotemporal
```

**提交全部实验（7 实验 x 5 种子 x 2 城市 = 70 个 job）**：

```bash
for EXP in baseline_lorenz baseline_pareto temporal spatial spatiotemporal context full; do
  for SEED in 42 123 456 789 1024; do
    sbatch jobs/ablation_xian.sh $SEED $EXP
    sbatch jobs/ablation_amsterdam.sh $SEED $EXP
  done
done
```

**或使用批量脚本**：

```bash
./jobs/run_all_ablations.sh            # 提交全部到 SLURM
./jobs/run_all_ablations.sh --dry-run  # 仅打印命令，不执行
./jobs/run_all_ablations.sh --local    # 本地顺序执行（测试用）
```

> **注意**：SLURM 脚本中的 Python 路径（`/home/dmichai/anaconda3/envs/mo-nw-design/bin/python`）需要改成你自己的 conda 环境路径。修改位置在 `jobs/ablation_xian.sh` 和 `jobs/ablation_amsterdam.sh` 的 `PYTHON=` 行。

### 5.3 本地运行

```bash
# 单个实验
bash jobs/ablation_xian.sh 42 temporal

# 直接用 python 命令（不依赖 SLURM 脚本）
python train_lcn.py --env xian --nr_groups 3 --nr_stations 20 \
  --starting_loc_x 9 --starting_loc_y 19 \
  --timesteps 30000 --batch_size 128 --hidden_dim 128 --lr 0.1 \
  --max_buffer_size 50 --nr_layers 1 \
  --num_er_episodes 50 --num_model_updates 5 --num_step_episodes 10 \
  --distance_ref interpolate3 --lcn_lambda 0.5 \
  --lambda_schedule cosine --lambda_start 1.0 --lambda_end 0.0 \
  --lambda_warmup_fraction 0.1 --lambda_freeze_fraction 0.1 \
  --spatial_alpha 0.5 --include_demand_context \
  --seed 42
```

---

## 6. 评估与结果分析

### 6.1 wandb 关键指标

| 类别 | 指标 | wandb key | 含义 |
|------|------|-----------|------|
| **效率** | Hypervolume | `eval/hypervolume` | Pareto 前沿质量（越大越好） |
| | EUM | `eval/eum` | Expected Utility Metric |
| **公平** | Gini (min) | `eval/gini_min` | 最公平解的基尼系数（越低越好） |
| | Sen Welfare | `eval/sen_welfare_max` | 效率 x 公平的综合指标（越大越好） |
| | Nash Welfare | `eval/nash_welfare_max` | 各目标乘积（越大越好） |
| **空间公平** | Max-Min Floor | `eval/maxmin_floor_max` | 最弱势区域满足率（越高越好） |
| | SW Ratio | `eval/spatial_sw_ratio` | 高低需求区均衡度（越接近 1 越好） |
| **训练** | Lambda | `train/lcn_lambda` | 当前 lambda 值曲线 |
| | Loss | `train/loss` | 训练损失 |
| | Train HV | `train/hypervolume` | ER buffer 中的 hypervolume |

### 6.2 核心假设验证

实验结果应验证以下假设：

1. **时间课程 > 固定 lambda**：`temporal` 的 HV 和 Sen Welfare 应同时优于 `baseline_lorenz` 和 `baseline_pareto`。
2. **空间调节有益**：`spatiotemporal` 的 spatial_sw_ratio 应优于 `temporal`。
3. **需求上下文提升空间公平**：`context` 和 `full` 的 maxmin_floor 应优于无上下文的对应实验。
4. **组合效果**：`full` 在综合指标上应为最优或接近最优。

### 6.3 结果保存

训练权重自动保存到 `./results/lcn_{env}_{timestamp}/` 目录下。

### 6.4 可视化工具

项目提供 `visualize_results.py` 脚本，支持以下可视化方式：

#### 方式一：从保存的模型生成图表

加载训练好的 `.pt` 模型，重新运行评估并绘制：

```bash
# 单个模型
python visualize_results.py --env dilemma --nr_stations 9 \
    --model_path results/lcn_amsterdam_20260428_00_28_22.687141/LCN_model_0.pt \
    --lcn_lambda 0.5 --distance_ref interpolate2

# 对比多个模型（例如消融实验）
python visualize_results.py --env dilemma --nr_stations 9 \
    --model_path results/run_baseline/LCN_model_0.pt \
               results/run_temporal/LCN_model_0.pt \
               results/run_full/LCN_model_0.pt \
    --labels "Baseline" "Temporal" "Full" \
    --lcn_lambda 0.5 --distance_ref interpolate2

# 保存图片到文件夹（不弹窗）
python visualize_results.py --env xian --nr_groups 3 --nr_stations 20 \
    --starting_loc_x 9 --starting_loc_y 19 \
    --model_path results/.../LCN_model_0.pt \
    --lcn_lambda 0.5 --distance_ref interpolate3 \
    --hidden_dim 128 \
    --save_dir figures/
```

生成的图表包括：

| 图表 | 文件名 | 说明 |
|------|--------|------|
| Pareto 前沿散点图 | `pareto_front.png` | 2 目标时为散点图，>2 目标时为平行坐标图 |
| 分组奖励柱状图 | `group_rewards.png` | 各 group 的平均 return 对比（含标准差） |
| 公平性指标对比 | `fairness_metrics.png` | Efficiency、Sen Welfare、Nash Welfare、1-Gini |
| 城市路线图 | `routes.png` | 在 OD 需求热力图上叠加生成的路线 |
| 格满足率热力图 | `cell_satisfaction.png` | 每个网格单元的平均 OD 满足率 |

#### 方式二：从 wandb CSV 导出绘制训练曲线

1. 在 wandb 项目页面，选中要对比的 runs
2. 点击右上角 "Export" → 下载 CSV
3. 运行：

```bash
python visualize_results.py --from_csv wandb_export.csv
python visualize_results.py --from_csv wandb_export.csv --save_dir figures/
```

会自动识别并分组绘制 Hypervolume、Loss、Gini、Sen Welfare、Lambda 曲线、Max-Min Floor、Spatial SW 等指标。

#### 方式三：直接使用 wandb 面板

训练时如果没有加 `--no_log`，所有指标会自动上传到 wandb。推荐在 wandb 上创建以下面板：

1. **训练监控面板**：
   - `train/hypervolume` — ER buffer 质量
   - `train/loss` — 模型收敛
   - `train/lcn_lambda` — lambda 调度曲线（验证课程是否正确执行）

2. **评估对比面板**：
   - `eval/hypervolume` — Pareto 前沿质量
   - `eval/sen_welfare_max` — 效率 x 公平
   - `eval/gini_min` — 最优公平性

3. **空间公平面板**：
   - `eval/maxmin_floor_max` — Rawlsian 公平
   - `eval/spatial_sw_high_median` vs `eval/spatial_sw_low_median` — 区域对比
   - `eval/spatial_sw_ratio` — 区域均衡度

---

## 7. 自定义实验指南

### 7.1 添加新的 Lambda 调度策略

编辑 `morl-baselines/morl_baselines/multi_policy/lcn/lambda_scheduler.py`：

1. 在 `__init__` 的 assert 中添加新类型名
2. 在 `get_base_lambda()` 中添加对应的计算逻辑
3. 在 `train_lcn.py` 的 `--lambda_schedule` choices 中添加新选项

### 7.2 调整空间 alpha 的计算方式

当前实现：`lambda_effective = max(lambda_base, alpha * mean_route_demand)`

如需修改（如使用 max 而非 mean，或使用非线性映射）：

编辑 `lcn.py` 的 `_compute_route_contexts()` 方法和 `_nlargest()` 中的 effective lambda 计算部分。

### 7.3 添加新环境

1. 在 `envs/mo-tndp/cities/` 下创建城市目录，包含 `config.txt`、`od.txt`、`groups.txt`
2. 在 `envs/mo-tndp/motndp/__init__.py` 中注册新环境
3. 在 `train_lcn.py` 中添加对应的 `elif args.env == 'your_city':` 分支，设置 scaling_factor、ref_point 等

### 7.4 修改评估频率

评估在每完成总步数的 1% 时触发。修改位置：`lcn.py` 的 `train()` 方法中：

```python
if self.global_step >= (n_checkpoints + 1) * total_timesteps / 100:
```

将 `100` 改为更小的值可以更频繁地评估（如 `20` = 每 5% 评估一次）。

---

## 8. 文件结构速查

```
train_lcn.py                                    # 训练入口 + CLI 参数定义
visualize_results.py                            # 可视化工具（Pareto 前沿、路线图等）
jobs/
  ablation_xian.sh                              # Xi'an 消融实验 SLURM 脚本
  ablation_amsterdam.sh                         # Amsterdam 消融实验 SLURM 脚本
  run_all_ablations.sh                          # 批量提交全部消融实验
envs/mo-tndp/motndp/
  motndp.py                                     # MOTNDP 环境（观测空间、奖励）
  city.py                                       # City 类（OD 矩阵、满足率计算）
morl-baselines/morl_baselines/
  multi_policy/lcn/
    lcn.py                                      # LCNTNDP 算法主体
    lambda_scheduler.py                         # Lambda 时间调度器
  common/
    evaluation.py                               # 评估日志（含新指标）
    performance_indicators.py                   # 指标函数（HV、Gini、新指标等）
```
