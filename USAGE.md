# TWIST2-tienkung 使用说明与架构说明（含 Stage1 AMP）

本文档面向当前仓库的实际代码状态，重点说明：

1. 改造后的训练流程（Stage1 AMP + Stage2 Student）
2. 运行环境约束（conda twist2 / gmr 双环境）
3. 关键脚本入口与常见排错
4. 代码架构与模块职责

---

## 1. 仓库目标与当前默认行为

本仓库当前支持三类核心能力：

- 低层策略训练（Isaac Gym + rsl_rl + legged_gym）
- ONNX 导出与 sim2sim/sim2real 低层控制部署
- 远程动作流（motion server / teleop）与 Redis 桥接

当前训练相关默认行为如下：

- Stage2 默认入口仍是 `train_stage2.sh`，任务是 `g1_stu_future`
- 新增 Stage1 AMP 入口是 `train_stage1_amp.sh`，任务是 `g1_priv_mimic_amp`
- 新增 AMP 不会污染旧任务：`g1_priv_mimic` 和 `g1_stu_future` 保持原行为

---

## 2. 环境与依赖（按当前机器配置）

### 2.1 Python 环境约定

仓库沿用双环境设计：

- `twist2`（conda，Python 3.8）：训练、导出、部署主流程
- `gmr`（conda/venv，Python 3.10+）：GMR 重定向相关

> 说明：当前脚本已按 conda `twist2` 进行修复和固化，不再依赖你手动先激活 `.venv-twist2`。

### 2.2 已加入的脚本级环境修复

以下脚本已内置：

- `source /home/vega/anaconda3/etc/profile.d/conda.sh`
- `conda activate twist2`
- `export LD_LIBRARY_PATH=/home/vega/anaconda3/envs/twist2/lib:${LD_LIBRARY_PATH}`

已更新脚本：

- `train_stage2.sh`
- `train_tienkung.sh`
- `train_stage1_amp.sh`
- `eval.sh`
- `eval_tienkung.sh`
- `to_onnx.sh`

这样可以规避 Isaac Gym 常见错误：`libpython3.8.so.1.0` 找不到。

---

## 3. 快速开始（建议顺序）

### 3.1 配置数据路径

按你要训练的任务，确认 motion yaml 路径和 root_path 配置正确：

- Stage1 AMP（Teacher）：`motion_data_configs/unitree_g1_retarget_fft.yaml`
- Stage2 Student：`motion_data_configs/unitree_g1_retarget.yaml`

### 3.2 Stage1：训练 AMP Teacher（新）

```bash
bash train_stage1_amp.sh 0410_amp_stage1 cuda:0
```

这条命令只会训练 Stage1 Teacher，任务为 `g1_priv_mimic_amp`。

训练日志中重点关注：

- `Loss/discriminator`
- `Loss/discriminator_accuracy`
- `Train/mean_amp_reward`
- PPO 原有 loss 项

### 3.3 Stage2：训练 Student（原默认流程）

```bash
bash train_stage2.sh 0410_stage2_stu cuda:0
```

默认任务是 `g1_stu_future`，算法是 `DaggerPPO`。

如需蒸馏到你新的 Stage1 Teacher，需要在 Stage2 配置/参数中指向对应 teacher 实验（例如 teacher experiment id / checkpoint）。

### 3.4 导出 ONNX

```bash
bash to_onnx.sh /path/to/your/checkpoint.pt
```

说明：AMP 改动只在训练链路，`save_onnx.py` 不依赖 AMP 模块，Student 导出路径保持可用。

### 3.5 sim2sim 最小验证

```bash
bash run_motion_server.sh
bash sim2sim.sh
```

如果是天工：

```bash
bash run_motion_server.sh tienkung /path/to/motion.pkl
bash sim2sim_tienkung.sh
```

### 3.6 离线评估仿真跟踪质量

仓库现在提供了一个离线评估脚本，用于比较“仿真中机器人实际动作”和“参考动作”之间的差异。这个流程不侵入实时控制回路，适合做回放分析、指标对比和批量回归。

评估入口：

- `deploy_real/eval_sim_tracking.py`
- 默认配置文件：`deploy_real/data_utils/eval_sim_tracking_config.json`

#### 第一步：采集仿真轨迹

评估脚本读取的是 `record_proprio` 生成的 pkl 记录文件。因此在采集评估数据前，需要先打开 `sim2sim.sh` 中的 `--record_proprio` 开关，重新跑一段仿真。

典型流程：

```bash
bash run_motion_server.sh
bash sim2sim.sh
```

采集完成后，`deploy_real/server_low_level_g1_sim.py` 会输出：

- `twist2_proprio_recordings.pkl`

该文件中会记录：

- 时间戳
- 关节位置 `dof_pos`
- 关节速度 `dof_vel`
- 根部位置 `root_pos`
- 根部姿态四元数 `root_quat`
- 根部线速度 `root_lin_vel`
- 根部角速度 `root_ang_vel`

这些字段足以支持关节误差、根部误差、速度误差和 key body 误差计算。

#### 第二步：单文件评估

```bash
/home/vega/anaconda3/envs/twist2/bin/python deploy_real/eval_sim_tracking.py \
  --sim_pkl deploy_real/twist2_proprio_recordings.pkl \
  --motion_file assets/example_motions/0807_yanjie_walk_001.pkl \
  --xml assets/g1/g1_sim2sim_29dof.xml \
  --out_dir deploy_real/eval_outputs
```

常用参数说明：

- `--sim_pkl`：待评估的仿真记录文件
- `--motion_file`：参考动作文件，传给 `MotionLib`
- `--xml`：用于 key body 正向运动学的 MuJoCo XML
- `--out_dir`：评估结果输出目录
- `--motion_id`：当 `motion_file` 是 yaml 或多 motion 集合时，指定使用哪个 motion
- `--start_time`：从参考动作的哪个时间点开始对齐
- `--max_duration`：限制评估时长，便于做短片段对比
- `--disable_yaw_align`：关闭 yaw 对齐，适合排查全局朝向误差

#### 第三步：批量评估与排行榜

当你有多段 pkl 记录需要横向比较时，可以使用批量模式：

```bash
/home/vega/anaconda3/envs/twist2/bin/python deploy_real/eval_sim_tracking.py \
  --batch_glob "deploy_real/records/*.pkl" \
  --motion_file assets/example_motions/0807_yanjie_walk_001.pkl \
  --xml assets/g1/g1_sim2sim_29dof.xml \
  --out_dir deploy_real/eval_batch_outputs
```

批量模式会：

- 对每个 pkl 单独生成一个结果子目录
- 额外生成总排行榜文件
- 按配置文件中的排序规则输出 leaderboard

#### 默认配置文件

评估默认阈值、权重和排行榜排序规则来自：

- `deploy_real/data_utils/eval_sim_tracking_config.json`

配置项包括：

- `weights`：综合分数各指标权重
- `thresholds`：每个指标的 success 阈值
- `pass_criteria.success_rate`：最低成功率要求
- `pass_criteria.score_mean`：平均综合分数上限
- `leaderboard.sort_by`：排行榜排序字段
- `leaderboard.ascending`：是否升序排序

如果你只想临时覆盖配置，而不改 JSON 文件，可以在命令行中追加：

```bash
--weights joint=0.4,key_body=0.2,root_pos=0.15,root_rot=0.15,velocity=0.1
--thresholds joint=0.12,key_body=0.1,root_pos=0.06,root_rot=8.0,velocity=0.2
```

#### 输出文件说明

单文件评估默认生成：

- `summary.json`
- `frame_metrics.csv`
- `metrics_plot.png`

各文件含义如下：

- `summary.json`
  - 本次评估的总体摘要
  - 包含 `success_rate`、`pass`、`frame_score` 统计、各项指标均值/P95/最大值
  - 包含本次运行使用的 `weights`、`thresholds`、`motion_id`、`key_bodies_used`

- `frame_metrics.csv`
  - 逐帧误差明细
  - 每一帧包含 joint/root/key body/velocity 误差、综合分数以及 success 标记
  - 适合后续做 pandas 分析或导入表格软件查看

- `metrics_plot.png`
  - 误差随时间变化曲线图
  - 便于快速定位哪一段动作开始偏离参考动作

批量评估会在总输出目录额外生成：

- `leaderboard.json`
- `leaderboard.csv`

其中：

- `leaderboard.json` 适合程序读取
- `leaderboard.csv` 适合人工浏览、排序和做实验记录

排行榜默认包含：

- `run_name`
- `success_rate`
- `frame_score_mean`
- `frame_score_p95`
- `joint_mean`
- `root_pos_mean`
- `root_rot_mean`
- `key_body_mean`
- `velocity_mean`
- `pass`

#### 指标解释

当前离线评估脚本默认统计以下误差项：

- `joint`：关节位置平均绝对误差
- `root_pos`：根部位置平均绝对误差
- `root_rot`：根部姿态角度误差，单位为度
- `key_body`：关键 body 相对根部位置误差
- `velocity`：关节速度与根部速度的综合误差

综合分数 `frame_score` 是按配置文件中的 `weights` 对归一化误差加权得到。数值越小，代表跟踪效果越好。

#### 结果解读建议

- `success_rate` 高、`frame_score_mean` 低：通常表示整体跟踪稳定
- `joint` 误差低但 `root_pos` 或 `root_rot` 误差高：通常说明局部姿态接近，但全局位姿漂移明显
- `key_body` 误差偏高：通常说明末端或上肢动作没有很好跟上参考动作
- `velocity` 误差偏高：通常说明动作节奏、加减速或切换阶段没有对齐

如果你只看一个指标，容易误判。更稳妥的做法是同时看：

- `summary.json` 的总体统计
- `metrics_plot.png` 的时间曲线
- `leaderboard.csv` 的横向排名

---

## 4. 改造后训练架构

### 4.1 总体流程

```text
               Stage1 (Teacher, AMP)
  motion_data_configs/unitree_g1_retarget_fft.yaml
                          |
                          v
               task: g1_priv_mimic_amp
               runner: OnPolicyAMPMimicRunner
               algo:   AMPPPO
               env:    G1MimicDistill(enable_amp=True)
                          |
                          v
                Teacher checkpoint (.pt)
                          |
                          v
               Stage2 (Student, DAggerPPO)
  motion_data_configs/unitree_g1_retarget.yaml
               task: g1_stu_future
```

### 4.2 新增 AMP 组件

- 判别器：`rsl_rl/rsl_rl/modules/amp_discriminator.py`
  - MLP + SiLU + BCE with logits
- 算法：`rsl_rl/rsl_rl/algorithms/amp_ppo.py`
  - 在 PPO 基础上加入判别器优化
  - 计算 AMP reward 并裁剪
  - gradient penalty 正则
- Runner：`rsl_rl/rsl_rl/runners/on_policy_amp_mimic_runner.py`
  - 采集 policy/demo 的 AMP 观测
  - rollout 中注入 AMP reward
  - 更新判别器，记录 AMP 指标
  - 保存/加载判别器权重与优化器状态

### 4.3 环境侧 AMP 观测接口

- `legged_gym/legged_gym/envs/base/humanoid_mimic.py`
  - `_get_amp_obs()`
  - `_get_amp_demo_obs()`
  - 在 `enable_amp=True` 时写入 `extras["amp_obs"]`、`extras["amp_demo_obs"]`
- `legged_gym/legged_gym/envs/g1/g1_mimic_distill.py`
  - 对应 AMP extras 注入逻辑

---

## 5. 任务映射与用途

| 任务名 | 用途 | 数据集 | 算法 | Runner | AMP |
|---|---|---|---|---|---|
| `g1_priv_mimic` | 原 Stage1 Teacher 基线 | `g1_omomo+mocap_static+amass_walk.yaml` | PPO | OnPolicyRunnerMimic | 否 |
| `g1_priv_mimic_amp` | 新 Stage1 Teacher（AMP） | `unitree_g1_retarget_fft.yaml` | AMPPPO | OnPolicyAMPMimicRunner | 是 |
| `g1_stu_future` | Stage2 Student | `unitree_g1_retarget.yaml` | DaggerPPO | OnPolicyDaggerRunner | 否 |

注册位置：`legged_gym/legged_gym/envs/__init__.py`

---

## 6. 关键配置入口

### 6.1 Stage1 AMP 配置类

文件：`legged_gym/legged_gym/envs/g1/g1_mimic_distill_config.py`

新增类：

- `G1MimicPrivAmpCfg`
  - `env.enable_amp = True`
  - `motion.motion_file = unitree_g1_retarget_fft.yaml`
- `G1MimicPrivAmpCfgPPO`
  - `algorithm_class_name = 'AMPPPO'`
  - `runner_class_name = 'OnPolicyAMPMimicRunner'`
  - AMP 参数：`amp_reward_scale`、`amp_obs_dim`、`amp_disc_*`

### 6.2 Stage2 保持不变

`train_stage2.sh` 依然训练 `g1_stu_future`，用于 Student 训练，不会自动切到 AMP。

---

## 7. 常见操作手册

### 7.1 先做配置级自检

```bash
source /home/vega/anaconda3/etc/profile.d/conda.sh
conda activate twist2
export LD_LIBRARY_PATH=/home/vega/anaconda3/envs/twist2/lib:${LD_LIBRARY_PATH}

python - <<'PY'
import sys
sys.path.insert(0, 'legged_gym')
sys.path.insert(0, 'rsl_rl')
import isaacgym
from legged_gym.envs import task_registry
for t in ['g1_priv_mimic', 'g1_priv_mimic_amp', 'g1_stu_future']:
    env_cfg, train_cfg = task_registry.get_cfgs(t)
    print(t, env_cfg.motion.motion_file.split('/')[-1], train_cfg.runner.algorithm_class_name)
PY
```

### 7.2 Stage1 AMP 短跑 smoke test

```bash
bash train_stage1_amp.sh 0410_amp_stage1_smoke cuda:0
```

建议先跑小步数观察是否出现 NaN、判别器指标是否更新。

### 7.3 Stage2 短跑 smoke test

```bash
bash train_stage2.sh 0410_stage2_smoke cuda:0
```

确认 Dagger/PPO loss 正常、teacher 加载链路正常。

---

## 8. 排错指南

### 8.1 `libpython3.8.so.1.0` 找不到

现象：Isaac Gym import 时报共享库错误。

处理：确认使用 `twist2`，并设置：

```bash
export LD_LIBRARY_PATH=/home/vega/anaconda3/envs/twist2/lib:${LD_LIBRARY_PATH}
```

### 8.2 在 `.venv-twist2` 中运行导致版本冲突

现象：Python 版本或依赖不匹配（尤其 Isaac Gym）。

处理：训练/导出/部署流程优先使用 `conda activate twist2`。

### 8.3 AMP 维度不匹配

现象：`amp_obs_dim` 相关 shape error。

处理：

1. 检查 `G1MimicPrivAmpCfgPPO.algorithm.amp_obs_dim`
2. 检查 `_get_amp_obs()` 组成维度是否与配置一致
3. 检查 key body 数量是否被改动

---

## 9. 目录架构（与训练/部署强相关）

```text
.
├── train_stage2.sh                 # Stage2 默认入口（g1_stu_future）
├── train_stage1_amp.sh             # Stage1 AMP 入口（g1_priv_mimic_amp）
├── to_onnx.sh                      # 导出入口
├── sim2sim.sh / sim2sim_tienkung.sh
├── run_motion_server.sh
├── deploy_real/
│   ├── server_motion_lib.py        # 高层动作源
│   ├── server_low_level_g1_sim.py  # 低层 sim 控制
│   ├── server_low_level_g1_real.py # 低层 real 控制
│   └── server_low_level_tienkung_sim.py
├── legged_gym/
│   └── legged_gym/
│       ├── envs/
│       │   ├── __init__.py         # task_registry 注册中心
│       │   ├── base/humanoid_mimic.py
│       │   └── g1/g1_mimic_distill_config.py
│       └── scripts/train.py
├── rsl_rl/
│   └── rsl_rl/
│       ├── modules/amp_discriminator.py
│       ├── algorithms/amp_ppo.py
│       └── runners/on_policy_amp_mimic_runner.py
└── motion_data_configs/
    ├── unitree_g1_retarget_fft.yaml
    └── unitree_g1_retarget.yaml
```

---

## 10. 推荐实验流程

1. 先跑 Stage1 AMP 短实验，确认判别器和 AMP reward 正常。
2. 固定一个稳定的 Stage1 checkpoint 作为 teacher。
3. 再跑 Stage2 student 蒸馏，观察跟踪稳定性与泛化表现。
4. 选取 Stage2 checkpoint 导出 ONNX。
5. 进入 sim2sim，再进入 sim2real。

---

## 11. 文档对照

- 安装与总体说明：`README.md`
- 训练架构与分析：`ARCHITECTURE.md`
- teleop 流程：`doc/TELEOP.md`
- 机器人部署：`doc/unitree_g1.md`、`doc/unitree_g1.zh.md`
- 颈部硬件：`doc/TWIST2_NECK.md`
- GMR 模块：`GMR/README.md`、`GMR/DOC.md`

如你后续希望，我可以继续补一版「一键最小可复现实验脚本清单」（按 15 分钟 smoke test 路径编排）。
