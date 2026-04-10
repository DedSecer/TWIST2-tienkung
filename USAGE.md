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
