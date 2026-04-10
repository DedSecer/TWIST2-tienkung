# TWIST2 架构分析与扩展建议

> 本文档基于对代码库的详细阅读，记录 TWIST2 低层控制器（RL 策略训练部分）的完整架构，并提供扩展建议。

---

## 目录

1. [系统总览](#1-系统总览)
2. [参考动作处理流程](#2-参考动作处理流程)
3. [两阶段蒸馏训练架构](#3-两阶段蒸馏训练架构)
4. [强化学习环境设计](#4-强化学习环境设计)
5. [网络结构详解](#5-网络结构详解)
6. [算法：DAggerPPO](#6-算法daggerppo)
7. [奖励函数](#7-奖励函数)
8. [课程学习机制](#8-课程学习机制)
9. [部署流程](#9-部署流程)
10. [扩展建议](#10-扩展建议)

---

## 1. 系统总览

TWIST2 的低层控制器训练遵循 **Motion Imitation + 知识蒸馏** 的范式：

```
参考动作数据 (.pkl)
       │
       ▼
  MotionLib（分帧查询、插值）
       │
       ▼
┌─────────────────────────────────────────┐
│  Stage 1：Teacher Policy 训练 (PPO)      │
│  输入：20帧未来参考动作（特权信息）        │
│  网络：ActorCriticMimic (Conv1D)         │
└─────────────────────────────────────────┘
       │  Teacher checkpoint
       ▼
┌─────────────────────────────────────────┐
│  Stage 2：Student Policy 训练 (DAggerPPO)│
│  输入：当前帧 + 历史 + 可选未来帧         │
│  网络：ActorCriticFuture (MoE)           │
│  损失：PPO + KL(student ‖ teacher)       │
└─────────────────────────────────────────┘
       │  .pt checkpoint
       ▼
  ONNX 导出 → 真机部署（50 Hz）
```

**核心设计思想**：Teacher 在模拟中获得未来动作的完整信息（特权观测），学会高质量的跟踪策略；Student 只能获得当前帧和历史信息（可部署），通过模仿 Teacher 的行为分布来弥补信息差距。

**关键文件索引**：

| 文件 | 职责 |
|------|------|
| `legged_gym/legged_gym/envs/base/humanoid_mimic.py` | Motion Imitation 环境核心逻辑 |
| `legged_gym/legged_gym/envs/g1/g1_mimic_distill_config.py` | Teacher (`G1MimicPrivCfg`) 配置 |
| `legged_gym/legged_gym/envs/g1/g1_mimic_future_config.py` | Student (`G1MimicStuFutureCfg`) 配置，默认训练目标 |
| `rsl_rl/rsl_rl/algorithms/dagger_ppo.py` | DAggerPPO 算法实现 |
| `rsl_rl/rsl_rl/modules/actor_critic_future.py` | Student 网络（含 FutureEncoder、MotionEncoder） |
| `rsl_rl/rsl_rl/runners/on_policy_dagger_runner.py` | 训练 Runner，管理 Teacher/Student 加载 |
| `legged_gym/motion_data_configs/unitree_g1_retarget.yaml` | 默认动作数据集（GMR 重定向 AMASS） |

---

## 2. 参考动作处理流程

### 2.1 数据格式

动作数据以 `.pkl` 文件存储，通过 [GMR](https://github.com/lithiumice/GMR) 工具从 AMASS/SMPLX 人体动作数据集重定向到机器人关节空间。

**YAML 配置**（`motion_data_configs/`）：
```yaml
root_path: /path/to/retargeted/motions
motions:
- file: relative/path/to/motion_01.pkl
  weight: 1.0      # 控制采样概率
- file: relative/path/to/motion_02.pkl
  weight: 2.0      # 此文件被采样概率是上面的 2 倍
```

**pkl 文件字段**：

| 字段 | 形状 | 含义 |
|------|------|------|
| `fps` | scalar | 动作帧率（通常 30 Hz） |
| `root_pos` | (T, 3) | 根节点世界坐标 |
| `root_rot` | (T, 4) | 根节点四元数 \[x,y,z,w\] |
| `root_vel` | (T, 3) | 根节点线速度 |
| `root_ang_vel` | (T, 3) | 根节点角速度 |
| `dof_pos` | (T, 29) | 关节角度（按机器人 DOF 顺序） |
| `dof_vel` | (T, 29) | 关节角速度 |
| `body_pos` | (T, num_bodies, 3) | 所有刚体在根节点局部坐标系中的位置 |
| `key_body_names` | list | 关键体名称（手、脚等） |

### 2.2 MotionLib 时间查询

`calc_motion_frame(motion_ids, times)` 内建线性插值，支持任意时刻查询：

```python
# humanoid_mimic.py:164
# 当前时刻 = 已过步数 × 控制周期 + 起始偏移（RSI）
motion_times = episode_length_buf * dt + motion_time_offsets
```

### 2.3 Reference State Initialization (RSI)

每个 episode 重置时，机器人以参考动作的随机时刻姿态初始化（`rand_reset=True`），避免策略只能从固定起点开始跟踪：

```python
# humanoid_mimic.py:249
self._reset_dofs(env_ids, ref_dof_pos, ref_dof_vel * 0.8)
self._reset_root_states(env_ids, root_vel=ref_root_vel * 0.8, ...)
```

速度按 0.8 倍初始化，防止初始速度过大导致不稳定。

---

## 3. 两阶段蒸馏训练架构

### 3.1 Stage 1：Teacher Policy（可选）

| 属性 | 值 |
|------|----|
| 任务名 | `g1_priv_mimic` |
| 算法 | PPO（`on_policy_runner_mimic.py`） |
| 网络 | `ActorCriticMimic` |
| obs 类型 | `'priv'`：完整特权信息 |

Teacher 观测结构（总维度约 1540+）：
```
n_priv_mimic_obs = 20帧 × (root_pos(3) + roll_pitch(2) + root_vel(3) +
                             yaw_ang_vel(1) + dof_pos(29) + key_body_pos(27))
               = 20 × 65 = 1300 维（未来参考动作）
+ n_proprio（本体感受）
+ n_priv_info（接触状态、质量参数等）
```

未来帧步数：`tar_motion_steps_priv = [1, 5, 10, 15, ..., 95]`（最远看 0.95秒后）。

### 3.2 Stage 2：Student Policy（默认训练目标）

| 属性 | 值 |
|------|----|
| 任务名 | `g1_stu_future` |
| 算法 | DAggerPPO（`on_policy_dagger_runner.py`） |
| 网络 | `ActorCriticFuture` |
| obs 类型 | `'student_future'`：当前帧 + 历史 + 可选未来帧 |

Student 观测结构：
```python
n_mimic_obs_single = 6 + 29 = 35      # 单步动作观测
n_proprio = 3 + 2 + 3 * 29 = 92       # 本体感受
n_obs_single = 35 + 92 = 127          # 单步完整观测
history_len = 10

num_observations = 127 * (10 + 1) + n_future_obs
               = 1397 + n_future_obs
```

单步动作观测分解（35维）：
```
root_pos_z(1) + roll_pitch(2) + root_vel_xy(2) + root_vel_z(1) +
yaw_ang_vel(1) + dof_pos(29) = 36... 具体字段见 humanoid_mimic.py:535
```

### 3.3 蒸馏关键：观测空间差异

| 观测成分 | Teacher | Student |
|----------|---------|---------|
| 未来动作帧 | 20帧（最远 95步） | 0-N 帧（默认 0） |
| 关键体全局位置 | ✓ | ✗ |
| 接触状态 | ✓ | ✗（必须估计） |
| 质量/摩擦参数 | ✓ | ✗（必须适应） |
| 历史观测 | 不需要 | 10步历史（补偿信息缺失） |

---

## 4. 强化学习环境设计

### 4.1 仿真参数

| 参数 | 值 |
|------|----|
| 物理步长 `dt` | 0.002 s |
| 控制 decimation | 10 步 |
| 有效控制频率 | 50 Hz |
| 并行环境数 | 4096 |
| 机器人 DOF | 29 |
| 关节控制 | PD 位置控制 |

**PD 刚度**（Nm/rad）：
```
腿部（hip/knee）: 100-150
踝关节: 40
腰部: 150
肩部: 40
手腕: 视配置
```

### 4.2 终止条件

```python
# humanoid_mimic.py:406
终止触发条件（任意一条）：
  ├─ 非足部接触力 > 1 N
  ├─ 根节点高度偏差超阈值
  ├─ roll / pitch 超限
  ├─ 速度 > 5 m/s
  ├─ 动作序列结束
  └─ 姿态跟踪失败：max(key_body_dist) > pose_termination_dist (0.7 m)
```

### 4.3 G1 机器人 DOF 顺序

```
索引  0-5:   左腿  (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
索引  6-11:  右腿  (同上)
索引 12-14:  腰部  (waist_yaw, waist_roll, waist_pitch)
索引 15-21:  左臂  (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow,
                    wrist_roll, wrist_pitch, wrist_yaw)
索引 22-28:  右臂  (同上)
```

### 4.4 Domain Randomization

| 类别 | 参数 |
|------|------|
| 摩擦系数 | 随机化 |
| 质量 | 随机化 |
| 质心偏移 | 随机化 |
| 重力方向 | 随机化 |
| 电机强度 | `motor_strength ∈ [0.9, 1.1]` |
| 动作延迟 | 随机化 |
| 动作观测噪声 | 可选（`motion_dr_enabled`，默认关闭） |

---

## 5. 网络结构详解

### 5.1 MotionEncoder（Conv1D 时序编码器）

用于编码多步动作参考观测，支持 `tsteps ∈ {1, 10, 20, 50}`：

```
输入: (N×T, input_size)
  └─ Linear(input→60)   # 线性投影
       └─ reshape → (N, 60, T)
            └─ Conv1d × 2-3层（stride 下采样）
                 └─ Flatten
                      └─ Linear(60→output_size=128)
```

### 5.2 HistoryEncoder

与 MotionEncoder 结构相同，编码 10 步本体感受历史，输出 128 维历史潜变量。

### 5.3 FutureMotionEncoder（MLP）

```
输入: (N, T, obs_per_step)  # 展平后 (N, T×obs_per_step)
  └─ Linear(→256) → Dropout → Linear(→128) → Dropout → Linear(→128)
```

### 5.4 ActorFuture（Actor 主干）

```
输入拼接: [single_motion_obs(35) | proprio(92) | motion_latent(128) |
           history_latent(128) | future_latent(128)]
  └─ MLP: 483 → 512 → 512 → 256 → 128 → 29
     激活：SiLU，最后两层 LayerNorm
```

### 5.5 Critic（使用特权观测）

```
特权 obs → MotionEncoder(Conv1D) → motion_latent(128)
  └─ 拼接非动作部分 → MLP: → 512 → 512 → 256 → 128 → 1
```

---

## 6. 算法：DAggerPPO

### 6.1 总损失

```
L_total = L_PPO + λ(t) × KL(π_student ‖ π_teacher)

L_PPO = L_surrogate + c_v × L_value - c_e × H[π_student]

L_surrogate = E[min(r_t A_t, clip(r_t, 1±ε) A_t)]  # clip PPO

λ(t) = λ_0 × 0.5 × (1 + cos(π × t / T))            # 余弦退火
   λ_0 = 0.2, λ_min = 0.1, T = 60000 步
```

### 6.2 KL 散度的计算

Teacher 和 Student 都输出高斯分布 `N(μ, σ)`：

```python
# dagger_ppo.py:44
KL = log(σ_t/σ_s) + (σ_s² + (μ_s - μ_t)²) / (2σ_t²) - 0.5
```

Teacher 使用 `critic_obs_batch`（特权观测）推理，Student 使用 `obs_batch`（受限观测）推理，损失驱动 Student 的行为分布向 Teacher 靠近。

### 6.3 动作标准差调度

```python
std_schedule = [1.0, 0.4, 4000, 1500]
# 含义：从 1.0 线性衰减到 0.4，从第 4000 步开始，经过 1500 步完成
```

探索早期用大方差，收敛后用小方差增加确定性。

---

## 7. 奖励函数

奖励函数整体使用 `exp(-scale × error²)` 形式，误差为零时奖励为 1。

### 主要跟踪奖励

| 奖励项 | 权重 | 计算 |
|--------|------|------|
| `tracking_joint_dof` | **2.0** | `exp(-0.15 × Σ w_i(q_ref - q)²)` |
| `tracking_keybody_pos` | **2.0** | `exp(-10 × Σ‖p_key - p_ref‖²)`（局部坐标） |
| `tracking_keybody_pos_global` | **2.0** | 同上（全局坐标） |
| `tracking_root_translation_z` | 1.0 | `exp(-5 × (z - z_ref)²)` |
| `tracking_root_rotation` | 1.0 | `exp(-5 × quat_diff_angle²)` |
| `tracking_root_linear_vel` | 1.0 | `exp(-1 × ‖v - v_ref‖²)` |
| `tracking_root_angular_vel` | 1.0 | `exp(-1 × ‖ω - ω_ref‖²)` |
| `tracking_joint_vel` | 0.2 | `exp(-0.01 × Σ(q̇_ref - q̇)²)` |
| `alive` | 0.5 | 常数 1.0（活着就有奖励） |

### 辅助奖励与惩罚

| 奖励项 | 权重 | 含义 |
|--------|------|------|
| `feet_air_time` | 5.0 | 足部离地时间（促进自然步态） |
| `dof_pos_limits` | -5.0 | 关节超出软限位惩罚 |
| `dof_torque_limits` | -1.0 | 力矩超限惩罚 |
| `feet_slip` | -0.1 | 足部接触时滑动惩罚 |
| `feet_contact_forces` | -5e-4 | 过大接触力惩罚 |
| `action_rate` | -0.05 | 动作变化过快惩罚 |
| `dof_acc` | -5e-8 | 关节加速度惩罚 |
| `ang_vel_xy` | -0.01 | 水平角速度惩罚（防止摇晃） |

---

## 8. 课程学习机制

### 8.1 Motion Difficulty Curriculum

每个运动片段独立维护一个难度系数（初始 100），根据完成率动态调整终止距离：

```python
# humanoid_mimic.py:299-355
completion_rate = episode_steps * dt / motion_length

if completion_rate ≤ 0.50:   difficulty *= (1 + γ)   # 放宽，更容忍失败
if completion_rate ≥ 0.95:   difficulty *= (1 - γ)   # 收紧，要求更精确
if completion_rate ≥ 0.99:   difficulty *= (1 - 20γ) # 大幅收紧

termination_dist = (0.7 - 0.2) × (difficulty / 10) + 0.2
# 范围：[0.2, 0.7] m
```

`motion_curriculum_gamma = 0.01`，难度范围 `[1, 10]`。

### 8.2 Error-Aware Sampling（默认关闭）

基于各动作历史最大关键体误差加权采样，优先训练机器人在困难动作上：

```python
# 采样概率 ∝ (max_key_body_error / threshold) ^ power
error_sampling_power = 5.0
error_sampling_threshold = 0.15  # m
```

### 8.3 Motion Domain Randomization（默认关闭）

对参考观测加噪，增强策略对动作信号质量的鲁棒性：

| 扰动类型 | 范围 |
|----------|------|
| 根节点位置 | ±1~5 cm |
| 根节点朝向 | ±5.7~11.4° |
| 根节点速度 | ±0.05~0.1 m/s |
| 关节角度 | ±0.05~0.1 rad |

---

## 9. 部署流程

```
训练完成 → logs/<project>/<exptid>/model_<iter>.pt
  │
  └─ bash to_onnx.sh → assets/ckpts/<name>.onnx
        │
        ├─ Sim 验证：
        │   bash run_motion_server.sh     # Redis 发布参考动作帧
        │   bash sim2sim.sh               # MuJoCo 仿真执行
        │
        └─ 真机部署：
            bash run_motion_server.sh     # 或 teleop.sh（在线遥操作）
            bash sim2real.sh              # 真机控制（Unitree SDK）
```

控制架构（Redis 通信）：
```
[Motion Server]  →(Redis)→  [Low-level Policy Server]  →(SDK)→  [G1 Robot]
run_motion_server.sh         server_low_level_g1_real.py
发布：目标关节角             接收：关节角目标
      根节点位置               推理：ONNX 策略
                               输出：关节力矩（50 Hz）
```

---

## 10. 扩展建议

### 10.1 适配新机器人

TWIST2 对新机器人的适配已有先例（参考 `envs/tienkung/`），步骤如下：

1. **添加 URDF**：放置在 `assets/<robot_name>/urdf/`
2. **重定向动作**：使用 GMR 工具生成 pkl，创建对应 YAML
3. **创建 Config**：继承 `G1MimicStuFutureCfg`，至少覆盖：
   ```python
   class MyRobotCfg(G1MimicStuFutureCfg):
       class env(G1MimicStuFutureCfg.env):
           num_actions = N        # 新机器人 DOF 数
           dof_err_w = [...]      # 各关节误差权重
       class motion(G1MimicStuFutureCfg.motion):
           motion_file = "path/to/my_robot.yaml"
       class asset(G1MimicStuFutureCfg.asset):
           file = "assets/my_robot/urdf/my_robot.urdf"
   ```
4. **注册任务**：在 `envs/__init__.py` 添加注册条目
5. **调整 obs 维度**：确认 `n_mimic_obs_single`、`n_proprio` 等维度正确

**注意**：`ActorCriticFuture` 的网络输入维度在初始化时自动根据 obs 维度计算，不需要手动修改网络代码。

---

### 10.2 使用自定义参考动作

**最简路径**（不改代码）：

```bash
# 1. 准备 pkl 文件（见数据格式要求）
# 2. 创建 YAML
cat > legged_gym/motion_data_configs/my_motions.yaml << EOF
root_path: /abs/path/to/my/motions
motions:
- file: walk_01.pkl
  weight: 1.0
- file: wave_hand.pkl
  weight: 0.5
EOF

# 3. 修改配置（g1_mimic_future_config.py 第 75 行）
# motion_file = ".../my_motions.yaml"

# 4. 训练
bash train.sh --exptid my_exp_001
```

**关键约束**：
- `dof_pos` 的列顺序必须与 G1 的 DOF 顺序严格对应（见第 4.3 节）
- `body_pos` 的体索引顺序必须与 URDF 中刚体顺序一致
- MotionLib 支持不同帧率动作（内部插值），但推荐 ≥ 30 Hz

---

### 10.3 增加未来帧信息（提升跟踪质量）

当前默认配置 `tar_motion_steps_future = [0]`（只用当前帧）。增加未来帧预期可以显著改善复杂动作的跟踪效果：

```python
# g1_mimic_future_config.py 第 7 行
TAR_MOTION_STEPS_FUTURE = [1, 2, 3, 4, 5]  # 看 0.1s 后的 5 帧

# obs 维度自动扩展：+ 5 × 35 = + 175 维
# FutureMotionEncoder 自动处理多步输入
```

**建议**：先用 `[1, 2, 3]`（0.06s 预览），观察 Episode 完成率提升后再扩大窗口。

---

### 10.4 开启 Motion Domain Randomization（提升 Sim2Real）

在 `g1_mimic_future_config.py` 中修改：

```python
class motion(G1MimicPrivCfg.motion):
    motion_dr_enabled = True          # 开启
    root_position_noise = [0.01, 0.03]
    root_orientation_noise = [0.05, 0.1]
    joint_position_noise = [0.02, 0.05]
    motion_dr_resampling = True       # 每步重新采样噪声
```

这模拟了真机部署时动作信号（来自 VR 头显/运动捕捉）的噪声与延迟。

---

### 10.5 Error-Aware Sampling（处理动作集难度不均）

当动作集中存在部分极难动作时（如高速翻转），普通随机采样会浪费大量时间在策略完全失败的片段上：

```python
class motion(G1MimicPrivCfg.motion):
    use_error_aware_sampling = True
    error_sampling_power = 5.0        # 幂次越大，越偏向难动作
    error_sampling_threshold = 0.15   # > 15cm 误差才算"困难"
```

---

### 10.6 FALCON 风格的力扰动训练（提升鲁棒性）

配置中已有力扰动课程学习框架（默认关闭），可用于训练抗推策略：

```python
class env(G1MimicStuFutureCfg.env):
    enable_force_curriculum = True
    class force_curriculum:
        force_apply_links = ['left_rubber_hand', 'right_rubber_hand']
        apply_force_x_range = [-40.0, 40.0]  # N
        apply_force_y_range = [-40.0, 40.0]
        apply_force_z_range = [-50.0, 5.0]
        randomize_force_duration = [10, 50]  # 控制步数
```

---

### 10.7 调整 Reward 权重

对于特定任务（如纯上肢操作），建议调整权重以平衡跟踪精度与运动质量：

```python
class rewards(G1MimicPrivCfg.rewards):
    class scales:
        # 上肢操作：提升手部跟踪权重
        tracking_keybody_pos = 5.0        # 默认 2.0
        tracking_joint_dof = 1.0          # 适当降低全身权重
        tracking_root_linear_vel = 0.5    # 降低移动跟踪权重

        # 稳定性惩罚不变或适当增大
        feet_slip = -0.2
        action_rate = -0.1
```

---

### 10.8 架构级改进思路

| 方向 | 现状 | 改进思路 |
|------|------|----------|
| **未来帧编码** | 简单 MLP flatten | Transformer 跨帧注意力（已有 `attention_heads` 参数预留） |
| **历史编码** | Conv1D（固定感受野） | Mamba/S4 等状态空间模型（更长历史依赖） |
| **MoE 路由** | 软路由（加权求和） | Top-K 硬路由（减少干扰，但需 load balancing） |
| **Teacher 设计** | 单一特权 Teacher | Ensemble Teacher（多个不同初始化，降低蒸馏误差） |
| **课程调度** | 基于完成率（每动作独立） | 跨动作难度聚类，批量调整相似动作 |
| **Sim2Real** | Domain Rand（参数随机化） | 加入系统辨识模块，在线估计真机参数 |
| **动作数据** | 离线重定向 pkl | 在线遥操作直接生成参考帧（`server_motion_lib.py` 已支持） |

---

*文档版本：基于代码库 commit `d5c7108`，分析时间 2026-04。*
