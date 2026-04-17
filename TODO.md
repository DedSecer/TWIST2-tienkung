# TODO: TWIST2-tienkung 仓库架构、内部训练方法与复现说明

本文档基于当前仓库代码状态整理，目标不是重复 README，而是把当前仓库真正可运行的训练架构、模块关系、内部训练方法和复现路径讲清楚。

适用范围：

- G1 低层控制器训练
- Tienkung 低层控制器训练入口
- ONNX 导出与 sim2sim/sim2real 部署链路
- teleop 与动作流在整体系统中的位置

不覆盖内容：

- GMR 的完整重定向算法细节
- 真机网络、VR、相机和颈部服务的全部硬件布线细节
- 单独高层策略仓库

---

## 1. 仓库整体目标

这个仓库的核心作用，是把“人类/重定向动作数据”训练成可部署的人形机器人低层控制策略，并将该策略用于：

1. 仿真中的动作跟踪和运动模仿
2. ONNX 导出后的低层实时控制
3. teleop 动作流与真机执行的桥接

当前代码主线不是端到端高层策略学习，而是“低层运动模仿控制器”的训练与部署。训练产物最终是一个 student policy，导出为 ONNX 后由 deploy_real 下的低层控制服务使用。

---

## 2. 仓库架构总览

从功能上，可以把仓库分为五层。

### 2.1 顶层脚本层

顶层 shell 脚本负责环境激活、切换到正确目录并调用 Python 入口。常用脚本如下：

- train_stage1_amp.sh：训练 Stage1 teacher/privileged AMP 策略
- train_stage2.sh：训练 G1 student future 策略
- train_tienkung.sh：训练 Tienkung student future 策略
- to_onnx.sh：把训练好的 student checkpoint 导出为 ONNX
- sim2sim.sh / sim2sim_tienkung.sh：仿真部署测试
- sim2real.sh：真机低层控制入口
- teleop.sh：遥操作入口
- data_record.sh：数据录制入口

这些脚本本身都很薄，真正逻辑都在 legged_gym 和 deploy_real 里。

### 2.2 训练框架层

训练主干由以下三个子模块组成：

- legged_gym：环境、配置、任务注册、训练脚本入口
- rsl_rl：PPO、DAggerPPO、AMP 判别器、policy/network、runner
- pose：动作数据与姿态处理依赖

训练入口实际是：

```text
train_stage*.sh
  -> legged_gym/legged_gym/scripts/train.py
  -> task_registry.make_env()
  -> task_registry.make_alg_runner()
  -> runner.learn()
```

### 2.3 动作数据层

动作数据由 GMR 或其他预处理模块生成，最终表现为：

- motion_data_configs/*.yaml：动作清单和采样权重
- 外部 root_path 下的大量 .pkl 动作文件

训练时由 MotionLib 加载、采样、插值，并在每个仿真步提供参考根部姿态、关节角、刚体位置等信息。

### 2.4 部署层

训练好的 student checkpoint 会被导出为 ONNX，然后由 deploy_real 中的低层服务加载，形成：

```text
motion server / teleop / redis
  -> low-level controller server
  -> ONNX policy inference
  -> robot PD/low-level execution
```

### 2.5 机器人对象层

当前仓库里至少有两条机器人配置线：

- G1
- Tienkung

两者共享相同训练范式，但各自有独立 env/config 文件和顶层训练脚本。

---

## 3. 训练架构的核心思想

当前仓库的低层训练遵循“参考动作模仿 + privileged teacher + student distillation”的架构。

可以概括为：

```text
重定向动作数据
  -> MotionLib 提供参考轨迹
  -> Stage1 训练 teacher（可带 AMP）
  -> Stage2 训练 student（PPO + teacher KL 蒸馏）
  -> 导出 student 为 ONNX
  -> sim2sim / sim2real 部署
```

这里最重要的是 teacher 和 student 的观测空间不一样。

### 3.1 Teacher 的职责

teacher 使用 privileged observation，也就是训练时能看到更完整、更未来、更难以部署的参考信息。它的任务是学出一个质量尽可能高的动作跟踪策略。

### 3.2 Student 的职责

student 使用可部署观测，只看当前帧、历史帧和可选的少量 future motion 信息。它通过 PPO 在环境里直接优化，同时通过 teacher 的动作分布做 KL 蒸馏，逼近 teacher 的行为。

### 3.3 为什么需要两阶段

直接让 student 用贫信息观测学习复杂动作跟踪，收敛会慢、稳定性差。先用 privileged teacher 学到“更容易”的参考动作控制，再把这个能力压缩到 student，是当前仓库的主思路。

---

## 4. 代码级训练入口是怎么组织的

### 4.1 train.py 做了什么

训练入口在 legged_gym/legged_gym/scripts/train.py。

它的主要流程是：

1. 根据命令行参数设置日志目录和 wandb
2. 从 task_registry 创建 env
3. 从 task_registry 创建对应 runner
4. 调用 runner.learn()

也就是说，训练的关键不是 train.py 里写了什么，而是：

- 当前 task 名称是什么
- 这个 task 对应哪个 env class
- 这个 task 对应哪个 train config
- train config 里指定哪个 runner、哪个 algorithm、哪个 policy

### 4.2 task registry 是训练调度中心

任务注册在 legged_gym/legged_gym/envs/__init__.py。

当前关键任务包括：

- g1_priv_mimic：G1 teacher，privileged PPO
- g1_priv_mimic_amp：G1 teacher，privileged AMP PPO
- g1_stu_future：G1 student future，DAggerPPO
- tienkung_priv_mimic：Tienkung teacher
- tienkung_stu_future：Tienkung student future

因此，顶层脚本只是把 task 名传给 train.py，真正训练行为由 task 对应的配置决定。

---

## 5. 环境层内部结构

环境主干位于：

- legged_gym/legged_gym/envs/base/humanoid_mimic.py
- legged_gym/legged_gym/envs/g1/
- legged_gym/legged_gym/envs/tienkung/

其中 humanoid_mimic.py 是核心基类，负责把“仿真状态”和“参考动作状态”对齐。

### 5.1 MotionLib 如何驱动参考轨迹

每个环境实例在 reset 时会采样一个动作序列和一个随机起始时间偏移：

- `_motion_ids`：当前 episode 对应哪段 motion
- `_motion_time_offsets`：从 motion 的哪个时刻开始

后续每一步通过：

```text
motion_time = episode_length * dt + motion_time_offset
```

查询参考轨迹，并得到：

- root_pos
- root_rot
- root_vel
- root_ang_vel
- dof_pos
- dof_vel
- body_pos

这样，环境在每一时刻都知道“机器人当前真实状态”和“这一时刻应该跟踪的参考状态”。

### 5.2 RSI: Reference State Initialization

当前仓库训练时默认启用随机参考状态初始化，也就是 episode 重置时直接把机器人初始化到参考动作某个随机时间点附近，而不是永远从动作开头起步。

这带来两个好处：

1. 训练覆盖整个动作分布，而不是只会起步阶段
2. 减少策略对固定初始姿态的依赖

代码里 reset 时会把机器人 DOF、根部姿态和速度设置到参考状态附近，并对速度使用衰减系数，降低初始不稳定。

### 5.3 观测、奖励、终止都是围绕“跟踪参考动作”构造的

环境不是做导航，也不是做高层决策，而是做 imitation tracking。因此其奖励基本都围绕：

- 关节角跟踪
- 关节速度跟踪
- 根部平移/旋转跟踪
- 根部线速度/角速度跟踪
- 关键 body 的位置跟踪

同时叠加一些正则项和安全项，例如：

- feet_slip
- feet_contact_forces
- feet_stumble
- dof_pos_limits
- dof_torque_limits
- dof_vel
- dof_acc
- action_rate

终止条件则主要包括：

- 姿态偏差过大
- 根部高度异常
- 非法接触
- 动作跟踪失败
- 动作序列结束

---

## 6. Stage1 内部训练方法：Privileged Teacher + AMP

### 6.1 Stage1 的入口

G1 Stage1 当前入口为：

```bash
bash train_stage1_amp.sh <experiment_id> <device>
```

例如：

```bash
bash train_stage1_amp.sh 0411_stage1_amp cuda:0
```

这个脚本会：

1. 激活 conda twist2
2. 进入 legged_gym/legged_gym/scripts
3. 调用 train.py
4. 使用 task `g1_priv_mimic_amp`

### 6.2 Stage1 用的是什么配置

对应配置类是：

- G1MimicPrivAmpCfg
- G1MimicPrivAmpCfgPPO

其核心特点：

- env 开启 AMP
- motion 文件切换到 unitree_g1_retarget_fft.yaml
- algorithm 使用 AMPPPO
- runner 使用 OnPolicyAMPMimicRunner

### 6.3 Teacher 看到什么观测

teacher 采用 privileged observation，包含：

- 多帧未来参考动作信息
- 本体感觉信息
- 额外 privileged 信息

在 G1 teacher 配置里：

- `tar_motion_steps_priv = [1, 5, 10, ..., 95]`
- 共 20 个未来步
- 每步包含根部、关节和关键 body 信息

这意味着 teacher 在做当前动作决策时，已经知道接下来接近 1 秒范围内的参考动作趋势。对于训练来说这很强，但部署时不现实，因此它只适合做 teacher。

### 6.4 Stage1 的损失不是只有 PPO

AMP 阶段在 PPO 基础上加了一个判别器：

- policy 产生的运动特征视为 fake
- 参考动作特征视为 real
- 判别器学习区分 real/fake
- policy 从判别器得到额外 AMP reward

内部逻辑可以概括为：

$$
R_{total} = R_{mimic} + R_{amp}
$$

其中：

- $R_{mimic}$ 是环境中的动作跟踪奖励
- $R_{amp}$ 来自 AMP discriminator 对策略运动风格的打分

判别器使用一个 MLP，输入维度在当前配置里是 130，隐藏层是 `[512, 256]`。训练时还加入 gradient penalty，防止判别器过强或训练不稳定。

### 6.5 Stage1 的实际意义

这一步训练出的 teacher 不只是“把关节误差做小”，而是更偏向学出自然、稳定、接近数据分布的动作风格。后续 student 蒸馏时会从这个 teacher 学习，而不是直接从原始轨迹学动作均值。

---

## 7. Stage2 内部训练方法：Student Future + DAggerPPO

### 7.1 Stage2 的入口

G1 student 入口：

```bash
bash train_stage2.sh <experiment_id> <device> [teacher_exptid]
```

例如：

```bash
bash train_stage2.sh 0411_stage2 cuda:0 0411_stage1_amp
```

当前脚本会把 `teacher_exptid` 传给 train.py，对应配置里的 teacher project 默认是 `g1_priv_mimic`。如果你用的是 AMP teacher，通常要检查训练配置与日志目录是否一致，确保 student 加载到正确的 teacher checkpoint。

Tienkung student 入口为：

```bash
bash train_tienkung.sh <experiment_id> <device> [teacher_exptid]
```

### 7.2 Stage2 对应什么任务

G1 当前默认 student 任务是：

- `g1_stu_future`

对应：

- env class: G1MimicFuture
- cfg: G1MimicStuFutureCfg
- runner: OnPolicyDaggerRunner
- algorithm: DaggerPPO
- policy: ActorCriticFuture

### 7.3 Student 为什么叫 future

因为它的观测不是单一当前帧，而是：

1. 当前帧 mimic obs
2. proprio obs
3. 历史观测序列
4. 可选 future motion obs

当前仓库默认 `tar_motion_steps_future = [0]`，也就是 future 支持已经在网络结构上预留出来了，但默认配置并没有打开一个很长的未来窗口。换句话说，student future 是一种“可扩展结构”，不一定意味着当前默认训练就真的吃很多未来帧。

### 7.4 Student 网络结构

ActorCriticFuture 里包含几块关键编码器：

- MotionEncoder：编码当前 mimic obs
- HistoryEncoder：编码历史观测
- FutureMotionEncoder：编码 future obs
- actor/critic backbone：输出动作分布和价值函数

当前实现中，MotionEncoder 和 HistoryEncoder 主要使用 Linear + Conv1D 压缩时间序列；FutureMotionEncoder 当前更像是一个把 future 展平后的 MLP 编码器。

这说明当前 student 不是简单 MLP，而是显式利用了时序结构。

### 7.5 Stage2 的优化目标

Stage2 不是纯 imitation，也不是纯 PPO，而是两者结合。

从 DaggerPPO 的代码看，核心损失可以概括为：

$$
L = L_{ppo} + \lambda_{dagger} \cdot KL(\pi_{student} \parallel \pi_{teacher})
$$

其中：

- $L_{ppo}$：标准 PPO surrogate loss + value loss - entropy
- $KL(\pi_{student} \parallel \pi_{teacher})$：student 与 teacher 动作分布之间的 KL 散度
- $\lambda_{dagger}$：蒸馏系数，对应配置中的 `dagger_coef`

当前 G1 student future 配置里：

- `dagger_coef = 0.2`
- `dagger_coef_min = 0.1`
- `dagger_coef_anneal_steps = 60000`

也就是说，训练前期更依赖 teacher 约束，后期逐步降低蒸馏强度，让 student 更多依靠自身在环境中的 PPO 优化。

### 7.6 DAggerRunner 如何加载 teacher

OnPolicyDaggerRunner 初始化时会：

1. 根据 teachercfg 构造 teacher network
2. 根据 `teacher_proj_name`、`teacher_experiment_name` 和 `teacher_checkpoint` 定位 teacher checkpoint
3. 如果成功加载，则训练时启用 KL distillation
4. 如果未加载，则退化为 student-only 训练

因此 Stage2 是否真的在做蒸馏，取决于 teacher 路径是否正确，而不是只取决于你有没有传第三个 shell 参数。

---

## 8. 当前 G1 训练配置里最重要的超参数

以下是当前代码里对复现影响最大的关键设置。

### 8.1 仿真与控制频率

- sim dt = 0.002
- decimation = 10
- 有效控制频率 = 50 Hz

这意味着策略每秒输出 50 次动作，而物理仿真内部每次动作会推进 10 个小步。

### 8.2 并行环境数

- G1 teacher 配置默认 `num_envs = 4096`

这是 Isaac Gym 下较典型的大并行配置，对显存和 GPU 要求较高。

### 8.3 episode 长度

环境会根据当前动作库中最长 motion 长度推导最大 episode 长度，而不是写死成一个固定短 horizon。

### 8.4 域随机化

当前 G1 privileged 配置默认开启多项 domain randomization，包括：

- gravity
- friction
- base mass
- base COM
- push robot
- motor strength
- action delay

student future 额外还支持：

- motion domain randomization
- 力扰动 curriculum 结构
- error-aware sampling 结构

其中部分增强项当前默认没有完全开启，但代码路径已经存在。

### 8.5 动作课程学习

MotionLib 采样并不是永远固定均匀的。环境内部维护：

- `motion_difficulty`
- `mean_motion_difficulty`
- 可选 `max_key_body_error`

这意味着动作库采样支持随训练过程进行课程学习和难例采样。

---

## 9. 从动作数据到训练样本的内部流动

这一段是理解“仓库内部训练方法”的关键。

### 9.1 动作文件组织

motion_data_configs 里的 YAML 文件描述：

- root_path：动作数据根目录
- motions：每个动作文件的相对路径、采样权重、描述

当前 G1 student future 默认使用：

- legged_gym/motion_data_configs/unitree_g1_retarget.yaml

Stage1 AMP 使用：

- legged_gym/motion_data_configs/unitree_g1_retarget_fft.yaml

### 9.2 训练时发生了什么

对每个 env：

1. 采样一个 motion id
2. 采样一个起始时间偏移
3. 将机器人 reset 到该参考状态附近
4. 每个 policy step 按当前 episode 时间查询 reference frame
5. 用真实状态和参考状态构造 observation、reward、done
6. PPO 或 DAggerPPO 收集 rollout 并更新

### 9.3 训练信号来自哪里

训练信号至少有三类：

1. mimic 奖励：跟踪参考动作
2. regularization：减少危险动作、平滑控制输出
3. 蒸馏或 AMP：来自 teacher 分布或判别器奖励

这也是当前仓库与“单纯 replay mocap joint target”的最大区别：它不是监督学习，而是强化学习环境中的 reference tracking。

---

## 10. 可复现训练流程

这里给出按当前仓库状态最务实的一条复现路径。

### 10.1 环境准备

当前仓库有两个 Python 环境约定：

- twist2：训练、导出、部署主流程，Python 3.8
- gmr：动作重定向相关，通常 Python 3.10+

如果只复现低层训练，先准备好 twist2 即可。

原则上需要：

1. 安装 Isaac Gym
2. 安装 legged_gym、rsl_rl、pose 等依赖
3. 确认训练脚本能激活 conda twist2
4. 确认 motion YAML 里的 root_path 指向真实存在的数据目录

### 10.2 第一步：准备动作数据

先确认以下文件中的 `root_path` 正确：

- legged_gym/motion_data_configs/unitree_g1_retarget.yaml
- legged_gym/motion_data_configs/unitree_g1_retarget_fft.yaml

如果路径不对，训练虽然能启动，但 MotionLib 无法真正加载动作库。

### 10.3 第二步：训练 Stage1 teacher

建议命令：

```bash
bash train_stage1_amp.sh 0411_stage1_amp cuda:0
```

输出日志通常会在：

```text
legged_gym/logs/g1_priv_mimic_amp/0411_stage1_amp/
```

关注内容包括：

- model_*.pt checkpoint
- wandb 指标
- 平均 episode reward
- AMP reward
- discriminator accuracy

### 10.4 第三步：训练 Stage2 student

建议命令：

```bash
bash train_stage2.sh 0411_stage2 cuda:0 0411_stage1_amp
```

你需要确认 student 配置中的 teacher project/name 与 teacher 日志目录一致，否则会出现“student 在训练，但没有真正加载 teacher”的情况。

输出日志通常位于：

```text
legged_gym/logs/g1_stu_future/0411_stage2/
```

### 10.5 第四步：导出 ONNX

训练完成后，导出 student：

```bash
bash to_onnx.sh /绝对路径/到/model_xxx.pt
```

当前 save_onnx.py 会：

1. 从 checkpoint 推断网络尺寸
2. 加载 normalizer
3. 构造硬件部署包装器
4. 导出同名 .onnx 文件

因此导出脚本默认面向的是 student future 架构，而不是任意旧 policy。

### 10.6 第五步：部署验证

仿真验证：

```bash
bash sim2sim.sh
```

真机验证：

```bash
bash sim2real.sh
```

在完整系统里，teleop、motion server 和 Redis 负责提供高层动作流，低层 student ONNX policy 负责把这些运动目标转成稳定控制输出。

---

## 11. Tienkung 分支与 G1 分支的关系

Tienkung 不是一个完全独立的新训练框架，而是沿用同一套 teacher/student imitation 架构，只是：

- env 类不同
- 配置不同
- 机器人模型和资产不同
- 顶层脚本不同

所以如果你已经理解了 G1 的训练链路，再看 Tienkung，重点只剩三类差异：

1. 资产模型和关节定义
2. 配置超参数
3. 日志目录和任务名

---

## 12. 常见易错点

### 12.1 teacher 路径和 student 配置不一致

最常见的问题不是训练起不来，而是 student 没有加载到正确的 teacher。结果表面看起来正常训练，实际上退化成纯 PPO student。

建议每次 Stage2 启动时都检查控制台是否明确打印了 teacher policy loaded。

### 12.2 动作数据路径失效

motion YAML 中使用的是绝对 root_path。如果你换了机器或目录结构，这一项很容易失效。

### 12.3 以为 Stage2 默认用了很多未来帧

当前 student future 结构支持 future 输入，但默认配置里的 `tar_motion_steps_future = [0]`，并不是一个很长的 future horizon。不要把“模型结构支持 future”误解成“当前训练一定使用了丰富未来信息”。

### 12.4 以为顶层脚本决定训练逻辑

shell 脚本主要决定的是：

- 激活哪个环境
- 使用哪个 task 名
- 传入哪个 exptid

真正决定训练逻辑的是 Python 配置类。

---

## 13. 一句话总结当前仓库的训练方法

当前仓库的低层控制训练，本质上是：

“基于参考动作库的强化学习模仿控制，其中 Stage1 用 privileged teacher 学高质量跟踪策略，Stage2 用 PPO + teacher KL 蒸馏把能力压缩到可部署的 student policy，最终导出为 ONNX 用于 sim2sim 和 sim2real。”

---

## 14. 建议阅读顺序

如果后续要继续改训练系统，建议按下面顺序读代码：

1. legged_gym/legged_gym/scripts/train.py
2. legged_gym/legged_gym/envs/__init__.py
3. legged_gym/legged_gym/gym_utils/task_registry.py
4. legged_gym/legged_gym/envs/base/humanoid_mimic.py
5. legged_gym/legged_gym/envs/g1/g1_mimic_distill_config.py
6. legged_gym/legged_gym/envs/g1/g1_mimic_future_config.py
7. rsl_rl/rsl_rl/runners/on_policy_runner_mimic.py
8. rsl_rl/rsl_rl/runners/on_policy_amp_mimic_runner.py
9. rsl_rl/rsl_rl/runners/on_policy_dagger_runner.py
10. rsl_rl/rsl_rl/algorithms/amp_ppo.py
11. rsl_rl/rsl_rl/algorithms/dagger_ppo.py
12. rsl_rl/rsl_rl/modules/actor_critic_future.py

读完这条链，基本就能完整理解当前仓库的训练架构。