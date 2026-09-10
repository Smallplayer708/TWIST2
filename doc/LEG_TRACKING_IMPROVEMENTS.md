# TWIST2 腿部抖动与跟踪精度改进方案汇总

> 目标问题（实时遥操作仿真，sim2sim / OrcaLab bridge）：
> 1. 运动过程中腿部存在高频抖动；
> 2. 腿部对参考动作的跟踪精度不足（尤其脚踝/脚部）。
>
> 本文把可行的改进方案分为「不需要重训」和「需要重训」两大类，逐个说明
> **原理（为什么有用）** 与 **实现方法**，最后给出推荐路径。
> 所有行号引用基于当前代码库。

---

## 0. 约束与根因

### 0.0 关键约束：实时遥操作（因果性）

本场景是**实时遥操作**：PICO 动捕 → GMR 逐帧 retarget → `mimic_obs`(35) → Redis →
policy → 仿真机器人。参考动作是**实时产生**的，推理时只有**当前这一帧**，
**未来轨迹不存在**。

由此产生的硬约束：

- **任何需要"未来参考动作"的方案都不可行**（部署 teacher、喂真实 future）。
- 已核实：部署的 `g1_stu_future` student 本身就是**因果的**——
  `g1_mimic_future_config.py:6-21` 里 `TAR_MOTION_STEPS_FUTURE = [0]`、
  `tar_motion_steps = [0]`（两个槽都是当前帧），所以部署端
  `future_obs = action_mimic.copy()` 与训练**一致，不是 bug**。
  作者把 `[1,2,3,4,5]`/`[5,10,...]` 等"真未来"选项注释掉，正是为了适配实时。
- 因此本文中"喂真实未来"和"部署 teacher"两类方案已**删除/标注为不适用**。

### 0.1 症状

- **抖动**：腿部关节出现与参考动作无关的高频振荡。
- **跟踪差**：参考要求抬脚/迈步时腿部姿态跟不上，脚踝/脚部偏差大。

### 0.2 根因（代码证据）

1. **student 蒸馏丢失精度**：student 是 teacher 的函数近似，只用当前帧 + 10 帧历史，
   抖动是蒸馏的常见副作用。这是因果策略的固有代价（未来不可得）。
2. **平滑正则偏弱**：shipped config 已把 `action_rate` 提到 `-0.05`
   （`g1_mimic_future_config.py:144`）、`tracking_joint_dof=2.0`，但 `dof_acc=-5e-8`、
   `ankle_dof_acc=-5e-8` 仍极弱，压不住高频抖动。
3. **腿部 PD 增益偏低（sim2sim 特有）**：`server_low_level_g1_sim.py:141-151` 注释实测
   MuJoCo 软接触下腿部跟踪 RMSE 在 gain=1.0 时 6.45°，gain=2.0 时 4.38°；
   `sim2sim.sh:19` 当前用的是 `--leg_pd_gain 1.0`。
   （OrcaLab bridge 是另一套代码，无此参数，见方案 A。）
4. **脚踝被主动放松**：teacher 的 `dof_err_w` 中脚踝只有 `0.1`
   （`g1_mimic_config.py:39-44`）；部署时脚踝速度被置零
   （`server_low_level_g1_sim.py:262`）。
5. **参考噪声**：GMR 增量 IK 每帧带噪声（PICO 光学腿估计），直接喂策略会放大成腿抖
   （`xrobot_teleop_to_robot_w_hand.py:494-496` 注释）。

---

## 1. 方案总览

| # | 方案 | 治抖动 | 治跟踪 | 是否重训 | 改动量 | 实时适用 |
|---|---|---|---|---|---|---|
| A | 提高腿部 PD 增益 | ★★ | ★★★ | 否 | 一个 flag | ✅（sim2sim；OrcaLab 需另改） |
| C | PD 目标低通滤波 | ★★★ | - | 否 | 小（1 文件） | ✅ |
| D | 强化脚部奖励+平滑正则+提脚踝权重 | ★★ | ★★ | 是 | 小（config） | ✅ |
| F | 不对称蒸馏（privilege distillation） | ★★ | ★★ | 是 | 中（两阶段） | ✅ |
| G | Latent-action（VAE/VQ） | ★★★ | ★★ | 是 | 大 | ✅ |
| H | Residual RL（学残差） | ★★ | ★★ | 是 | 中 | ✅ |
| I | 分层 RL + WBC/QP | ★★★ | ★★★ | 是 | 很大 | ✅ |
| J | Diffusion Policy student（iDP3） | ★★ | ★ | 是 | 大 | ✅ |
| K | AMP + PPO | ✗（可能加重） | - | 是 | 大 | ✅（但不推荐） |
| ~~B~~ | ~~修复 future_obs~~ | — | — | — | — | ❌ 已删除（伪问题） |
| ~~E~~ | ~~免蒸馏部署 teacher~~ | — | — | — | — | ❌ 需未来，实时不可行 |

> 已删除的 B（喂真实 future）：已核实策略本身是因果的，`future_obs` 不是 bug，无需修。
> 已标注的 E（部署 teacher）：teacher 需要未来 20 步，实时遥操作不可用。

---

## 2. 不需要重训的方案

### A. 提高腿部 PD 增益（`--leg_pd_gain`）

**原理**

控制器是 PD：`τ = kp·(q* − q) − kd·q̇`。存在重力、接触力等持续干扰时，
位置跟踪稳态残差近似与 `1/kp` 成正比——`kp` 越大，实际到达角度越接近目标。
MuJoCo 软接触让脚/踝持续承受干扰力，`kp` 偏低时腿部角度跟踪自然差。
适当同时提高 `kd` 可保持阻尼、避免振荡。

**为什么对本问题有用**

直接减小腿部跟踪残差；更"硬"的跟踪通常也意味着更小的来回漂移。

**实现**

- **sim2sim**：`sim2sim.sh` 里 `--leg_pd_gain 1.0 → 2.0`
  （该旋钮只缩放腿部 12 关节 stiffness/damping，`server_low_level_g1_sim.py:144-147`）。
- **OrcaLab bridge**：无此参数。bridge 已有 `_override_pd_gains()`
  （`bridge_twist2_to_orcalab.py:794-814`）把 29 个 body 执行器覆盖成 TWIST2 训练值，
  但无"腿 ×2"乘子。需在函数里对腿 12 关节（idx 0-11）的 kp/kv 额外乘一个增益。

**注意**

`kp` 过大会激发高频振荡，需与 `kd` 一起调；sim2sim 建议 1.5~2.5 扫。
OrcaLab 是 MuJoCo position actuator，软接触补偿量可能与 sim2sim 不同，需单独扫。

---

### C. PD 目标低通滤波（EMA 平滑）

**原理**

抖动是**高频**分量。policy 输出的动作序列里，有用控制信号集中在低频
（人体运动 < 10Hz），蒸馏噪声/观测扰动集中在高频。对 PD 目标做一阶低通：
```
target_t = α·target_{t-1} + (1-α)·raw_t
```
只保留低频、衰减高频，直接抑制抖动。代价是相位滞后——截止频率越低滞后越大。

**为什么对本问题有用**

零成本压抖动，且不改变训练好的策略，**实时可用**（纯因果）。

**实现**

在 `server_low_level_g1_sim.py` 的 `raw_action`/`pd_target` 计算后，对腿部 12 关节做 EMA：
```python
self.last_pd_target[:12] = alpha * self.last_pd_target[:12] + (1 - alpha) * pd_target[:12]
```

**延迟量化（policy 50Hz，20ms/帧）**

| α（上一帧权重） | 截止频率 fc | 时间常数 τ | 影响 |
|---|---|---|---|
| 0.5 | ~8 Hz | ~29 ms | 安全 |
| 0.7 | ~3.4 Hz | ~45 ms | 中等，注意摆腿响应 |
| 0.85 | ~1.4 Hz | ~95 ms | 危险，拖慢平衡/抬腿，可能失稳 |

**注意**

- 从 `α=0.5~0.7` 起步，别一上来 0.85。
- 更好替代：**rate limiter**（只削尖峰、不整体延迟）或 **one-euro filter**
  （低速强滤波、高速弱滤波，延迟更小）。

---

## 3. 需要重训的方案

### D. 强化脚部奖励 + 平滑正则 + 提高脚踝权重

**原理**

- **平滑正则**：`action_rate`（相邻步动作差）、`dof_acc`（关节加速度）、
  `ankle_dof_acc/ankle_dof_vel` 以负权重惩罚动作的高频变化，直接压低抖动。
- **脚部追踪**：`dof_err_w` 是每个关节在 `tracking_joint_dof` 里的权重
  （`humanoid_mimic.py:685-697`）。脚踝 0.1 太低，脚踝角度基本不被追。
  提高它、启用 `feet_air_time / feet_clearance / feet_contact_number`
  （函数已写在 `humanoid.py`，只是没启用）能改善摆动期抬脚高度与接触时机。

**为什么对本问题有用**

最"对症"的重训改动：平滑正则治抖动，脚踝权重+脚部奖励治跟踪。
**实时可用**（纯因果，不引入未来）。

**实现**

在 `g1_mimic_future_config.py`（当前 `g1_stu_future` 任务的实际 config）基础上改：
1. 在 `rewards.scales` 增加/加强：
   ```python
   action_rate = -0.1       # 原 -0.05，继续加强
   dof_acc = -5e-7          # 原 -5e-8
   ankle_dof_acc = -1e-6
   ankle_dof_vel = -2e-4
   feet_air_time = 5.0
   feet_clearance = 2.0
   feet_contact_number = 1.0
   ```
2. 在 teacher config（`g1_mimic_config.py`）提高脚踝 `dof_err_w`：`0.1 → 0.5~1.0`。
3. **保持 `TAR_MOTION_STEPS_FUTURE = [0]` 不动**（保持因果，适配实时）。

**注意**

shipped config 已经把 `action_rate` 提到 `-0.05`、`tracking_joint_dof=2.0`，
说明作者做过一轮抗抖调参，在其基础上继续加码即可。

---

### F. 不对称蒸馏（Asymmetric Actor-Critic / privilege distillation）

**原理**

普通行为克隆（DAgger 匹配 teacher 的 action）只监督"输出动作"，丢失了 teacher
内部对状态的**价值**信息。不对称 AC 让：
- student 的 **actor** 只看 proprio（部署可见、因果的观测）；
- student 的 **critic** 看 privileged（含完整未来 motion + 基座速度等，仅训练时可见）。

critic 的价值梯度引导 actor 学出"在有限/有噪观测下仍逼近 teacher 行为"的策略，
而不是死记 action。比纯 BC 更抗观测噪声、蒸馏动作更平滑。

**为什么对本问题有用**

不改变部署接口（student 仍只吃 proprio、因果），同时降低蒸馏精度损失与抖动。

**实现（两阶段）**

1. 训 teacher（`g1_priv_mimic`）。
2. 训 student，换成不对称 AC。仓库已有半成品：
   `rsl_rl/rsl_rl/modules/actor_critic_mimic.py` 的 `priv_encoder` + `priv_reg_coef`。
   补全 critic 吃 privileged、actor 吃 proprio 的前向与损失即可。

**注意**

需理解 `n_priv_latent / n_priv_mimic_obs / n_priv_info` 的观测切分
（`g1_mimic_distill_config.py:22-29`）。

---

### G. Latent-action（VAE / VQ-VAE 隐空间动作）

**原理**

把参考动作序列编码到**低维隐空间**（VAE 高斯连续 / VQ 离散码本）。
policy 在隐空间回归/采样，隐变量经 decoder 展开成完整关节目标。
- **低维**：策略拟合难度下降；
- **平滑先验**：隐变量小扰动经 decoder 展开成"全身协调"动作，
  天然抑制逐关节高频抖动。

**为什么对本问题有用**

专门针对"动作抖动"的一类方法，纯因果、实时可用。

**实现**

1. 用参考动作数据训练动作 VAE（encoder: 动作→隐变量；decoder: 隐变量→关节目标）。
2. policy 输出改为隐变量，替换 `num_actions` 语义。
3. 训练/部署都接 decoder。

**注意**

工程量大，decoder 有额外推理开销；需权衡隐空间维度与重建精度。

---

### H. Residual RL（学参考 PD 目标的残差）

**原理**

让策略输出"对参考 PD 目标的**残差修正**"，而非完整目标：
```
q*_final = q*_ref + Δq_policy
```
参考轨迹 `q*_ref` 本身平滑，策略只需输出**小幅度**残差，噪声被限制在小范围 → 抖动大幅降低；
参考轨迹提供强先验，跟踪不丢。

**为什么对本问题有用**

降低策略输出幅度 → 压制抖动，保留跟踪。纯因果、实时可用。

**实现**

1. env 里 action 语义从"绝对 PD 目标"改为"相对参考 dof_pos 的残差"。
2. 训练时 `pd_target = ref_dof_pos + action_scale * action`。
3. 部署时同样加回参考 dof_pos。

**注意**

需改 env 的 action 构建与部署的 PD 目标计算两处，保持一致。

---

### I. 分层 RL + Whole-Body Control（WBC / QP）

**原理**

抖动和跟踪误差很多来自"用低频 RL 输出直接驱动高频关节力矩"——中间隔着 PD，
模型误差大、无接触/动力学约束。WBC 用 QP 在**任务空间**（COM、脚位、动量、接触力）
求解满足动力学方程与摩擦锥/接触约束的关节力矩，跟踪精度和顺滑度远高于 RL 直出 PD 目标。
RL 只负责发高层目标（COM 轨迹、脚步位置、接触时序）。

**为什么对本问题有用**

真机人形主流方案，跟踪精度最高、抖动最小。纯因果、实时可用。

**实现**

1. 建立 G1 动力学模型（URDF → 惯量/质量矩阵）。
2. 写 WBC/QP 求解器（任务优先级、接触约束）。
3. RL 输出改为高层目标，替换现有 29 维 PD 目标动作空间。

**注意**

工作量非常大（仓库里只留了 `g1_wbc` 路径名，相当于从零搭）；
收益最大但周期最长。与现有 RL 的输出语义冲突（WBC 会篡改策略目标），需谨慎。

---

### J. Diffusion Policy student（iDP3）

**原理**

扩散模型学习动作的**分布**而非单点回归，生成动作更平滑、能表达多模态。
作为蒸馏 student 时，扩散模型能更好拟合 teacher 的动作分布，减少"回归平均化"导致的抖动。

**为什么对本问题有用**

可能改善抖动（动作更平滑），对"逐帧跟踪精度"帮助有限。纯因果、实时可用。

**实现**

1. 把 student 的动作头换成扩散去噪头。
2. 用 teacher 的动作作为蒸馏目标训练扩散 student（两阶段）。
3. 部署推理需多步去噪，注意 50Hz 控制周期的实时性。

**注意**

推理慢是主要瓶颈，实时 50Hz 下较紧张。

---

### K. AMP + PPO（不推荐用于本问题）

**原理**

AMP 用判别器区分"专家状态分布"与"policy rollout 分布"，奖励 `-log(1-D(s))`。
它提供**相位无关的风格先验**，让动作"看起来像专家"，但：
- 不直接惩罚抖动，也不直接提高逐帧跟踪精度；
- 判别器是额外的高频非光滑信号，对抗训练**往往加剧抖动**。

**为什么对本问题无用/有害**

抖动和跟踪精度属于"平滑性 + 追踪准确性"问题，不是"风格"问题。

**实现（如仍要）**

判别器网络 + AMP 算法（继承 PPO 加 `update_discriminator`）+ 专家数据
（可从现有 `.pkl` 提取 `dof_pos/dof_vel/root_vel/root_ang_vel` 作为判别器观测）
+ env 加 `_reward_amp`。纯因果，实时可用。

---

## 4. 推荐路径（实时遥操作约束下）

1. **先做不需要重训的（A + C）**：`--leg_pd_gain 2.0`（sim2sim）+ 腿部 PD 目标低通
   （α=0.5~0.7）或 one-euro/rate limiter。零训练成本，实时可用，先看能否解决。
2. **若还不够，做 D**：在现有 `g1_mimic_future_config.py` 基础上加强平滑正则 +
   提高脚踝权重，重训（保持 `TAR_MOTION_STEPS_FUTURE=[0]` 不变）。
3. **长期/要根治抖动**：F（不对称蒸馏）或 G（latent-action）或 H（residual）；
   要根治跟踪精度上 I（WBC）。
4. **明确排除**：部署 teacher（需未来）、喂真实 future（需未来）、
   AMP（不治抖动）。

> 关键判断：抖动 + 跟踪差主要来自**蒸馏信息损失 + 平滑正则弱 + PD 增益不足 + 参考噪声**，
> 而非"动作风格"。实时遥操作下只能在**因果**框架内优化，
> 优先走输出侧平滑（A/C）与重训平滑正则（D），而不是 AMP 或任何依赖未来的方案。
