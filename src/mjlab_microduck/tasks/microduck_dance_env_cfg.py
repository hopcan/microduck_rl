"""Microduck DANCE 任务 —— 在平地上跟踪一段「动作序列」（编舞）。

这个环境让策略跟踪一段**预定义的身体位姿序列**（而不是随机采样的位姿，也不
是原来那个按节拍程序生成的动作）。核心机制是 mdp.py 里新增的
``SequencePoseCommand``：它把一列关键帧（keyframe）按时间轴连续播放，每个控制
步把「当前参考位姿」写进 6D 的 ``body_pose`` 命令槽，再由 ``body_pose_tracking_6d``
这个奖励去追踪它。

关键点（也是「动作如何传入」的答案）：
  - 13D 命令块布局不变：``[twist(3), head_pose(4), body_pose(6)]``，61D 观测契约
    和运行时热插拔策略照旧成立。
  - ``body_pose`` 槽（6D）就是「动作序列」的载体：``[x, y, z, roll, pitch, yaw]``，
    全是相对名义站姿的增量。
  - 序列本身在文件顶部的 ``DANCE_SEQUENCE`` 常量里定义，是一列
    ``(持续时间秒, 6D 位姿)`` 的关键帧，作为 ``SequencePoseCommandCfg.keyframes`` 传入。
  - 相邻关键帧之间用 ``interp_s`` 秒做**匀速线性插值**，参考位姿一直在动 —— 这正是
    让策略去「跟踪」动作、而不是在关键帧上「蹲点」的关键（见 AGENTS.md 的
    no-keyframe-trajectory 教训：固定路点会让策略卡在路点不动，只有移动的参考目标
    才能训练出跟踪能力）。

训练 vs 部署：
  - 训练时 ``random_phase_offset=True``：每个 env 从一个随机相位开始，让单个 batch
    每一步都覆盖整段序列（策略必须会跟踪序列里的**任意**一段）。
  - 部署时把它设成 ``False`` 从关键帧 0 开始播放，或由运行时按同样的插值规则逐帧
    写 6D 参考位姿进 ``body_pose`` 槽。

三个动作全部由 6D 身体位姿解析式表达（无需原来的 beat 相位）：
  - 下蹲        —— 躯干 z 下降 2.5 cm
  - 重心左右移  —— 躯干 roll ±8°
  - 前倾        —— 躯干 pitch +6°

基于 ``make_microduck_velocity_env_cfg()`` 构建，所以 DR / 观测噪声 / 延迟 / NaN 加固 /
BAM 摩擦等全部白拿；下面只重调或删除步行相关项：
  - twist(3) 保留速度语义但范围极小（本任务是原地舞，槽位保持存活）
  - head_pose(4) 保持小范围（不扩宽 curriculum），head_pose_tracking 降权 —— 头部
    只被轻轻拉向近中立命令，不抢身体序列的主导权
  - 删掉步态项（air_time / foot_clearance / foot_swing_height）：双脚全程贴地，
    由 no_stepping 强制
  - 运动阻隔正则调低（舞蹈是动态任务）：body_ang_vel -0.05→-0.01、
    angular_momentum -0.02→-0.005
  - pose 正则放松（1.0→0.25、std 放宽）：下蹲/侧倾必须让腿离开 HOME，不能被紧
    站姿拉回来
  - 主目标 body_pose_tracking 权重 0→2.5（追踪序列），nominal_height 用实测站高
    STAND_Z=0.115，z_std=1cm、angle_std=6° 收紧到真正去够到目标位姿
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import RewardTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    NUM_STEPS_PER_ENV,
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG

ENABLE_SYMMETRY = False

# ─────────────────────────────────────────────────────────────────────────────
# 动作序列（编舞）。每个元素 = (持续时间秒, 6D 身体位姿 delta)。
# 6D = [x, y, z, roll, pitch, yaw]，全部是相对名义站姿的增量：
#   x, y   —— 相对出生点的平移（原地舞，恒为 0）
#   z      —— 相对 DANCE_NOMINAL_HEIGHT 的高度增量（下蹲 = 负值）
#   roll   —— 躯干侧倾（重心左右移用）
#   pitch  —— 躯干前倾/后仰
#   yaw    —— 躯干朝向（原地舞，恒为 0）
# ─────────────────────────────────────────────────────────────────────────────
DANCE_SEQUENCE = (
    # (时长 s, (dx,  dy,  dz,     roll, pitch, yaw))
    (1.0, (0.0, 0.0,  0.000,  0.00, 0.00, 0.0)),  # 中立站姿
    (1.0, (0.0, 0.0, -0.025,  0.00, 0.00, 0.0)),  # 下蹲 2.5 cm
    (1.0, (0.0, 0.0, -0.025,  0.14, 0.00, 0.0)),  # 下蹲 + 右倾 ~8°
    (1.0, (0.0, 0.0, -0.025, -0.14, 0.00, 0.0)),  # 下蹲 + 左倾 ~8°
    (1.0, (0.0, 0.0,  0.000,  0.00, 0.10, 0.0)),  # 站起 + 前倾 ~6°
)

# 相邻关键帧之间的匀速插值时间（秒）。参考位姿一直在动 → 策略学「跟踪」而非「蹲点」。
DANCE_SEQ_INTERP_S = 0.3

# 本任务不驱动的槽位的保活采样范围（死权重守卫：命令槽从第 0 步就非零，输入神经元不僵死）。
TWIST_TINY_LIN = 0.02   # m/s
TWIST_TINY_ANG = 0.05   # rad/s

# 放宽的 pose 正则 std（站立档 = 基础步行的 std，再打开 hip_roll，好让重心侧倾的
# 髋关节参考不被紧站姿顶回来）。
DANCE_STD_POSE = {
    r".*hip_yaw.*": 0.3,
    r".*hip_roll.*": 0.15,
    r".*hip_pitch.*": 0.4,
    r".*knee.*": 0.4,
    r".*ankle.*": 0.25,
}

# STAND_Z —— 实测站立躯干 z（与 standup/ball_kick 同一常量）。
DANCE_NOMINAL_HEIGHT = 0.115


def make_microduck_dance_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """创建 Microduck 动作序列跟踪（编舞）环境配置。"""

    cfg = make_microduck_velocity_env_cfg(play=play)

    # === 命令 ===
    # twist：保留速度语义，范围极小（死权重守卫）。基础工厂已经 deepcopy 命令并包成
    # VelocityCommandCommandOnlyCfg，这些范围只对本 cfg 生效。
    twist = cfg.commands["twist"]
    twist.ranges.lin_vel_x = (-TWIST_TINY_LIN, TWIST_TINY_LIN)
    twist.ranges.lin_vel_y = (-TWIST_TINY_LIN, TWIST_TINY_LIN)
    twist.ranges.ang_vel_z = (-TWIST_TINY_ANG, TWIST_TINY_ANG)

    # head_pose：保留基础工厂的小范围；下方删除扩宽 curriculum。头部参考（近中立）
    # 由 head_pose_tracking 轻量追踪，不再有单独的 head_bob 关节参考。
    # （cfg.commands["head_pose"] 不动。）

    # body_pose：动作序列命令 —— 把 6D 槽从「随机采样」换成「按时间轴播放预定义序列」。
    cfg.commands["body_pose"] = microduck_mdp.SequencePoseCommandCfg(
        keyframes=DANCE_SEQUENCE,
        interp_s=DANCE_SEQ_INTERP_S,
        loop=True,               # 序列播完回到开头循环
        random_phase_offset=True,  # 训练多样性：每个 env 随机相位起播，batch 覆盖整段序列
    )

    # === 奖励 ===
    # 步态项关闭 —— 三个动作双脚都贴地。
    for name in ("air_time", "foot_clearance", "foot_swing_height"):
        cfg.rewards.pop(name, None)

    # 原地待命：twist 命令≈0，速度追踪即「站住别动」目标，给个适中权重。
    cfg.rewards["track_linear_velocity"].weight = 0.5
    cfg.rewards["track_angular_velocity"].weight = 0.5

    # upright 仍有意义（别摔倒），但比步行弱：重心侧倾动作本身就要 roll ±8°，
    # 不能被罚没了（body_pose_tracking 会为命中 roll 付 2.5）。
    cfg.rewards["upright"].weight = 1.0

    # 运动阻隔正则放低（动态任务）：弹跳/侧倾本来就要求躯干有角运动。
    cfg.rewards["body_ang_vel"].weight = -0.01
    cfg.rewards["angular_momentum"].weight = -0.005

    # pose 正则放松：让序列参考位姿拥有运动的支配权。
    cfg.rewards["pose"].weight = 0.25
    cfg.rewards["pose"].params["std_standing"] = DANCE_STD_POSE
    cfg.rewards["pose"].params["std_walking"] = DANCE_STD_POSE
    cfg.rewards["pose"].params["std_running"] = DANCE_STD_POSE

    # 头部位姿槽：保持存活、近中立；权重压低，不抢身体序列主导权。
    cfg.rewards["head_pose_tracking"].weight = 0.1

    # 主目标：追踪动作序列。基础工厂已注册 body_pose_tracking（body_pose_tracking_6d，
    # 权重 0 禁用），这里把权重提上来、并把 nominal_height 换成实测站高、收紧 z/角度
    # 容差，让策略真正去够到每个关键帧位姿。
    cfg.rewards["body_pose_tracking"].weight = 2.5
    cfg.rewards["body_pose_tracking"].params["nominal_height"] = DANCE_NOMINAL_HEIGHT
    cfg.rewards["body_pose_tracking"].params["z_std"] = 0.010
    cfg.rewards["body_pose_tracking"].params["angle_std"] = math.radians(6)
    # xy_std 保持基础的 0.05（dx=dy=0，原地舞，容差宽松）。

    # 双脚贴地（自否成本 ≥ 0 → 负权重）。
    cfg.rewards["no_stepping"] = RewardTermCfg(
        func=microduck_mdp.no_stepping_penalty,
        weight=-0.5,
        params={"sensor_name": "feet_ground_contact"},
    )

    # === Curriculum ===
    # 删除针对步行语义的课程：
    #   standing_envs   —— 斜坡 twist rel_standing_envs（步行专属）
    #   head_pose_range —— 扩宽头部命令（舞蹈保持小范围）
    #   body_pose_range —— 会把 UniformPoseCommand 的 ranges 写到 SequencePoseCommand
    #                      上（无意义）
    for name in ("standing_envs", "head_pose_range", "body_pose_range"):
        cfg.curriculum.pop(name, None)

    # 平滑惩罚温和上升、封顶 -0.3（vs 步行 -1.0）：1.5–2.3 Hz 的节奏动作必须负担得起。
    cfg.curriculum["action_rate_weight"].params["weight_stages"] = [
        {"step": 0, "weight": -0.05},
        {"step": 500 * NUM_STEPS_PER_ENV, "weight": -0.1},
        {"step": 1000 * NUM_STEPS_PER_ENV, "weight": -0.2},
        {"step": 1500 * NUM_STEPS_PER_ENV, "weight": -0.3},
    ]
    # 保留继承项：com_range、head_com_range、head_pose_bias_weight。
    # head_pose_bias 度量的是相对近中立 head_pose 命令的误差；其 1s EMA 让头部
    # 振荡互相抵消，只罚直流下垂 —— 正是本意。

    return cfg


MicroduckDanceRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="dance",  # 目录名
    run_name="dance",         # wandb 里 <datetime>_dance 的后缀
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=50_000,
)
