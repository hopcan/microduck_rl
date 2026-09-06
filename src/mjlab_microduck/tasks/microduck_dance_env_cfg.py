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

动作全部由 6D 身体位姿解析式表达（无需原来的 beat 相位），一列 8 个关键帧在
0.4s/拍 下连续播放，核心是「重拍下蹲、轻拍弹起」的弹跳律动（groove）：
  - 弹跳律动     —— 躯干 z 重拍下蹲 4.5cm、轻拍弹回站高（律动主干）
  - 重心左右移   —— 躯干 roll ±17° 左右摇摆（每侧占 2 拍）
  - 前倾/后仰    —— 躯干 pitch +8.6°/-6.9° 前后点头（每方向占 2 拍）

基于 ``make_microduck_velocity_env_cfg()`` 构建，所以 DR / 观测噪声 / 延迟 / NaN 加固 /
BAM 摩擦等全部白拿；下面只重调或删除步行相关项：
  - twist(3) 保留速度语义但范围极小（本任务是原地舞，槽位保持存活）
  - head_pose(4) 同样换成序列命令（SequencePoseCommand），头部跟着同一 0.4s 拍子
    点头/转头/侧倾，head_pose_tracking 权重提到 1.5（身体序列仍为主导）
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
    # (时长 s, (dx, dy, dz,      roll, pitch, yaw))  —— 0.4s/拍：重拍下蹲、轻拍弹起，
    # 左右摇摆 / 前后点头各占 2 拍，8 拍循环 = 3.2s 的连续律动
    (0.4, (0.0, 0.0, -0.045,  0.30,  0.00, 0.0)),   # 蹲 + 右倾 17°
    (0.4, (0.0, 0.0,  0.000,  0.30,  0.00, 0.0)),   # 弹回站高 + 右倾
    (0.4, (0.0, 0.0, -0.045, -0.30,  0.00, 0.0)),   # 蹲 + 左倾 17°
    (0.4, (0.0, 0.0,  0.000, -0.30,  0.00, 0.0)),   # 弹回站高 + 左倾
    (0.4, (0.0, 0.0, -0.045,  0.00,  0.15, 0.0)),   # 蹲 + 前倾 8.6°
    (0.4, (0.0, 0.0,  0.000,  0.00,  0.15, 0.0)),   # 弹回站高 + 前倾
    (0.4, (0.0, 0.0, -0.045,  0.00, -0.12, 0.0)),   # 蹲 + 后仰 6.9°
    (0.4, (0.0, 0.0,  0.000,  0.00, -0.12, 0.0)),   # 弹回站高 + 后仰
)

# 相邻关键帧之间的匀速插值时间（秒）。0.15s 让每拍先稳住重拍、再快速过渡到下一拍，
# 打出清晰的律动点；参考位姿全程在动 → 策略学「跟踪」而非「蹲点」。
DANCE_SEQ_INTERP_S = 0.15

# 头部编舞（4D 增量，顺序 [neck_pitch, head_pitch, head_yaw, head_roll]，相对 HOME）。
# 与 DANCE_SEQUENCE 同 8 拍 × 0.4s = 3.2s：部署时 random_phase_offset=False 两者都从
# 关键帧 0 起播、同相位同步；训练时各自独立随机相位 → batch 覆盖所有相位组合，更鲁棒。
# 方向约定（相对 HOME，需在 sim 里核对正负号）：neck/head_pitch 正 = 更前倾点头；
# head_yaw 正 = 右转；head_roll 正 = 右倾。
HEAD_SEQUENCE = (
    # (时长 s, (neck_pitch, head_pitch, head_yaw, head_roll))
    (0.4, ( 0.25,  0.25,  0.40,  0.15)),  # 点头 + 右转 + 右倾（随身体右倾）
    (0.4, (-0.15, -0.15,  0.40,  0.15)),  # 抬头 + 右转 + 右倾
    (0.4, ( 0.25,  0.25, -0.40, -0.15)),  # 点头 + 左转 + 左倾（随身体左倾）
    (0.4, (-0.15, -0.15, -0.40, -0.15)),  # 抬头 + 左转 + 左倾
    (0.4, ( 0.25,  0.25,  0.00,  0.00)),  # 点头（身体前倾）
    (0.4, (-0.15, -0.15,  0.00,  0.00)),  # 抬头
    (0.4, ( 0.25,  0.25,  0.00,  0.00)),  # 点头（身体后仰）
    (0.4, (-0.15, -0.15,  0.00,  0.00)),  # 抬头
)

# 脚部编舞（6D 增量，顺序 [左x, 左y, 左z, 右x, 右y, 右z]，身体系，相对 HOME 脚位）。
# z 正 = 抬脚（脚远离地面）；x/y 先保持 0（原地踏步，不做前后点步）。与 body/head 同
# 8 拍 × 0.4s = 3.2s 同步。抬脚安排在「弹回站高」相位（身体近名义、耦合最小），且
# 与侧倾同侧对应：右倾抬左脚、左倾抬右脚（承重脚在倾侧，对侧脚空出）。
# nominal 是 HOME 下左右脚在身体系的实测位置（见 foot_pose_tracking_6d 的 nominal）。
DANCE_FOOT_NOMINAL = (-0.0413, 0.0066, -0.11713, 0.0413, -0.00657, -0.11713)
FOOT_SEQUENCE = (
    # (时长 s, (Lx, Ly, Lz, Rx, Ry, Rz))
    (0.4, (0.0, 0.0, 0.00, 0.0, 0.0, 0.00)),  # 双脚贴地（蹲 + 右倾）
    (0.4, (0.0, 0.0, 0.04, 0.0, 0.0, 0.00)),  # 抬左脚 4cm（弹回 + 右倾）
    (0.4, (0.0, 0.0, 0.00, 0.0, 0.0, 0.00)),  # 双脚贴地（蹲 + 左倾）
    (0.4, (0.0, 0.0, 0.00, 0.0, 0.0, 0.04)),  # 抬右脚 4cm（弹回 + 左倾）
    (0.4, (0.0, 0.0, 0.00, 0.0, 0.0, 0.00)),  # 双脚贴地（蹲 + 前倾）
    (0.4, (0.0, 0.0, 0.04, 0.0, 0.0, 0.00)),  # 抬左脚 4cm（弹回 + 前倾）
    (0.4, (0.0, 0.0, 0.00, 0.0, 0.0, 0.00)),  # 双脚贴地（蹲 + 后仰）
    (0.4, (0.0, 0.0, 0.00, 0.0, 0.0, 0.04)),  # 抬右脚 4cm（弹回 + 后仰）
)

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

    # head_pose：头部也走预定义序列（4D 增量，顺序 [neck_pitch, head_pitch, head_yaw,
    # head_roll]），与身体同一 0.4s 拍同步点头/转头/侧倾。SequencePoseCommand 从首个
    # 关键帧推断维度=4，直接喂进 head_pose_tracking。
    cfg.commands["head_pose"] = microduck_mdp.SequencePoseCommandCfg(
        keyframes=HEAD_SEQUENCE,
        interp_s=DANCE_SEQ_INTERP_S,
        loop=True,
        random_phase_offset=True,
    )

    # body_pose：动作序列命令 —— 把 6D 槽从「随机采样」换成「按时间轴播放预定义序列」。
    cfg.commands["body_pose"] = microduck_mdp.SequencePoseCommandCfg(
        keyframes=DANCE_SEQUENCE,
        interp_s=DANCE_SEQ_INTERP_S,
        loop=True,               # 序列播完回到开头循环
        random_phase_offset=True,  # 训练多样性：每个 env 随机相位起播，batch 覆盖整段序列
    )

    # 脚不占命令槽 / obs：脚部编舞由 foot_pose_tracking_6d 直接按 episode 时钟播放
    # FOOT_SEQUENCE（时间驱动，策略只通过奖励感知），观测保持 61D 契约不变。

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

    # 头部走序列：把 head_pose_tracking 从「近中立存活」提到次主目标（身体 2.5 仍是
    # 主导）。std=0.3 让头部真正够到点头/转头目标（基础 0.5 太松，头会随重力下垂）。
    cfg.rewards["head_pose_tracking"].weight = 1.5
    cfg.rewards["head_pose_tracking"].params["std"] = 0.3

    # 主目标：追踪动作序列。基础工厂已注册 body_pose_tracking（body_pose_tracking_6d，
    # 权重 0 禁用），这里把权重提上来、并把 nominal_height 换成实测站高、收紧 z/角度
    # 容差，让策略真正去够到每个关键帧位姿。
    cfg.rewards["body_pose_tracking"].weight = 2.5
    cfg.rewards["body_pose_tracking"].params["nominal_height"] = DANCE_NOMINAL_HEIGHT
    cfg.rewards["body_pose_tracking"].params["z_std"] = 0.010
    cfg.rewards["body_pose_tracking"].params["angle_std"] = math.radians(6)
    # xy_std 保持基础的 0.05（dx=dy=0，原地舞，容差宽松）。

    # 脚部编舞：时间驱动追踪 6D 身体系脚位增量（body 2.5 主导、head 1.5、脚 1.5 次主）。
    # keyframes=FOOT_SEQUENCE 让奖励按 episode 时钟播放编舞（不占命令槽/obs）；nominal 用
    # HOME 下左右脚实测位置；std=2cm 是抬脚 4cm 的一半 —— 抬半脚仍有一半奖励、抬满满分。
    cfg.rewards["foot_pose_tracking"] = RewardTermCfg(
        func=microduck_mdp.foot_pose_tracking_6d,
        weight=1.5,
        params={
            "nominal": DANCE_FOOT_NOMINAL,
            "std": 0.02,
            "keyframes": FOOT_SEQUENCE,
            "interp_s": DANCE_SEQ_INTERP_S,
            "loop": True,
        },
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
