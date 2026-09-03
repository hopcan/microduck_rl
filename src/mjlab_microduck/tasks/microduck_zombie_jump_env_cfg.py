"""Microduck 僵尸跳（双脚并拢向前跳）环境。

在 velocity 行走配方基础上，把奖励重新配比成「双脚同步 + 离地 + 向前」：
  - leg_sync（5.0）压出双脚并拢 —— 左右腿关节是镜像符号（left = -right 才是
    物理上双脚同相），所以 reward = -|left+right|² 在双脚并拢时取到最大 0。
  - air_time（4.0）奖励腾空，threshold_min 抬高到 0.12，只认真跳、不把走路摆腿
    也算进去。
  - track_linear_velocity（4.0）让「向前」成为第一目标，压过原地跳。
  - 命令只保留前进/后退（lin_vel_x ±0.2），禁横向和转向（lin_vel_y、ang_vel_z 均为 0）。
  - 头/身体 pose 命令照 velocity 保留，head_pose_tracking 作为主目标之一。

注意：这是纯仿真任务，未搬 velocity 的 DR / 观测噪声栈（不需要 sim2real）。
只保留了基础事件 + NaN 终结 + critic 传感器 NaN 加固（这些是防崩，不是随机化）。
"""

import math
from copy import deepcopy
import mujoco as _mujoco
import mjlab.terrains as terrain_gen
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlModelCfg,
)
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG

# 观测里重力项用躯干坐标系下的重力投影
USE_PROJECTED_GRAVITY = True

# 是否启用对称镜像损失（symmetry mirror loss）。僵尸跳不是左右对称任务，保持 OFF。
ENABLE_SYMMETRY = False

# 前进/后退速度命令范围（m/s）。只保留这一轴，横向和转向见下面的 twist 配置。
LIN_VEL = 0.2



# pose 奖励的逐关节高斯 std —— 站立时收紧，让机器人贴住 HOME 姿态。
std_standing = {
    # 下半身：站立时收紧，保持 HOME 姿态
    r".*hip_yaw.*": 0.1,
    r".*hip_roll.*": 0.05,  # 0.1→0.06→0.05 —— 锁住 5° 内八站位（脚底平贴），防腿外撇
    r".*hip_pitch.*": 0.15,
    r".*knee.*": 0.15,
    r".*ankle.*": 0.1,
}

# pose 奖励在行走/奔跑档的 std —— 放松，允许大步幅摆动。
std_running = {
    # 下半身
    r".*hip_yaw.*": 0.3,
    r".*hip_roll.*": 0.05,  # 0.1→0.06→0.05 —— 锁住 5° 内八，防腿撇到竖直
    r".*hip_pitch.*": 0.4,
    r".*knee.*": 0.4,
    r".*ankle.*": 0.25,  # 曾为 0.15
}


# head_pose / body_pose 命令的重新采样间隔（秒），在这个区间内随机重采样一次命令。
HEAD_POSE_CMD_RESAMPLE_S = (2.0, 5.0)
BODY_POSE_CMD_RESAMPLE_S = (2.0, 5.0)


def make_microduck_velocity_zombie_jump_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """构建 Microduck 僵尸跳（双脚并拢向前跳）环境配置。"""
    site_names = ["left_foot", "right_foot"]

    # 脚底接触传感器：primary 是两只脚（LEFT 在前、RIGHT 在后，顺序固定），
    # secondary 是地面。track_air_time=True 让它记录每只脚的滞空时间，
    # 这是 air_time 奖励的数据来源。
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(left_foot_collision|right_foot_collision)$",  # LEFT 在前，RIGHT 在后
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    # 自碰撞传感器：检测躯干（trunk_base）子树之间是否碰撞，用来惩罚腿撞躯干。
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    # 每只脚的射线高度传感器：测脚离地面高度， foot_height 观测和
    # foot_clearance / foot_swing_height 两个奖励。
    foot_height_scan_cfg = TerrainHeightSensorCfg(
        name="foot_height_scan",
        frame=tuple(ObjRef(type="site", name=s, entity="robot") for s in site_names),
        pattern=RingPatternCfg.single_ring(radius=0.04, num_samples=2),
        ray_alignment="yaw",
        max_distance=1.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),
        debug_vis=False,
    )

    #  mjlab 的 velocity 基础模板
    cfg = make_velocity_env_cfg()

    # === 机器人 ===
    cfg.scene.entities = {"robot": MICRODUCK_WALK_ROBOT_CFG}
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg, foot_height_scan_cfg)
    cfg.viewer.body_name = "trunk_base"

    # === 动作 ===
    # 位置控制，scale=1.0 表示动作直接作为关节位置目标（14 个伺服）。
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # === 速度命令（twist）===
    # 只保留前进/后退这一轴：lin_vel_x ±0.2，横向和转向都置 0。
    # 任务被简化成纯前后向的僵尸跳，不再训练横移/转向。
    twist = cfg.commands["twist"]
    twist.ranges.lin_vel_x = (-LIN_VEL, LIN_VEL)
    twist.ranges.lin_vel_y = (0, 0)
    twist.ranges.ang_vel_z = (0, 0)

    # === 头部姿态命令（4D：neck_pitch, head_pitch, head_yaw, head_roll）===
    # 从 HOME 偏移的 delta 命令。初始范围很小但非零，让输入从第 0 步就活着，
    # 之后由 head_pose_range curriculum 逐步放宽。
    cfg.commands["head_pose"] = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=HEAD_POSE_CMD_RESAMPLE_S,
        ranges=(
            (-0.05, 0.05),    # neck_pitch
            (-0.05, 0.05),    # head_pitch
            (-0.07, 0.07),    # head_yaw
            (-0.015, 0.015),  # head_roll（更紧 —— 机械行程远小于其余关节）
        ),
    )

    # === 身体姿态命令（6D：x, y, z, roll, pitch, yaw）===
    # 名义站姿的 delta。采样极小范围、且 body_pose_tracking 权重 0，只为让 obs 槽位
    # 保持存活，不真正驱动策略。
    cfg.commands["body_pose"] = microduck_mdp.UniformPoseCommandCfg(
        resampling_time_range=BODY_POSE_CMD_RESAMPLE_S,
        ranges=(
            (-0.005, 0.005),  # x (m)
            (-0.005, 0.005),  # y (m)
            (-0.005, 0.005),  # z (m)
            (-0.05, 0.05),    # roll (rad)
            (-0.05, 0.05),    # pitch (rad)
            (-0.05, 0.05),    # yaw (rad)
        ),
    )

    # === 地形 ===
    # 平地（无 terrain generator），纯跳任务不需要粗糙地形。
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # === 奖励 ===
    # pose：把腿部关节拉回 HOME/站姿。std_standing 收紧、std_running 放松。
    # 关键：asset_cfg 的正则把 neck/head 排除在外 —— 头由 head_pose_tracking 单独管，
    # 若头也进 pose，会和 head_pose_tracking 打架（一个拉回 HOME、一个拉向命令）。
    cfg.rewards["pose"].params["std_standing"] = std_standing
    cfg.rewards["pose"].params["std_walking"] = std_running
    cfg.rewards["pose"].params["std_running"] = std_running
    cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=(r"^(?!passive_|.*neck.*|.*head.*).*",)
    )
    cfg.rewards["pose"].params["walking_threshold"] = 0.01
    cfg.rewards["pose"].weight = 0.8

    # head_pose_tracking：追踪头部命令的主目标。std=0.5 的逐关节高斯，最终是 4 个
    # 关节的均值，所以部分追踪给部分奖励（不会全有或全无）。
    cfg.rewards["head_pose_tracking"] = RewardTermCfg(
        func=microduck_mdp.head_pose_tracking,
        weight=2.0,
        params={"command_name": "head_pose", "std": 0.5},
    )

    # body_pose_tracking：身体姿态追踪，权重 0 —— 只为保留奖励槽位/obs 契约，
    # 不实际驱动（跳不需要身体姿态偏移）。
    cfg.rewards["body_pose_tracking"] = RewardTermCfg(
        func=microduck_mdp.body_pose_tracking_6d,
        weight=0.0,
        params={
            "command_name": "body_pose",
            "nominal_height": 0.095,
            "xy_std": 0.05,
            "z_std": 0.02,
            "angle_std": math.radians(15),
        },
    )

    # upright：躯干保持竖直（相对世界重力方向）。weight=2.0 / std²=0.05 足够梯度把
    # 躯干拉平 —— 跳完落地必须站得回来，否则会脸朝下栽。
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["upright"].weight = 2.0
    cfg.rewards["upright"].params["std"] = math.sqrt(0.05)

    # foot_clearance / foot_slip 的 asset_cfg 指定两只脚。
    for reward_name in ["foot_clearance", "foot_slip"]:
        cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

    # body_ang_vel：躯干角速度惩罚（防乱甩）。
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)

    # foot_slip：脚步打滑惩罚。故意设弱（-0.1），太强会限制小机器人的转动。
    cfg.rewards["foot_slip"].weight = -0.1
    cfg.rewards["foot_slip"].params["command_threshold"] = 0.01

    # 去掉 soft_landing。
    cfg.rewards.pop("soft_landing", None)

    # self_collisions：惩罚腿撞躯干。
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )

    # leg_sync：跳的核心 —— 压出「双脚并拢」。reward = -|left+right|²，
    # 左右腿是镜像符号（双脚同相 = left ≈ -right），所以并拢时 |left+right|≈0 拿满分，
    # 交替步态（left ≈ right 同号）被重罚。
    cfg.rewards["leg_sync"] = RewardTermCfg(
        func=microduck_mdp.leg_sync_reward,
        weight=5.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # air_time：奖励滞空。threshold_min=0.12 只认真正的腾空，threshold_max=0.25 封顶。
    # 关键 —— 用 forward_hop_air_time 把「腾空」和「实际前进」绑在一起：只有身体真的
    # 朝命令方向移动时，腾空才给分。否则策略会学成原地垂直跳 / 原地踏步（腾空有分、
    # 前进没分，air_time+leg_sync+upright 已经白拿 ~13 的 mass），这正是「跳出来还是
    # 原地踏步」的根因。vel_gate_ref 是身体前进速度的饱和点（0→1 斜坡）。
    cfg.rewards["air_time"].func = microduck_mdp.forward_hop_air_time
    cfg.rewards["air_time"].weight = 4
    cfg.rewards["air_time"].params["command_threshold"] = 0.01
    cfg.rewards["air_time"].params["threshold_min"] = 0.12
    cfg.rewards["air_time"].params["threshold_max"] = 0.25
    cfg.rewards["air_time"].params["vel_gate_ref"] = 0.1

    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.1

    # === 速度追踪奖励 ===
    # track_linear_velocity：让向前成为第一目标（4.0 ≥ air_time 4.0），
    # 否则策略会选原地跳而不是向前跳。std=sqrt(0.15) 较松，容忍跳跃时速度振荡。
    cfg.rewards["track_linear_velocity"].weight = 4.0
    cfg.rewards["track_linear_velocity"].params["std"] = math.sqrt(0.15)
    # track_angular_velocity：角速度追踪（命令 ang_vel_z=0，即禁止旋转）。
    cfg.rewards["track_angular_velocity"].weight = 2.0
    cfg.rewards["track_angular_velocity"].params["std"] = math.sqrt(0.5)

    # action_rate_l2：动作平滑（二阶），初始 -0.1，由 action_rate_weight curriculum
    # 渐增到 -1.0。
    cfg.rewards["action_rate_l2"].weight = -0.1

    # foot_clearance：摆动脚离地高度目标 0.02m（机器人抬脚 ~1-2cm，匹配实际能力）。
    cfg.rewards["foot_clearance"].params["command_threshold"] = 0.01
    cfg.rewards["foot_clearance"].params["target_height"] = 0.02

    # foot_swing_height：摆动脚峰值高度目标。跳跃时全身离地、脚会比走路更高，
    cfg.rewards["foot_swing_height"].params["command_threshold"] = 0.01
    cfg.rewards["foot_swing_height"].params["target_height"] = 0.03

    # === 事件 & 终结 ===
    # BAM 启动事件：把每个 env 的 dof_frictionloss/dof_damping 字段展开（no-op 注册）。
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields,
        mode="startup",
    )

    # 重置时清空动作历史（动作延迟缓冲），避免跨 episode 污染。
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )

    # NaN 状态立即终结：MuJoCo 极端接触脉冲可能产生 NaN 关节位置，
    # 立即重置到有效状态，防止 NaN 扩散进观测缓冲区污染网络权重。
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
        params={"sensor_names": (feet_ground_cfg.name,)},
    )

    # 出生时躯干 z 高度范围（站姿高度附近）。
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.12, 0.13)

    # === 观测 ===
    gravity_term_name = "projected_gravity" if USE_PROJECTED_GRAVITY else "raw_accelerometer"

    # 删除 actor 的 base_lin_vel：策略不直接看自身速度（dance/beat 风格，靠重力+命令
    # +关节状态推断）。
    del cfg.observations["actor"].terms["base_lin_vel"]
    # 删除 height_scan：本环境没有 body-mounted 地形扫描传感器，actor/critic 都删。
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    # 深拷贝重力项和角速度项，避免和 critic 共享可变对象。
    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(
        cfg.observations["actor"].terms[gravity_term_name]
    )
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["actor"].terms["base_ang_vel"]
    )

    # 只给 critic 加 base_lin_vel（特权信息）：critic 能看到真实速度，actor 看不到。
    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel,
        scale=1.0,
    )

    # joint_pos/joint_vel 只保留主动关节（排除 passive_*），让观测维度 = 动作维（14），
    # 而不是原始关节数（16）。先深拷贝再改，因为 actor/critic 共享同一 term 对象。
    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)

    # 把 head/body 命令追加进 actor 和 critic 观测。
    # 顺序即运行时 61D 契约：[twist(3), head_pose(4), body_pose(6)]。
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "head_pose"},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "body_pose"},
        )

    # critic 的传感器观测换成 _safe 版本：这是防 NaN 崩 run 的加固，不是随机化。
    # nan_state 只检查关节+root 状态，护不住这些射线/接触传感器读出的非有限值；
    # 一个 NaN 就会让 rsl_rl 的 check_nan 崩掉整个 run。critic-only，policy 无代价。
    for _term, _safe in (
        ("foot_contact_forces", microduck_mdp.foot_contact_forces_safe),
        ("foot_height", microduck_mdp.foot_height_safe),
        ("foot_air_time", microduck_mdp.foot_air_time_safe),
    ):
        if _term in cfg.observations["critic"].terms:
            cfg.observations["critic"].terms[_term].func = _safe

    # === Curriculum ===
    # 删除默认的 terrain_levels 和 command_vel（本任务用平地、固定命令范围）。
    del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    # action_rate 权重斜坡：步态起步时轻平滑，之后收紧到 -1.0（iter 1500）。
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.1},
                {"step": 500 * 24, "weight": -0.2},
                {"step": 750 * 24, "weight": -0.4},
                {"step": 1000 * 24, "weight": -0.6},
                {"step": 1250 * 24, "weight": -0.8},
                {"step": 1500 * 24, "weight": -1.0},
            ],
        },
    )

    # 站立环境占比：步态建立后逐步增加到 25% 站立（命令=0），明确训练「静止」状态。
    cfg.curriculum["standing_envs"] = CurriculumTermCfg(
        func=microduck_mdp.standing_envs_curriculum,
        params={
            "command_name": "twist",
            "standing_stages": [
                {"step": 0,           "rel_standing_envs": 0.02},
                {"step": 500 * 24,    "rel_standing_envs": 0.05},
                {"step": 750 * 24,    "rel_standing_envs": 0.1},
                {"step": 1000 * 24,   "rel_standing_envs": 0.15},
                {"step": 1500 * 24,   "rel_standing_envs": 0.2},
                {"step": 2000 * 24,   "rel_standing_envs": 0.25},
            ],
        },
    )

    # 头部命令范围 curriculum：5 档逐步放宽到每个关节的机械可达 delta。
    cfg.curriculum["head_pose_range"] = CurriculumTermCfg(
        func=microduck_mdp.pose_command_range_curriculum,
        params={
            "command_name": "head_pose",
            "range_stages": [
                # step,                ranges = ((neck_pitch), (head_pitch), (head_yaw),  (head_roll))
                {"step": 0,         "ranges": ((-0.05, 0.05),  (-0.05, 0.05),  (-0.07, 0.07),  (-0.015, 0.015))},
                {"step": 500 * 24,  "ranges": ((-0.17, 0.17),  (-0.17, 0.17),  (-0.21, 0.21),  (-0.047, 0.047))},
                {"step": 1000 * 24, "ranges": ((-0.39, 0.39),  (-0.39, 0.39),  (-0.49, 0.49),  (-0.11, 0.11))},
                {"step": 1500 * 24, "ranges": ((-0.72, 0.72),  (-0.72, 0.72),  (-0.91, 0.91),  (-0.20, 0.20))},
                {"step": 2000 * 24, "ranges": ((-1.10, 1.10),  (-1.10, 1.10),  (-1.40, 1.40),  (-0.31, 0.31))},
            ],
        },
    )

    # 身体命令范围 curriculum：跳保持极小（body_pose_tracking 权重也是 0），
    # 只为让 obs 槽位存活。
    cfg.curriculum["body_pose_range"] = CurriculumTermCfg(
        func=microduck_mdp.pose_command_range_curriculum,
        params={
            "command_name": "body_pose",
            "range_stages": [
                {"step": 0, "ranges": (
                    (-0.005, 0.005),  # x (m)
                    (-0.005, 0.005),  # y (m)
                    (-0.005, 0.005),  # z (m)
                    (-0.05, 0.05),    # roll
                    (-0.05, 0.05),    # pitch
                    (-0.05, 0.05),    # yaw
                )},
            ],
        },
    )
    return cfg


# === RL 配置 ===
MicroduckZombieJumpRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,          # 观测归一化 ON —— 导出 ONNX 时必须 bake 进去
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
        symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,  # ENABLE_SYMMETRY=False → 关对称损失
    ),
    wandb_project="mjlab_microduck",
    experiment_name="jump",  # 目录名
    run_name="dance",        # wandb 里 <datetime>_dance 的后缀（注意：和 experiment_name 不一致，建议统一）
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=50_000,
)
