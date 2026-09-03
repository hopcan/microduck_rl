"""Microduck DANCE task — beat-conditioned dancing on flat ground.

The policy dances on the spot to a sampled beat: a DanceCommand term drives the
6D "body_pose" command slot with [sin(φ/2), cos(φ/2), tempo_norm, move 2-bit id]
(0 squat_bounce / 1 weight_shift / 2 head_bob / 3 climax)
(see the DANCE section header in mdp.py for the exact mapping — the 13D command
block layout [twist(3), head_pose(4), body_pose(6)] is UNCHANGED, so the 61D
obs contract and runtime policy hot-swap still hold).

Three procedural moves, all referenced analytically from the beat phase:
  0 squat_bounce  — trunk z dips 2.5 cm sinusoidally, lowest point ON the beat
  1 weight_shift  — trunk roll ±8°, period 2 beats, hip_roll joint reference
  2 head_bob      — head_pitch nods ±15° at 2× the beat frequency

Built on make_microduck_velocity_env_cfg() so DR / obs noise / delays / NaN
guards / BAM friction plumbing come for free; the walking-specific terms are
then re-tuned or removed:

  - twist(3) keeps its velocity semantics but with TINY ranges (this task is
    danced in place; the slots stay alive for later spin/step_touch moves)
  - head_pose(4) keeps its semantics with the small initial ranges (no widening
    curriculum); head_bob's reference nod is carried by dance_joint_tracking,
    not by the head_pose slot
  - gait terms removed (air_time, foot_clearance, foot_swing_height): the feet
    stay planted; no_stepping_penalty enforces that
  - motion-blocker regularizers lowered vs walking (dance is dynamic):
    body_ang_vel -0.05 → -0.01, angular_momentum -0.02 → -0.005
  - pose regularizer loosened (weight 1.0 → 0.25, standing std = walking std):
    squatting/rolling must move the legs away from HOME without fighting a
    tight standing pose
  - head_pose_tracking 2.0 → 0.3 (slot alive; the bob reference at weight 1.0
    wins for move 2 — for moves 0/1 the head stays near the ~neutral command)
  - the velocity env's body_pose_tracking_6d infra is REMOVED: its semantics
    (6D pose deltas) clash with the dance command mapping on the same slot
  - action_rate ramps only to -0.6 (rhythmic motion is smoother-taxable but
    must not be blocked)
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import CurriculumTermCfg, RewardTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    NUM_STEPS_PER_ENV,
    make_microduck_velocity_env_cfg,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG

ENABLE_SYMMETRY = False

# Beat / move sampling
DANCE_BPM_RANGE = microduck_mdp.DANCE_BPM_RANGE            # (90, 140)
DANCE_MOVE_LEN_BEATS = microduck_mdp.DANCE_MOVE_LEN_BEATS  # (8, 16) beats per move

# Dead-weight-guard sampling ranges for the slots this task doesn't drive.
TWIST_TINY_LIN = 0.02   # m/s
TWIST_TINY_ANG = 0.05   # rad/s

# Loosened pose stds for dancing (standing std = the base's walking std, with
# hip_roll opened up so the weight_shift hip reference isn't fought).
DANCE_STD_POSE = {
    r".*hip_yaw.*": 0.3,
    r".*hip_roll.*": 0.15,
    r".*hip_pitch.*": 0.4,
    r".*knee.*": 0.4,
    r".*ankle.*": 0.25,
}

# STAND_Z — measured standing trunk z (same constant as standup/ball_kick).
DANCE_NOMINAL_HEIGHT = 0.115


def make_microduck_dance_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create the Microduck beat-conditioned dance environment configuration."""

    cfg = make_microduck_velocity_env_cfg(play=play)

    # === COMMANDS ===
    # twist: velocity semantics kept, tiny ranges (dead-weight guard). The base
    # factory already deepcopied the command + wrapped it in
    # VelocityCommandCommandOnlyCfg, so these ranges are private to this cfg.
    twist = cfg.commands["twist"]
    twist.ranges.lin_vel_x = (-TWIST_TINY_LIN, TWIST_TINY_LIN)
    twist.ranges.lin_vel_y = (-TWIST_TINY_LIN, TWIST_TINY_LIN)
    twist.ranges.ang_vel_z = (-TWIST_TINY_ANG, TWIST_TINY_ANG)

    # head_pose: keep the base factory's small initial ranges; the widening
    # curriculum is removed below. head_bob's nod is a joint reference in
    # dance_joint_tracking, NOT a head_pose command.
    # (cfg.commands["head_pose"] untouched.)

    # body_pose: the dance command (semantic remap documented in the module
    # docstring / mdp.py DANCE header).
    cfg.commands["body_pose"] = microduck_mdp.DanceCommandCfg(
        bpm_range=DANCE_BPM_RANGE,
        move_len_beats=DANCE_MOVE_LEN_BEATS,
    )

    # === REWARDS ===
    # Gait terms off — the feet stay planted for all three moves.
    for name in ("air_time", "foot_clearance", "foot_swing_height"):
        cfg.rewards.pop(name, None)

    # In-place stay: the twist command is ~0, so velocity tracking pays for
    # standing still — kept at a modest weight as the stay-in-place objective.
    cfg.rewards["track_linear_velocity"].weight = 0.5
    cfg.rewards["track_angular_velocity"].weight = 0.5

    # Upright stays meaningful (don't fall) but weaker than walking: the
    # weight_shift move rolls the trunk ±8° BY DESIGN and shouldn't be taxed
    # out of existence (dance_body_tracking pays 2.0 for hitting the roll).
    cfg.rewards["upright"].weight = 1.0

    # Motion-blockers LOW for a dynamic task (AGENTS.md): bouncing and rolling
    # physically require trunk angular motion.
    cfg.rewards["body_ang_vel"].weight = -0.01
    cfg.rewards["angular_momentum"].weight = -0.005

    # Pose regularizer: loosened so the dance references own the motion.
    cfg.rewards["pose"].weight = 0.25
    cfg.rewards["pose"].params["std_standing"] = DANCE_STD_POSE
    cfg.rewards["pose"].params["std_walking"] = DANCE_STD_POSE
    cfg.rewards["pose"].params["std_running"] = DANCE_STD_POSE

    # Head pose slot: alive, near-neutral; the bob reference out-weighs it.
    cfg.rewards["head_pose_tracking"].weight = 0.1

    # The base's body_pose_tracking_6d infra reads the body_pose slot as 6D pose
    # deltas — wrong semantics here (the slot is a dance command). Removed.
    cfg.rewards.pop("body_pose_tracking", None)

    # Main objective: track the analytic reference motion.
    cfg.rewards["dance_body_tracking"] = RewardTermCfg(
        func=microduck_mdp.dance_body_tracking,
        weight=2.5,
        params={
            "command_name": "body_pose",
            "nominal_height": DANCE_NOMINAL_HEIGHT,
            "z_std": 0.010,
            "angle_std": math.radians(6),
        },
    )
    cfg.rewards["dance_joint_tracking"] = RewardTermCfg(
        func=microduck_mdp.dance_joint_tracking,
        weight=1.0,
        params={"command_name": "body_pose", "std": 0.10},
    )
    # Beat synchrony: potential-based Δ-alignment shaping (unfarmable).
    cfg.rewards["dance_beat_sync"] = RewardTermCfg(
        func=microduck_mdp.dance_beat_sync,
        weight=1.5,
        params={"command_name": "body_pose"},
    )
    # Feet planted (self-negating cost ≥ 0 → negative weight).
    cfg.rewards["no_stepping"] = RewardTermCfg(
        func=microduck_mdp.no_stepping_penalty,
        weight=-0.5,
        params={"sensor_name": "feet_ground_contact"},
    )

    # === CURRICULUM ===
    # Remove curricula that target walking semantics:
    #   standing_envs  — ramps twist rel_standing_envs (walk-specific)
    #   head_pose_range — widens head commands (dance keeps them small)
    #   body_pose_range — would write UniformPoseCommand ranges onto the
    #                     DanceCommand cfg (harmless but meaningless)
    for name in ("standing_envs", "head_pose_range", "body_pose_range"):
        cfg.curriculum.pop(name, None)

    # Smoothness ramps gently and caps at -0.6 (vs -1.0 for walking): rhythmic
    # 1.5–2.3 Hz motion must stay affordable.
    cfg.curriculum["action_rate_weight"].params["weight_stages"] = [
        {"step": 0, "weight": -0.05},
        {"step": 500 * NUM_STEPS_PER_ENV, "weight": -0.1},
        {"step": 1000 * NUM_STEPS_PER_ENV, "weight": -0.2},
        {"step": 1500 * NUM_STEPS_PER_ENV, "weight": -0.3},
    ]
    # Kept as inherited: com_range, head_com_range, head_pose_bias_weight.
    # head_pose_bias measures the error vs the ~neutral head_pose command; its
    # 1 s EMA lets the head_bob oscillation (3–4.6 Hz) cancel, so it prices
    # only DC droop — exactly the intent.

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
    experiment_name="dance",  # Directory name
    run_name="dance",  # Appended to datetime in wandb: <datetime>_dance
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=50_000,
)