# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from copy import deepcopy

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import (
    ReachEnvCfg,
)

##
# Pre-defined configs
##
from isaaclab_assets import FRANKA_PANDA_CFG  # isort: skip


@configclass
class FrankaPushTCommandsCfg:
    """Command terms for the Franka Push-T task."""

    goal_region = mdp.GoalRegionCommandCfg(
        asset_name="object",
        resampling_time_range=(1e6, 1e6),
        debug_vis=True,
        ranges=mdp.GoalRegionCommandCfg.Ranges(
            pos_x=(0.45, 0.65),
            pos_y=(-0.20, 0.20),
            pos_z=(0.055, 0.055),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(-math.pi, math.pi),
        ),
    )

    reach_target = mdp.ReachTargetCommandCfg(
        asset_name="object",
        resampling_time_range=(1e6, 1e6),
        point_cloud_scale=0.001,
        debug_vis=True,
    )


@configclass
class FrankaPushTObservationsCfg:
    """Observation specifications for the Push-T task."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations."""

        # robot state
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        # object state
        object_pos = ObsTerm(
            func=mdp.root_pos_w,
            params={"asset_cfg": SceneEntityCfg("object")},
        )
        object_quat = ObsTerm(
            func=mdp.root_quat_w,
            params={
                "asset_cfg": SceneEntityCfg("object"),
                "make_quat_unique": True,
            },
        )
        # command state
        goal_region = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "goal_region"},
        )

        # reach_target = ObsTerm(
        #     func=mdp.generated_commands,
        #     params={"command_name": "reach_target"},
        # )
        # policy action memory
        
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class FrankaPushTRewardsCfg:
    """Reward terms for the Push-T task."""

    object_position_tracking = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=5.0,
        params={
            "asset_cfg": SceneEntityCfg("object", body_ids=[0]),
            "std": 0.05,
            "command_name": "goal_region",
        },
    )
    object_orientation_tracking = RewTerm(
        func=mdp.orientation_command_error,
        weight=-0.25,
        params={
            "asset_cfg": SceneEntityCfg("object", body_ids=[0]),
            "command_name": "goal_region",
        },
    )
    end_effector_to_reach_target = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["panda_hand"]),
            "std": 0.08,
            "command_name": "reach_target",
        },
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1.0e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-5.0)


@configclass
class FrankaPushTEventCfg:
    """Reset events for the Push-T task."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.8, 1.2),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_object_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "pose_range": {
                "x": (-0.03, 0.03),
                "y": (-0.10, 0.10),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )


@configclass
class FrankaPushTTerminationsCfg:
    """Termination settings for the Push-T task."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # object_fall = DoneTerm(
    #     func=mdp.root_height_below_minimum,
    #     params={"asset_cfg": SceneEntityCfg("object"), "minimum_height": 0.02},
    # )


@configclass
class FrankaPushTEnvCfg(ReachEnvCfg):
    """Manager-based RL environment config for Franka Push-T."""

    commands: FrankaPushTCommandsCfg = FrankaPushTCommandsCfg()
    observations: FrankaPushTObservationsCfg = FrankaPushTObservationsCfg()
    rewards: FrankaPushTRewardsCfg = FrankaPushTRewardsCfg()
    events: FrankaPushTEventCfg = FrankaPushTEventCfg()
    terminations: FrankaPushTTerminationsCfg = FrankaPushTTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # scene assets
        self.scene.robot = deepcopy(FRANKA_PANDA_CFG)
        self.scene.robot.prim_path = "{ENV_REGEX_NS}/Robot"
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.50, 0.0, 0.055),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
            spawn=sim_utils.UsdFileCfg(
                usd_path="assets/red_T_flat.usd",
                scale=(0.02, 0.02, 0.02),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=4,
                    max_depenetration_velocity=2.0,
                ),
            ),
        )

        # joint-space control for standard PPO training
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=0.5,
            use_default_offset=True,
        )

        self.episode_length_s = 10.0


@configclass
class FrankaPushTEnvCfg_PLAY(FrankaPushTEnvCfg):
    """Play variant for Franka Push-T."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.commands.goal_region.debug_vis = True
        self.commands.reach_target.debug_vis = True
