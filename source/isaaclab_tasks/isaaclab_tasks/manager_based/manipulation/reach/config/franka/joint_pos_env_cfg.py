# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets import FRANKA_PANDA_CFG  # isort: skip


##
# Environment configuration
##


# We just use inherited SceneCfg.

# Define Commmand Configuration
@configclass
class FrankaReachCommandCfg:
    """Command configuraiton for Franka Push T task"""


    # define goal region for Push T task, should randomize within the region of the table.
    goal_region = mdp.GoalRegionCommandCfg(
        asset_name="object",
        resampling_time_range=(2.0, 4.0),
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


    # define reach target for intrinsic reward, which can be the sampled point cloud 
    # the surface of the T block
    reach_target = mdp.ReachTargetCommandCfg(
        asset_name="object",
        resampling_time_range=(2.0, 4.0),
        debug_vis=True,
    )




@configclass
class FrankaReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to franka
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["panda_hand"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["panda_hand"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["panda_hand"]

        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        # override command generator body
        # end-effector is along z-direction
        self.commands.ee_pose.body_name = "panda_hand"
        self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)


@configclass
class FrankaReachEnvCfg_PLAY(FrankaReachEnvCfg):

    # we do not define scene here since we use inherited scene
    # but do not forget to set configurations below
    commands: None
    actions: None
    observations: None
    rewards: None
    curriculum: None
    events: None
    terminations: None


    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
