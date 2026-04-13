# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_error_magnitude, quat_from_euler_xyz, quat_unique

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _resolve_repo_path(path: str) -> str:
    """Resolve repo-local paths like 'assets/...' to absolute paths."""
    path_obj = Path(path)
    if path_obj.is_absolute():
        return str(path_obj)
    for parent in Path(__file__).resolve().parents:
        candidate = parent / path_obj
        if candidate.exists():
            return str(candidate)
    return str(path_obj)


class GoalRegionCommand(CommandTerm):
    """Command term that samples a goal region pose for the object on the table."""

    cfg: GoalRegionCommandCfg

    def __init__(self, cfg: GoalRegionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.object: RigidObject = env.scene[cfg.asset_name]
        self.robot: Articulation = env.scene[cfg.robot_asset_name]
        self.ee_body_idx = self.robot.find_bodies(cfg.ee_body_name)[0][0]

        # Command pose is represented in world frame: (x, y, z, qw, qx, qy, qz)
        self.goal_region_w = torch.zeros(self.num_envs, 7, device=self.device)
        self.goal_region_w[:, 3] = 1.0

        self._success_pos_thresh = 0.10
        self._success_ang_thresh_rad = torch.deg2rad(torch.tensor(10.0, device=self.device))

        self.metrics["object_goal_distance_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["object_goal_angular_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["success_rate"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.goal_region_w

    def _update_metrics(self):
        object_pos_w = self.object.data.root_pos_w
        object_quat_w = self.object.data.root_quat_w

        dist_err = torch.norm(object_pos_w - self.goal_region_w[:, :3], dim=1)
        ang_err = quat_error_magnitude(object_quat_w, self.goal_region_w[:, 3:])
        is_success = torch.logical_and(dist_err < self._success_pos_thresh, ang_err < self._success_ang_thresh_rad)

        self.metrics["object_goal_distance_error"] = dist_err
        self.metrics["object_goal_angular_error"] = ang_err
        # Stored as per-env 0/1 so logger aggregation gives success rate.
        self.metrics["success_rate"] = is_success.to(torch.float32)

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return

        env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        env_origins = self._env.scene.env_origins[env_ids_tensor]

        r = torch.empty(len(env_ids), device=self.device)
        self.goal_region_w[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x) + env_origins[:, 0]
        self.goal_region_w[env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y) + env_origins[:, 1]
        self.goal_region_w[env_ids, 2] = r.uniform_(*self.cfg.ranges.pos_z) + env_origins[:, 2]

        euler_angles = torch.zeros(len(env_ids), 3, device=self.device)
        euler_angles[:, 0].uniform_(*self.cfg.ranges.roll)
        euler_angles[:, 1].uniform_(*self.cfg.ranges.pitch)
        euler_angles[:, 2].uniform_(*self.cfg.ranges.yaw)
        quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        self.goal_region_w[env_ids, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_region_visualizer"):
                self.goal_region_visualizer = VisualizationMarkers(self.cfg.goal_region_visualizer_cfg)
            self.goal_region_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_region_visualizer"):
                self.goal_region_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.object.is_initialized:
            return
        self.goal_region_visualizer.visualize(self.goal_region_w[:, :3], self.goal_region_w[:, 3:])


@configclass
class GoalRegionCommandCfg(CommandTermCfg):
    """Configuration for sampling a target goal-region pose on the tabletop."""

    class_type: type = GoalRegionCommand

    asset_name: str = "object"
    """Name of the object asset in the scene."""

    robot_asset_name: str = "robot"
    """Name of the robot articulation asset in the scene."""

    ee_body_name: str = "panda_hand"
    """Name of the end-effector body used for metric computation."""

    reach_target_command_name: str = "reach_target"
    """Name of the reach-target command term used for distance metric computation."""

    make_quat_unique: bool = True
    """Whether to ensure positive real quaternion component for uniqueness."""

    @configclass
    class Ranges:
        """Sampling ranges for goal region poses in world frame."""

        pos_x: tuple[float, float] = (0.45, 0.65)
        pos_y: tuple[float, float] = (-0.20, 0.20)
        pos_z: tuple[float, float] = (0.0, 0.0)
        roll: tuple[float, float] = (0.0, 0.0)
        pitch: tuple[float, float] = (0.0, 0.0)
        yaw: tuple[float, float] = (-3.141592653589793, 3.141592653589793)

    ranges: Ranges = Ranges()

    goal_region_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/goal_region",
        markers={
            "goal_region": sim_utils.UsdFileCfg(
                usd_path="assets/red_T_flat.usd",
                scale=(0.02, 0.02, 0.02),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            )
        },
    )

    def __post_init__(self):
        marker_cfg = self.goal_region_visualizer_cfg.markers["goal_region"]
        usd_path = getattr(marker_cfg, "usd_path", None)
        if isinstance(usd_path, str):
            setattr(marker_cfg, "usd_path", _resolve_repo_path(usd_path))
