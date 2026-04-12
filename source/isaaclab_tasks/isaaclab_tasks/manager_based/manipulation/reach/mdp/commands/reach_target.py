# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
	from isaaclab.envs import ManagerBasedRLEnv


class ReachTargetCommand(CommandTerm):
	"""Command term that samples a reach target from a .ply point cloud."""

	cfg: ReachTargetCommandCfg

	def __init__(self, cfg: ReachTargetCommandCfg, env: ManagerBasedRLEnv):
		super().__init__(cfg, env)

		self.object: RigidObject = env.scene[cfg.asset_name]
		self.robot: Articulation = env.scene[cfg.robot_asset_name]
		self.ee_body_idx = self.robot.find_bodies(cfg.ee_body_name)[0][0]

		self._point_cloud_local = self._load_point_cloud(
			cfg.point_cloud_path,
			cfg.point_cloud_scale,
		).to(self.device)
		self._num_points = self._point_cloud_local.shape[0]
		if self._num_points == 0:
			raise ValueError(f"Point cloud contains no points: {cfg.point_cloud_path}")

		self.reach_target_local = torch.zeros(self.num_envs, 3, device=self.device)
		self.reach_target_w = torch.zeros(self.num_envs, 3, device=self.device)


	@staticmethod
	def _load_point_cloud(path: str, scale: float) -> torch.Tensor:
		try:
			import open3d as o3d
		except ImportError as exc:
			raise ImportError("open3d is required for ReachTargetCommand. Install it in your environment.") from exc

		pcd = o3d.io.read_point_cloud(path)
		points = torch.tensor(pcd.points, dtype=torch.float32)
		if scale != 1.0:
			points = points * scale
		return points

	@property
	def command(self) -> torch.Tensor:
		"""Object-local sampled surface points with shape (num_envs, 3)."""
		return self.reach_target_local

    # Notice that here we should use the distance between the end-effector and the reach target, instead of the object.
	def _update_metrics(self):
		ee_pos_w = self.robot.data.body_pos_w[:, self.ee_body_idx]
		self.metrics["ee_reach_target_distance_error"] = torch.norm(self.reach_target_w - ee_pos_w, dim=1)

	def _resample_command(self, env_ids: Sequence[int]):
		if len(env_ids) == 0:
			return

		sample_ids = torch.randint(0, self._num_points, (len(env_ids),), device=self.device)
		sampled_local = self._point_cloud_local[sample_ids]
		self.reach_target_local[env_ids] = sampled_local

	def _update_command(self):
		object_pos_w = self.object.data.root_pos_w
		object_quat_w = self.object.data.root_quat_w
		self.reach_target_w = quat_apply(object_quat_w, self.reach_target_local) + object_pos_w

	def _set_debug_vis_impl(self, debug_vis: bool):
		if debug_vis:
			if not hasattr(self, "reach_target_visualizer"):
				self.reach_target_visualizer = VisualizationMarkers(self.cfg.reach_target_visualizer_cfg)
			self.reach_target_visualizer.set_visibility(True)
		else:
			if hasattr(self, "reach_target_visualizer"):
				self.reach_target_visualizer.set_visibility(False)

	def _debug_vis_callback(self, event):
		if not self.object.is_initialized:
			return
		orientation = torch.zeros((self.num_envs, 4), device=self.device)
		orientation[:, 0] = 1.0
		self.reach_target_visualizer.visualize(self.reach_target_w, orientation)


@configclass
class ReachTargetCommandCfg(CommandTermCfg):
	"""Configuration for the reach target command term."""

	class_type: type = ReachTargetCommand

	asset_name: str = "object"
	"""Name of the rigid object asset used as the local frame for point-cloud targets."""

	robot_asset_name: str = "robot"
	"""Name of the robot articulation asset in the scene."""

	ee_body_name: str = "panda_hand"
	"""Name of the end-effector body used for metric computation."""

	point_cloud_path: str = "assets/filtered_T_block_point_cloud.ply"
	"""Path to the .ply file containing point-cloud samples in object-local coordinates."""

	point_cloud_scale: float = 1.0
	"""Uniform scale applied to loaded point-cloud coordinates."""

	# pink sphere, which is the actual reach_target visualizer
	reach_target_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
		prim_path="/Visuals/Command/reach_target",
		markers={
			"target": sim_utils.SphereCfg(
				radius=0.05,
				visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.2, 0.95)),
			)
		},
	)


