# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject
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

		self._point_cloud_local = self._load_point_cloud(
			cfg.point_cloud_path,
			cfg.point_cloud_scale,
		).to(self.device)
		self._num_points = self._point_cloud_local.shape[0]
		if self._num_points == 0:
			raise ValueError(f"Point cloud contains no points: {cfg.point_cloud_path}")

		self.reach_target_local = torch.zeros(self.num_envs, 3, device=self.device)
		self.reach_target_w = torch.zeros(self.num_envs, 3, device=self.device)
		self.sampled_point_w = torch.zeros(self.num_envs, 3, device=self.device)

		self.metrics["distance_error"] = torch.zeros(self.num_envs, device=self.device)

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
		"""World-frame target points with shape (num_envs, 3)."""
		return self.reach_target_w

    # Notice that here we should use the distance between the end-effector and the reach target, instead of the object.
	def _update_metrics(self):
		object_pos_w = self.object.data.root_pos_w
		self.metrics["distance_error"] = torch.norm(self.reach_target_w - object_pos_w, dim=1)

	def _resample_command(self, env_ids: Sequence[int]):
		if len(env_ids) == 0:
			return

		sample_ids = torch.randint(0, self._num_points, (len(env_ids),), device=self.device)
		sampled_local = self._point_cloud_local[sample_ids]
		self.reach_target_local[env_ids] = sampled_local

		# Keep a world-frame snapshot of the sampled point for debugging.
		object_pos_w = self.object.data.root_pos_w[env_ids]
		object_quat_w = self.object.data.root_quat_w[env_ids]
		self.sampled_point_w[env_ids] = quat_apply(object_quat_w, sampled_local) + object_pos_w

	def _update_command(self):
		object_pos_w = self.object.data.root_pos_w
		object_quat_w = self.object.data.root_quat_w
		self.reach_target_w = quat_apply(object_quat_w, self.reach_target_local) + object_pos_w

	def _set_debug_vis_impl(self, debug_vis: bool):
		if debug_vis:
			if not hasattr(self, "reach_target_visualizer"):
				self.reach_target_visualizer = VisualizationMarkers(self.cfg.reach_target_visualizer_cfg)
			if not hasattr(self, "sampled_point_visualizer"):
				self.sampled_point_visualizer = VisualizationMarkers(self.cfg.sampled_point_visualizer_cfg)
			self.reach_target_visualizer.set_visibility(True)
			self.sampled_point_visualizer.set_visibility(True)
		else:
			if hasattr(self, "reach_target_visualizer"):
				self.reach_target_visualizer.set_visibility(False)
			if hasattr(self, "sampled_point_visualizer"):
				self.sampled_point_visualizer.set_visibility(False)

	def _debug_vis_callback(self, event):
		if not self.object.is_initialized:
			return
		orientation = torch.zeros((self.num_envs, 4), device=self.device)
		orientation[:, 0] = 1.0
		self.reach_target_visualizer.visualize(self.reach_target_w, orientation)
		self.sampled_point_visualizer.visualize(self.sampled_point_w, orientation)


@configclass
class ReachTargetCommandCfg(CommandTermCfg):
	"""Configuration for the reach target command term."""

	class_type: type = ReachTargetCommand

	asset_name: str = "object"
	"""Name of the rigid object asset used as the local frame for point-cloud targets."""

	point_cloud_path: str = "assets/red_T_flat_point_cloud.ply"
	"""Path to the .ply file containing point-cloud samples in object-local coordinates."""

	point_cloud_scale: float = 1.0
	"""Uniform scale applied to loaded point-cloud coordinates."""

	reach_target_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
		prim_path="/Visuals/Command/reach_target",
		markers={
			"target": sim_utils.SphereCfg(
				radius=0.01,
				visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.2, 0.95)),
			)
		},
	)

	sampled_point_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
		prim_path="/Visuals/Command/sampled_point",
		markers={
			"sampled_point": sim_utils.SphereCfg(
				radius=0.007,
				visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.75, 1.0)),
			)
		},
	)
