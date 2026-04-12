from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_inv, quat_unique, subtract_frame_transforms

if TYPE_CHECKING:
	from isaaclab.envs import ManagerBasedRLEnv


def previous_action(env: ManagerBasedRLEnv) -> torch.Tensor:
	"""Return previous action tensor for policy observations."""
	if hasattr(env.action_manager, "prev_action"):
		return env.action_manager.prev_action
	return env.action_manager.action


def previous_action_clamped(env: ManagerBasedRLEnv, clip_limit: float = 1.0) -> torch.Tensor:
	"""Return previous action observation clamped to ``[-clip_limit, clip_limit]``."""
	combined_actions = previous_action(env)
	return torch.clamp(combined_actions, min=-clip_limit, max=clip_limit)


def object_pos_in_robot_frame(
	env: ManagerBasedRLEnv,
	object_asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
	robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
	"""Return object root position expressed in the robot root frame."""
	obj = env.scene[object_asset_cfg.name]
	robot = env.scene[robot_asset_cfg.name]
	obj_pos_in_robot, _ = subtract_frame_transforms(
		robot.data.root_pos_w,
		robot.data.root_quat_w,
		obj.data.root_pos_w,
		obj.data.root_quat_w,
	)
	return obj_pos_in_robot


def object_quat_in_robot_frame(
	env: ManagerBasedRLEnv,
	object_asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
	robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
	make_quat_unique: bool = True,
) -> torch.Tensor:
	"""Return object root orientation expressed in the robot root frame."""
	obj = env.scene[object_asset_cfg.name]
	robot = env.scene[robot_asset_cfg.name]
	_, obj_quat_in_robot = subtract_frame_transforms(
		robot.data.root_pos_w,
		robot.data.root_quat_w,
		obj.data.root_pos_w,
		obj.data.root_quat_w,
	)
	if make_quat_unique:
		obj_quat_in_robot = quat_unique(obj_quat_in_robot)
	return obj_quat_in_robot


def object_lin_vel_in_robot_frame(
	env: ManagerBasedRLEnv,
	object_asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
	robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
	"""Return object root linear velocity relative to robot, expressed in robot frame."""
	obj = env.scene[object_asset_cfg.name]
	robot = env.scene[robot_asset_cfg.name]
	rel_lin_vel_w = obj.data.root_lin_vel_w - robot.data.root_lin_vel_w
	return quat_apply(quat_inv(robot.data.root_quat_w), rel_lin_vel_w)


def object_ang_vel_in_robot_frame(
	env: ManagerBasedRLEnv,
	object_asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
	robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
	"""Return object root angular velocity relative to robot, expressed in robot frame."""
	obj = env.scene[object_asset_cfg.name]
	robot = env.scene[robot_asset_cfg.name]
	rel_ang_vel_w = obj.data.root_ang_vel_w - robot.data.root_ang_vel_w
	return quat_apply(quat_inv(robot.data.root_quat_w), rel_ang_vel_w)


def object_state_in_robot_frame(
	env: ManagerBasedRLEnv,
	object_asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
	robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
	make_quat_unique: bool = True,
) -> torch.Tensor:
	"""Return full object root state in robot frame: pos, quat, lin vel, ang vel."""
	return torch.cat(
		[
			object_pos_in_robot_frame(env, object_asset_cfg=object_asset_cfg, robot_asset_cfg=robot_asset_cfg),
			object_quat_in_robot_frame(
				env,
				object_asset_cfg=object_asset_cfg,
				robot_asset_cfg=robot_asset_cfg,
				make_quat_unique=make_quat_unique,
			),
			object_lin_vel_in_robot_frame(env, object_asset_cfg=object_asset_cfg, robot_asset_cfg=robot_asset_cfg),
			object_ang_vel_in_robot_frame(env, object_asset_cfg=object_asset_cfg, robot_asset_cfg=robot_asset_cfg),
		],
		dim=-1,
	)


def command_in_object_frame(
	env: ManagerBasedRLEnv,
	command_name: str,
	object_asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
	make_quat_unique: bool = True,
) -> torch.Tensor:
	"""Return command state expressed in the object frame.

	Supported command layouts:
	- ``(N, 3)``: position vector in world frame
	- ``(N, 7)``: pose in world frame as ``(x, y, z, qw, qx, qy, qz)``
	"""
	obj = env.scene[object_asset_cfg.name]
	command = env.command_manager.get_command(command_name)

	if command.shape[-1] == 3:
		delta_pos_w = command - obj.data.root_pos_w
		return quat_apply(quat_inv(obj.data.root_quat_w), delta_pos_w)

	if command.shape[-1] == 7:
		cmd_pos_obj, cmd_quat_obj = subtract_frame_transforms(
			obj.data.root_pos_w,
			obj.data.root_quat_w,
			command[:, :3],
			command[:, 3:7],
		)
		if make_quat_unique:
			cmd_quat_obj = quat_unique(cmd_quat_obj)
		return torch.cat([cmd_pos_obj, cmd_quat_obj], dim=-1)

	raise ValueError(
		f"Unsupported command shape {tuple(command.shape)} for command '{command_name}'. "
		"Expected last dimension of 3 or 7."
	)
