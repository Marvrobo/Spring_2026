from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
	from isaaclab.envs import ManagerBasedRLEnv


__all__ = [
	"pushable_keypoints_w",
	"goal_keypoints_w",
	"keypoint_mean_distance",
	"keypoint_yaw_error_deg_xy",
	"keypoint_alignment_error",
	"keypoint_alignment_exp",
]


def pushable_keypoints_w(
	env: ManagerBasedRLEnv,
	object_asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
	"""Return object-attached keypoints in world frame. Shape: ``(N, 8, 3)``.

	The keypoints are defined once in the object's local frame and transformed by the
	current root pose, so they move rigidly with the object at every simulation step.
	"""
	local_kps = _get_object_local_keypoints(env, object_asset_cfg=object_asset_cfg)
	obj = env.scene[object_asset_cfg.name]
	return _transform_points(local_kps, obj.data.root_pos_w, obj.data.root_quat_w)


def goal_keypoints_w(
	env: ManagerBasedRLEnv,
	goal_term_name: str = "goal_region",
	object_asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
	"""Return goal keypoints in world frame with the same local geometry as the object."""
	local_kps = _get_object_local_keypoints(env, object_asset_cfg=object_asset_cfg)
	goal_pos_w, goal_quat_w = _get_goal_pose_w(env, goal_term_name)
	return _transform_points(local_kps, goal_pos_w, goal_quat_w)


def keypoint_mean_distance(k_a_w: torch.Tensor, k_b_w: torch.Tensor) -> torch.Tensor:
	"""Mean Euclidean distance between corresponding keypoints."""
	return torch.linalg.norm(k_a_w - k_b_w, dim=-1).mean(dim=-1)


def keypoint_yaw_error_deg_xy(k_src_w: torch.Tensor, k_tgt_w: torch.Tensor) -> torch.Tensor:
	"""Best-fit signed yaw error in degrees aligning source to target keypoints in XY."""
	src_xy = k_src_w[..., :2]
	tgt_xy = k_tgt_w[..., :2]

	src_centered = src_xy - src_xy.mean(dim=-2, keepdim=True)
	tgt_centered = tgt_xy - tgt_xy.mean(dim=-2, keepdim=True)

	cov = src_centered.transpose(-1, -2) @ tgt_centered
	u, _, vh = torch.linalg.svd(cov)
	v = vh.transpose(-1, -2)
	ut = u.transpose(-1, -2)

	det = torch.linalg.det(v @ ut)
	sign = torch.where(det < 0.0, -torch.ones_like(det), torch.ones_like(det))
	correction = torch.zeros_like(cov)
	correction[..., 0, 0] = 1.0
	correction[..., 1, 1] = sign

	rot = v @ correction @ ut
	yaw_err = torch.atan2(rot[..., 1, 0], rot[..., 0, 0])
	yaw_err = torch.remainder(yaw_err + torch.pi, 2.0 * torch.pi) - torch.pi
	return torch.rad2deg(yaw_err)


def keypoint_alignment_error(
	env: ManagerBasedRLEnv,
	goal_term_name: str = "goal_region",
	object_asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
	"""Return mean keypoint mismatch between the object and goal poses."""
	return keypoint_mean_distance(
		pushable_keypoints_w(env, object_asset_cfg=object_asset_cfg),
		goal_keypoints_w(env, goal_term_name=goal_term_name, object_asset_cfg=object_asset_cfg),
	)


def keypoint_alignment_exp(
	env: ManagerBasedRLEnv,
	goal_term_name: str = "goal_region",
	sigma: float = 0.10,
	object_asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
	"""Dense alignment score from object-attached keypoints to goal keypoints."""
	err = keypoint_alignment_error(env, goal_term_name=goal_term_name, object_asset_cfg=object_asset_cfg)
	denom = max(sigma * sigma, 1.0e-8)
	return torch.exp(-err / denom)


def _get_object_local_keypoints(
	env: ManagerBasedRLEnv,
	object_asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
	"""Return cached local-frame keypoints for the object. Shape: ``(N, 8, 3)``."""
	device = env.device
	obj = env.scene[object_asset_cfg.name]
	num_envs = env.scene.num_envs
	cache_key = f"_{object_asset_cfg.name}_keypoints_local"

	if hasattr(env, cache_key):
		cached = getattr(env, cache_key)
		if isinstance(cached, torch.Tensor) and cached.shape == (num_envs, 8, 3):
			return cached

	base_keypoints = _infer_local_keypoints_from_stage(env, object_asset_cfg, device)
	local_keypoints = base_keypoints.unsqueeze(0).repeat(num_envs, 1, 1)

	# Recenter XY so the local keypoints are attached to the object's root pose.
	center_xy = local_keypoints[:, :, :2].mean(dim=1, keepdim=True)
	local_keypoints[:, :, :2] -= center_xy

	setattr(env, cache_key, local_keypoints)
	return local_keypoints


def _infer_local_keypoints_from_stage(
	env: ManagerBasedRLEnv,
	object_asset_cfg: SceneEntityCfg,
	device: torch.device | str,
) -> torch.Tensor:
	"""Infer local keypoints from the first object prim in the scene, falling back to a unit box."""
	fallback = _box_keypoints_from_bounds((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5), device)
	try:
		from isaacsim.core.utils.stage import get_current_stage
		from pxr import Usd, UsdGeom
	except Exception:
		return fallback

	obj = env.scene[object_asset_cfg.name]
	prim_paths = None
	root_physx_view = getattr(obj, "root_physx_view", None)
	if root_physx_view is not None:
		prim_paths = getattr(root_physx_view, "prim_paths", None)
	if not prim_paths:
		return fallback

	stage = get_current_stage()
	if stage is None:
		return fallback

	prim = stage.GetPrimAtPath(prim_paths[0])
	if prim is None or not prim.IsValid():
		return fallback

	bbox_cache = UsdGeom.BBoxCache(
		Usd.TimeCode.Default(),
		[UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
	)
	geom_type, min_v, max_v, axis = _classify_prim_and_bounds(prim, bbox_cache)
	if min_v is None or max_v is None:
		return fallback

	if geom_type == "cylinder":
		return _cylinder_keypoints_from_bounds(min_v, max_v, axis, device)
	return _box_keypoints_from_bounds(min_v, max_v, device)


def _get_goal_pose_w(env: ManagerBasedRLEnv, goal_term_name: str) -> tuple[torch.Tensor, torch.Tensor]:
	"""Read goal pose from the active goal command term."""
	term = env.command_manager.get_term(goal_term_name)
	command = env.command_manager.get_command(goal_term_name)

	goal = getattr(term, "goal_region_w", None)
	if goal is None:
		goal = getattr(term, "goal_w", None)
	if goal is None:
		goal = command

	if goal.shape[-1] >= 7:
		return goal[:, :3], goal[:, 3:7]

	identity = torch.zeros((goal.shape[0], 4), device=goal.device, dtype=goal.dtype)
	identity[:, 0] = 1.0
	return goal[:, :3], identity


def _classify_prim_and_bounds(prim, bbox_cache):
	"""Return ``(geom_type, min_v, max_v, axis)`` in local frame."""
	from pxr import UsdGeom

	min_v, max_v = _local_bounds(prim, bbox_cache)
	axis = "Z"
	found_mesh = False
	stack = [prim]

	while stack:
		curr_prim = stack.pop()
		if curr_prim is None or not curr_prim.IsValid():
			continue

		if curr_prim.IsA(UsdGeom.Cylinder):
			cyl = UsdGeom.Cylinder(curr_prim)
			axis = cyl.GetAxisAttr().Get() or "Z"
			return "cylinder", min_v, max_v, axis

		if curr_prim.IsA(UsdGeom.Mesh):
			found_mesh = True

		stack.extend(curr_prim.GetChildren())

	return ("mesh" if found_mesh else "box"), min_v, max_v, axis


def _local_bounds(prim, bbox_cache):
	try:
		bbox = bbox_cache.ComputeLocalBound(prim)
		rng = bbox.GetRange()
		min_v = rng.GetMin()
		max_v = rng.GetMax()
		return (
			(float(min_v[0]), float(min_v[1]), float(min_v[2])),
			(float(max_v[0]), float(max_v[1]), float(max_v[2])),
		)
	except Exception:
		return None, None


def _box_keypoints_from_bounds(min_v, max_v, device):
	xs = (min_v[0], max_v[0])
	ys = (min_v[1], max_v[1])
	zs = (min_v[2], max_v[2])
	pts = []
	for x in xs:
		for y in ys:
			for z in zs:
				pts.append((x, y, z))
	return torch.tensor(pts, device=device, dtype=torch.float32)


def _cylinder_keypoints_from_bounds(min_v, max_v, axis: str, device):
	cx = 0.5 * (min_v[0] + max_v[0])
	cy = 0.5 * (min_v[1] + max_v[1])
	cz = 0.5 * (min_v[2] + max_v[2])
	sx = max_v[0] - min_v[0]
	sy = max_v[1] - min_v[1]
	sz = max_v[2] - min_v[2]

	axis = str(axis).upper()
	if axis == "X":
		h = sx
		ry = 0.5 * sy
		rz = 0.5 * sz
		bottom = cx - 0.5 * h
		top = cx + 0.5 * h
		pts = [
			(bottom, cy + ry, cz),
			(bottom, cy - ry, cz),
			(bottom, cy, cz + rz),
			(bottom, cy, cz - rz),
			(top, cy + ry, cz),
			(top, cy - ry, cz),
			(top, cy, cz + rz),
			(top, cy, cz - rz),
		]
	elif axis == "Y":
		h = sy
		rx = 0.5 * sx
		rz = 0.5 * sz
		bottom = cy - 0.5 * h
		top = cy + 0.5 * h
		pts = [
			(cx + rx, bottom, cz),
			(cx - rx, bottom, cz),
			(cx, bottom, cz + rz),
			(cx, bottom, cz - rz),
			(cx + rx, top, cz),
			(cx - rx, top, cz),
			(cx, top, cz + rz),
			(cx, top, cz - rz),
		]
	else:
		h = sz
		rx = 0.5 * sx
		ry = 0.5 * sy
		bottom = cz - 0.5 * h
		top = cz + 0.5 * h
		pts = [
			(cx + rx, cy, bottom),
			(cx - rx, cy, bottom),
			(cx, cy + ry, bottom),
			(cx, cy - ry, bottom),
			(cx + rx, cy, top),
			(cx - rx, cy, top),
			(cx, cy + ry, top),
			(cx, cy - ry, top),
		]

	return torch.tensor(pts, device=device, dtype=torch.float32)


def _transform_points(points_local: torch.Tensor, pos_w: torch.Tensor, quat_w: torch.Tensor) -> torch.Tensor:
	num_envs, num_keypoints, _ = points_local.shape
	q = quat_w[:, None, :].expand(num_envs, num_keypoints, 4).reshape(-1, 4)
	v = points_local.reshape(-1, 3)
	v_w = _quat_apply_safe(q, v).reshape(num_envs, num_keypoints, 3)
	return v_w + pos_w[:, None, :]


def _quat_apply_safe(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
	if hasattr(math_utils, "quat_apply"):
		return math_utils.quat_apply(q, v)

	w = q[:, 0:1]
	xyz = q[:, 1:4]
	t = 2.0 * torch.cross(xyz, v, dim=-1)
	return v + w * t + torch.cross(xyz, t, dim=-1)
