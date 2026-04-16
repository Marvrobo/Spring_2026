# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_apply, quat_error_magnitude, quat_inv, quat_mul

from .keypoints import keypoint_alignment_error

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame) and maps it with a tanh kernel.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore
    return quat_error_magnitude(curr_quat_w, des_quat_w)


def sparse_success_reward(
    env: ManagerBasedRLEnv,
    pos_tol: float,
    ang_tol: float,
    goal_term_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Return 1.0 when object pose is within position and angle tolerances, else 0.0.

    Position success is measured in the XY plane against the goal command.
    Angular success uses quaternion shortest-path error in radians.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    goal_w = env.command_manager.get_command(goal_term_name)

    goal_xy = goal_w[:, :2]
    obj_xy = asset.data.root_pos_w[:, :2]
    pos_err = torch.norm(goal_xy - obj_xy, dim=1)

    goal_quat_w = goal_w[:, 3:7]
    obj_quat_w = asset.data.root_quat_w
    ang_err = quat_error_magnitude(obj_quat_w, goal_quat_w)

    is_success = torch.logical_and(pos_err <= pos_tol, ang_err <= ang_tol)
    return is_success.to(torch.float32)


def object_goal_distance_exp(
    env: ManagerBasedRLEnv,
    goal_term_name: str = "goal_region",
    sigma: float = 0.10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Dense reward: exp(-||p_goal - p_obj||_obj / sigma^2)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    goal_w = env.command_manager.get_command(goal_term_name)

    obj_pos_w = asset.data.root_pos_w
    obj_quat_w = asset.data.root_quat_w
    goal_pos_w = goal_w[:, :3]

    pos_err_w = goal_pos_w - obj_pos_w
    pos_err_obj = quat_apply(quat_inv(obj_quat_w), pos_err_w)
    dist = torch.norm(pos_err_obj, dim=1)

    denom = max(sigma * sigma, 1.0e-8)
    return torch.exp(-dist / denom)


def object_goal_orientation_exp(
    env: ManagerBasedRLEnv,
    goal_term_name: str = "goal_region",
    sigma: float = 0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Dense reward: exp(-theta_obj / sigma^2) for object-goal orientation."""
    asset: RigidObject = env.scene[asset_cfg.name]
    goal_w = env.command_manager.get_command(goal_term_name)

    obj_quat_w = asset.data.root_quat_w
    goal_quat_w = goal_w[:, 3:7]

    goal_quat_obj = quat_mul(quat_inv(obj_quat_w), goal_quat_w)
    identity = torch.zeros_like(goal_quat_obj)
    identity[:, 0] = 1.0
    ang_err = quat_error_magnitude(goal_quat_obj, identity)

    denom = max(sigma * sigma, 1.0e-8)
    return torch.exp(-ang_err / denom)


def object_goal_pose_exp(
    env: ManagerBasedRLEnv,
    goal_term_name: str = "goal_region",
    pos_sigma: float = 0.10,
    ang_sigma: float = 0.35,
    pos_weight: float = 1.0,
    ang_weight: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Dense reward that combines translation and rotation pose errors.

    The translational error is measured in the object frame and the rotational error
    uses quaternion shortest-path magnitude in radians.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    goal_w = env.command_manager.get_command(goal_term_name)

    obj_pos_w = asset.data.root_pos_w
    obj_quat_w = asset.data.root_quat_w
    goal_pos_w = goal_w[:, :3]
    goal_quat_w = goal_w[:, 3:7]

    pos_err_w = goal_pos_w - obj_pos_w
    pos_err_obj = quat_apply(quat_inv(obj_quat_w), pos_err_w)
    pos_err = torch.norm(pos_err_obj, dim=1)

    goal_quat_obj = quat_mul(quat_inv(obj_quat_w), goal_quat_w)
    identity = torch.zeros_like(goal_quat_obj)
    identity[:, 0] = 1.0
    ang_err = quat_error_magnitude(goal_quat_obj, identity)

    pos_denom = max(pos_sigma * pos_sigma, 1.0e-8)
    ang_denom = max(ang_sigma * ang_sigma, 1.0e-8)
    pose_err = pos_weight * (pos_err / pos_denom) + ang_weight * (ang_err / ang_denom)
    return torch.exp(-pose_err)


def reach_reward_exp(
    env: ManagerBasedRLEnv,
    box_name: str,
    reach_term_name: str = "reach_target",
    ee_body_name: str = "panda_hand",
    sigma1: float = 0.38,
) -> torch.Tensor:
    """Reach reward: exp(-||p_er|| / sigma1^2) to the sampled reach target.

    Notes:
        The success-related arguments are kept for backward compatibility with existing
        configs, but this dense reward does not gate/hold the reward on success.
    """
    target_w = None
    try:
        term = env.command_manager.get_term(reach_term_name)
        target_w = getattr(term, "reach_target_w", None)
        if target_w is None:
            target_w = getattr(term, "reach_pos_w", None)
        if target_w is None:
            cmd = getattr(term, "command", None)
            if cmd is not None:
                if cmd.shape[1] >= 7:
                    target_w = cmd[:, :3]
                elif cmd.shape[1] == 3:
                    box: RigidObject = env.scene[box_name]
                    target_w = quat_apply(box.data.root_quat_w, cmd) + box.data.root_pos_w
    except Exception:
        target_w = None

    if target_w is None:
        box: RigidObject = env.scene[box_name]
        target_w = box.data.root_pos_w

    if not hasattr(env, "_reach_reward_ee_body_idx"):
        robot = env.scene["robot"]
        env._reach_reward_ee_body_idx = robot.find_bodies(ee_body_name)[0][0]
    ee_body_idx = env._reach_reward_ee_body_idx
    ee_w = env.scene["robot"].data.body_pos_w[:, ee_body_idx]

    dist = torch.linalg.norm(target_w - ee_w, dim=1)
    denom = max(sigma1 * sigma1, 1.0e-8)
    return torch.exp(-dist / denom)


def keypoint_alignment_reward(
    env: ManagerBasedRLEnv,
    goal_term_name: str = "goal_region",
    sigma: float = 0.10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Dense reward that measures pose alignment through object-attached keypoints."""
    err = keypoint_alignment_error(env, goal_term_name=goal_term_name, object_asset_cfg=asset_cfg)
    denom = max(sigma * sigma, 1.0e-8)
    return torch.exp(-err / denom)


def ee_touch_object_reward(
    env: ManagerBasedRLEnv,
    ee_body_name: str = "panda_hand",
    object_asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    touch_dist_thresh: float = 0.03,
    binary: bool = True,
    sigma: float = 0.5916,
) -> torch.Tensor:
    """Reward end-effector proximity to the object using root distance.

    If ``binary`` is True, returns 1.0 when within ``touch_dist_thresh`` and 0.0 otherwise.
    If ``binary`` is False, returns a smooth exponential reward based on nearest distance.
    """
    if not hasattr(env, "_touch_reward_ee_body_idx"):
        robot = env.scene["robot"]
        env._touch_reward_ee_body_idx = robot.find_bodies(ee_body_name)[0][0]

    ee_body_idx = env._touch_reward_ee_body_idx
    ee_pos_w = env.scene["robot"].data.body_pos_w[:, ee_body_idx]
    obj: RigidObject = env.scene[object_asset_cfg.name]
    dist = torch.linalg.norm(obj.data.root_pos_w - ee_pos_w, dim=1)

    if binary:
        return (dist <= touch_dist_thresh).to(torch.float32)

    denom = max(sigma, 1.0e-8)
    return torch.exp(-dist / denom)


def non_ee_tblock_contact_penalty(
    env: ManagerBasedRLEnv,
    ee_body_name: str = "panda_hand",
    object_asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    touch_dist_thresh: float = 0.05,
    binary: bool = True,
    sigma: float = 0.05,
) -> torch.Tensor:
    """Penalty when robot bodies other than the end-effector get too close to the T block.

    This uses the minimum distance from any non-end-effector body to the object root as a contact proxy.
    """
    robot = env.scene[robot_asset_cfg.name]
    obj: RigidObject = env.scene[object_asset_cfg.name]

    if not hasattr(env, "_non_ee_contact_body_ids"):
        ee_body_ids, _ = robot.find_bodies(ee_body_name)
        env._non_ee_contact_body_ids = [
            body_id for body_id in range(robot.data.body_pos_w.shape[1]) if body_id not in ee_body_ids
        ]

    body_ids = env._non_ee_contact_body_ids
    if len(body_ids) == 0:
        return torch.zeros(env.scene.num_envs, device=env.device)

    body_pos_w = robot.data.body_pos_w[:, body_ids, :]
    dist = torch.linalg.norm(body_pos_w - obj.data.root_pos_w[:, None, :], dim=-1).min(dim=1).values

    if binary:
        return (dist <= touch_dist_thresh).to(torch.float32)

    denom = max(sigma, 1.0e-8)
    return torch.exp(-dist / denom)


def object_velocity_toward_goal_reward(
    env: ManagerBasedRLEnv,
    goal_term_name: str = "goal_region",
    sigma2: float = 0.4,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    use_xy_only: bool = True,
) -> torch.Tensor:
    """Velocity-toward-goal reward: exp((v·p_hat)/sigma2^2 - 1)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    goal_w = env.command_manager.get_command(goal_term_name)

    goal_vec = goal_w[:, :3] - asset.data.root_pos_w
    obj_vel = asset.data.root_lin_vel_w

    if use_xy_only:
        goal_vec = goal_vec[:, :2]
        obj_vel = obj_vel[:, :2]

    goal_dir = goal_vec / torch.linalg.norm(goal_vec, dim=1, keepdim=True).clamp_min(1.0e-6)
    vel_toward_goal = torch.sum(obj_vel * goal_dir, dim=1)

    denom = max(sigma2 * sigma2, 1.0e-8)
    return torch.exp(vel_toward_goal / denom - 1.0)


def object_stall_penalty(
    env: ManagerBasedRLEnv,
    goal_term_name: str = "goal_region",
    vel_thresh: float = 0.01,
    pos_tol: float = 0.1,
    ang_tol: float = 0.174,
    stall_duration_s: float = 4.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    use_xy_only: bool = True,
) -> torch.Tensor:
    """Binary penalty for persistent object stalling before reaching the goal.

    Returns 1.0 only when object speed stays below ``vel_thresh`` for at least
    ``stall_duration_s`` while the object is not at the goal.
    Returns 0.0 when the object is at the goal, moving above threshold,
    or has not been stalled long enough.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    goal_w = env.command_manager.get_command(goal_term_name)

    goal_xy = goal_w[:, :2]
    obj_xy = asset.data.root_pos_w[:, :2]
    pos_err = torch.norm(goal_xy - obj_xy, dim=1)

    goal_quat_w = goal_w[:, 3:7]
    obj_quat_w = asset.data.root_quat_w
    ang_err = quat_error_magnitude(obj_quat_w, goal_quat_w)

    is_at_goal = torch.logical_and(pos_err <= pos_tol, ang_err <= ang_tol)

    obj_lin_vel = asset.data.root_lin_vel_w
    obj_ang_vel = asset.data.root_ang_vel_w
    if use_xy_only:
        obj_lin_vel = obj_lin_vel[:, :2]
        obj_ang_vel = obj_ang_vel[:, :2]
    combined_vel = torch.cat([obj_lin_vel, obj_ang_vel], dim=1)
    speed = torch.linalg.norm(combined_vel, dim=1)
    is_stalled = speed < vel_thresh

    if (not hasattr(env, "_object_stall_time_buf")) or (env._object_stall_time_buf.shape[0] != env.scene.num_envs):
        env._object_stall_time_buf = torch.zeros(env.scene.num_envs, device=env.device, dtype=torch.float32)

    # Reset carry-over at episode boundaries and accumulate consecutive stalled time.
    just_reset = env.episode_length_buf == 0
    stall_mask = torch.logical_and(~is_at_goal, is_stalled)
    env._object_stall_time_buf = torch.where(
        torch.logical_and(stall_mask, ~just_reset),
        env._object_stall_time_buf + float(env.step_dt),
        torch.zeros_like(env._object_stall_time_buf),
    )

    return (env._object_stall_time_buf >= stall_duration_s).to(torch.float32)
