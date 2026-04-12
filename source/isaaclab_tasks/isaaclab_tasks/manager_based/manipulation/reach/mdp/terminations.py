from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def time_out(
    env: ManagerBasedRLEnv,
    max_duration_s: float = 20.0,
) -> torch.Tensor:
    """Terminate when elapsed environment time exceeds ``max_duration_s``.

    This uses actual elapsed environment time computed as:
    ``elapsed_time_s = episode_length_buf * step_dt``.
    """

    elapsed_time_s = (
        env.episode_length_buf.to(dtype=torch.float32) * env.step_dt
    )
    return elapsed_time_s >= max_duration_s


def root_velocity_limits(
    env: ManagerBasedRLEnv,
    max_lin_vel: float,
    max_ang_vel: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when root linear or angular velocity exceeds limits."""
    asset = env.scene[asset_cfg.name]
    lin_vel_norm = torch.linalg.norm(asset.data.root_lin_vel_w, dim=-1)
    ang_vel_norm = torch.linalg.norm(asset.data.root_ang_vel_w, dim=-1)
    return (lin_vel_norm > max_lin_vel) | (ang_vel_norm > max_ang_vel)


def joint_velocity_limits(
    env: ManagerBasedRLEnv,
    max_vel: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when any joint velocity exceeds the given absolute limit."""
    asset = env.scene[asset_cfg.name]
    return torch.any(torch.abs(asset.data.joint_vel) > max_vel, dim=-1)
