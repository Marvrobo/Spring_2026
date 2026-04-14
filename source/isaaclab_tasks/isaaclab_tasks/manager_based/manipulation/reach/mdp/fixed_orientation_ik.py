# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class FixedOrientationDifferentialInverseKinematicsAction(DifferentialInverseKinematicsAction):
    """3D delta-position IK action with a fixed end-effector orientation.

    The policy only commands Cartesian position deltas ``(dx, dy, dz)``.
    The desired end-effector quaternion is held fixed for the whole episode,
    using either the configured quaternion or the current orientation at reset.
    """

    cfg: FixedOrientationDifferentialInverseKinematicsActionCfg

    def __init__(self, cfg: FixedOrientationDifferentialInverseKinematicsActionCfg, env: ManagerBasedEnv):
        self._fixed_quat_initialized = False
        super().__init__(cfg, env)
        self._fixed_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self._fixed_quat[:, 0] = 1.0
        if self.cfg.fixed_orientation_quat is not None:
            self._fixed_quat[:] = torch.tensor(self.cfg.fixed_orientation_quat, device=self.device)
            self._fixed_quat_initialized = True

    @property
    def action_dim(self) -> int:
        """The reduced action space only controls xyz motion."""
        return 3

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions[:] = self.raw_actions * self._scale
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        if not self._fixed_quat_initialized:
            self._fixed_quat[:] = ee_quat_curr
            self._fixed_quat_initialized = True

        pos_des = ee_pos_curr + self._processed_actions
        pose_command = torch.cat((pos_des, self._fixed_quat), dim=-1)
        self._ik_controller.set_command(pose_command, ee_pos_curr, ee_quat_curr)

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        if self.cfg.fixed_orientation_quat is not None:
            if env_ids is None:
                self._fixed_quat[:] = torch.tensor(self.cfg.fixed_orientation_quat, device=self.device)
            else:
                self._fixed_quat[env_ids] = torch.tensor(self.cfg.fixed_orientation_quat, device=self.device)
            self._fixed_quat_initialized = True
        elif env_ids is None:
            self._fixed_quat_initialized = False


@configclass
class FixedOrientationDifferentialInverseKinematicsActionCfg(DifferentialInverseKinematicsActionCfg):
    """Configuration for 3D position-only IK with fixed end-effector orientation."""

    class_type: type = FixedOrientationDifferentialInverseKinematicsAction

    fixed_orientation_quat: tuple[float, float, float, float] | None = None
    """Fixed quaternion as ``(w, x, y, z)``.

    If ``None``, the end-effector orientation at episode reset is used and then held fixed.
    """
