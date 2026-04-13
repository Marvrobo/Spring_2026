# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import configclass

from ..keypoints import goal_keypoints_w, pushable_keypoints_w

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class KeypointDebugCommand(CommandTerm):
    """Command term that visualizes object and goal keypoints."""

    cfg: KeypointDebugCommandCfg

    def __init__(self, cfg: KeypointDebugCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.object: RigidObject = env.scene[cfg.asset_name]
        self._object_asset_cfg = SceneEntityCfg(cfg.asset_name)
        self._dummy_command = torch.zeros(self.num_envs, 1, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._dummy_command

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        pass

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "keypoint_visualizer"):
                self.keypoint_visualizer = VisualizationMarkers(
                    self.cfg.keypoint_visualizer_cfg
                )
            self.keypoint_visualizer.set_visibility(True)
        else:
            if hasattr(self, "keypoint_visualizer"):
                self.keypoint_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.object.is_initialized:
            return

        env = cast("ManagerBasedRLEnv", self._env)
        object_kps_w = pushable_keypoints_w(
            env,
            object_asset_cfg=self._object_asset_cfg,
        )
        goal_kps_w = goal_keypoints_w(
            env,
            goal_term_name=self.cfg.goal_term_name,
            object_asset_cfg=self._object_asset_cfg,
        )

        num_object = object_kps_w.shape[0] * object_kps_w.shape[1]
        translations = torch.cat([object_kps_w, goal_kps_w], dim=1).reshape(
            -1,
            3,
        )
        marker_indices = torch.cat(
            [
                torch.zeros(num_object, dtype=torch.long, device=self.device),
                torch.ones(num_object, dtype=torch.long, device=self.device),
            ],
            dim=0,
        )

        self.keypoint_visualizer.visualize(
            translations=translations,
            marker_indices=marker_indices,
        )


@configclass
class KeypointDebugCommandCfg(CommandTermCfg):
    """Configuration for keypoint debug visualization."""

    class_type: type = KeypointDebugCommand

    asset_name: str = "object"
    goal_term_name: str = "goal_region"

    keypoint_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/keypoints",
        markers={
            "object_keypoint": sim_utils.SphereCfg(
                radius=0.0075,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.1, 0.55, 1.0)
                ),
            ),
            "goal_keypoint": sim_utils.SphereCfg(
                radius=0.0075,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.1, 1.0, 0.2)
                ),
            ),
        },
    )

    def __post_init__(self):
        # Use a large finite range so CommandTerm.reset() can call
        # torch.uniform_ safely.
        self.resampling_time_range = (1.0e6, 1.0e6)
