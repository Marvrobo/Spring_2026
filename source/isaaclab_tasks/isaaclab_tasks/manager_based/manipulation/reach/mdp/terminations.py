from __future__ import annotations

from typing import TYPE_CHECKING

import torch

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
