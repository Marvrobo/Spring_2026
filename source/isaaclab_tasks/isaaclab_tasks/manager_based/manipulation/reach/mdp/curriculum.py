from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def modify_reward_weight_after_iterations(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    weight: float,
    num_iterations: int,
    steps_per_iteration: int | None = None,
) -> torch.Tensor:
    """Modify a reward term weight after a given number of learning iterations.

    The current iteration is estimated as
    ``common_step_counter // steps_per_iteration``.
    If ``steps_per_iteration`` is not provided, the function tries to use
    ``env._rsl_num_steps_per_env`` and falls back to 1.
    """
    del env_ids

    if steps_per_iteration is None:
        steps_per_iteration = int(getattr(env, "_rsl_num_steps_per_env", 0) or 0)
    if steps_per_iteration <= 0:
        steps_per_iteration = 1

    curr_iter = int(env.common_step_counter // steps_per_iteration)
    if curr_iter >= int(num_iterations):
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        term_cfg.weight = weight
        env.reward_manager.set_term_cfg(term_name, term_cfg)

    curr_weight = env.reward_manager.get_term_cfg(term_name).weight
    return torch.tensor(curr_weight, device=env.device)
