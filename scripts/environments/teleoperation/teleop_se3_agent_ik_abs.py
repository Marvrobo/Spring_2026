# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard teleoperation with relative (delta) control for Isaac-Reach-Franka-IK-Abs-v0.

The keyboard outputs 6-D delta SE(3) commands each step:
  [dx, dy, dz, rx, ry, rz]

This script integrates those deltas into an absolute target pose
  [x, y, z, qw, qx, qy, qz]
and sends it to the IK-absolute environment, which expects absolute end-effector poses.

Key bindings (Se3Keyboard):
  W / S  — move along +x / -x
  A / D  — move along +y / -y
  Q / E  — move along +z / -z
  Z / X  — rotate around x-axis
  T / G  — rotate around y-axis
  C / V  — rotate around z-axis
  L      — clear accumulated keyboard delta
  R      — reset the environment
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Keyboard teleoperation (relative delta) for Isaac-Reach-Franka-IK-Abs-v0."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity scale for keyboard input.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
from scipy.spatial.transform import Rotation

import gymnasium as gym

from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

# The IK-absolute Franka Reach environment
TASK_NAME = "Isaac-Reach-Franka-IK-Abs-v0"

# End-effector body name as configured in the environment
EE_BODY_NAME = "panda_hand"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_ee_pose(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Return current end-effector position and quaternion (w, x, y, z) for all envs.

    Args:
        env: The unwrapped ManagerBasedRLEnv.

    Returns:
        pos (num_envs, 3) and quat_wxyz (num_envs, 4).
    """
    robot = env.scene["robot"]
    body_ids, _ = robot.find_bodies(EE_BODY_NAME)
    body_idx = body_ids[0]
    pos = robot.data.body_pos_w[:, body_idx, :].clone()       # (num_envs, 3)
    quat_wxyz = robot.data.body_quat_w[:, body_idx, :].clone()  # (num_envs, 4)
    return pos, quat_wxyz


def integrate_delta(
    desired_pos: torch.Tensor,
    desired_quat_wxyz: torch.Tensor,
    delta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Integrate a 6-D keyboard delta into the current absolute desired pose.

    The rotation delta is expressed as a rotation vector in the world frame and is
    pre-multiplied onto the current orientation (delta applied in world frame).

    Args:
        desired_pos: Current desired position, shape (num_envs, 3).
        desired_quat_wxyz: Current desired quaternion (w, x, y, z), shape (num_envs, 4).
        delta: 6-D keyboard delta tensor [dx, dy, dz, rx, ry, rz] on the simulation device.

    Returns:
        Updated (desired_pos, desired_quat_wxyz).
    """
    # -- position: add delta directly
    desired_pos = desired_pos + delta[:3].unsqueeze(0)  # broadcast over envs

    # -- orientation: compose rotation-vector delta onto current quaternion
    d_rot_np = delta[3:].cpu().numpy()
    angle = float((delta[3:] ** 2).sum().sqrt())
    if angle > 1e-8:
        # scipy uses (x, y, z, w); Isaac Lab uses (w, x, y, z)
        dq_xyzw = Rotation.from_rotvec(d_rot_np).as_quat()  # (x, y, z, w)

        updated_quats = []
        for i in range(desired_quat_wxyz.shape[0]):
            # Convert current quat from (w,x,y,z) → (x,y,z,w) for scipy
            q_curr = desired_quat_wxyz[i].cpu().numpy()  # (w,x,y,z)
            q_curr_xyzw = q_curr[[1, 2, 3, 0]]

            # Compose: delta applied in world frame → q_new = q_delta * q_curr
            q_new_xyzw = (Rotation.from_quat(dq_xyzw) * Rotation.from_quat(q_curr_xyzw)).as_quat()

            # Convert back to (w, x, y, z)
            q_new_wxyz = torch.tensor(
                [q_new_xyzw[3], q_new_xyzw[0], q_new_xyzw[1], q_new_xyzw[2]],
                dtype=desired_quat_wxyz.dtype,
                device=desired_quat_wxyz.device,
            )
            updated_quats.append(q_new_wxyz)

        desired_quat_wxyz = torch.stack(updated_quats, dim=0)

    return desired_pos, desired_quat_wxyz


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # Parse env config and disable timeout so teleoperation runs indefinitely
    env_cfg = parse_env_cfg(TASK_NAME, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.terminations.time_out = None

    # Create the environment
    env = gym.make(TASK_NAME, cfg=env_cfg).unwrapped

    # Create keyboard device; gripper_term=False because Reach has no gripper action
    sensitivity = args_cli.sensitivity
    keyboard = Se3Keyboard(
        Se3KeyboardCfg(
            pos_sensitivity=0.005 * sensitivity,
            rot_sensitivity=0.05 * sensitivity,
            gripper_term=False,
        )
    )

    # Reset callback ---------------------------------------------------
    should_reset = False

    def reset_env() -> None:
        nonlocal should_reset
        should_reset = True
        print("Reset triggered — environment will reset on next step.")

    keyboard.add_callback("R", reset_env)

    # Info
    print(keyboard)
    print(f"\nTask : {TASK_NAME}")
    print("Press 'R' to reset | 'L' to clear keyboard delta state.\n")

    # Initialise -------------------------------------------------------
    env.reset()
    keyboard.reset()

    # Seed desired pose from the current end-effector pose so there is no
    # discontinuous jump on the first action.
    desired_pos, desired_quat = get_ee_pose(env)

    # Simulation loop --------------------------------------------------
    while simulation_app.is_running():
        with torch.inference_mode():
            # 6-D delta: [dx, dy, dz, rx, ry, rz] — keyboard holds keys steady,
            # so the delta is re-applied every step while the key is held down.
            delta = keyboard.advance().to(desired_pos.device)  # (6,) → sim device

            # Integrate delta → new absolute desired pose
            desired_pos, desired_quat = integrate_delta(desired_pos, desired_quat, delta)

            # Build 7-D absolute action: [x, y, z, qw, qx, qy, qz]
            action = torch.cat([desired_pos, desired_quat], dim=-1)  # (num_envs, 7)

            env.step(action)

            if should_reset:
                env.reset()
                keyboard.reset()
                desired_pos, desired_quat = get_ee_pose(env)
                should_reset = False
                print("Environment reset complete.")

    env.close()
    print("Environment closed.")


if __name__ == "__main__":
    main()
    simulation_app.close()
