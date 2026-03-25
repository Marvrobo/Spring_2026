Spring 2026 UROP

To run the keyboard control (SE(3)) for the Franka robot:
```bash
python scripts/environments/teleoperation/teleop_se3_agent.py --task Isaac-Lift-Cube-Franka-IK-Rel-v0 --teleop_device keyboard --num_envs 1
```

Note: `teleop_se3_agent.py` sends relative SE(3) actions, so use `IK-Rel` tasks (e.g., `Isaac-Lift-Cube-Franka-IK-Rel-v0`).