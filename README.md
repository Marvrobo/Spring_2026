Spring 2026 UROP

To run the keyboard control (SE(3)) for the Franka robot:
```bash
python scripts/environments/teleoperation/teleop_se3_agent.py --task Isaac-Lift-Cube-Franka-IK-Rel-v0 --teleop_device keyboard --num_envs 1
```

Note: `teleop_se3_agent.py` sends relative SE(3) actions, so use `IK-Rel` tasks (e.g., `Isaac-Lift-Cube-Franka-IK-Rel-v0`).

Under `source/`, there are four submodules:
+ `isaaclab`: the core framework package that defines the reusable primitives that the rest of the repo builds on top of (i.e., simulation abstractions, scene management, robot and articulation interfaces, actuator models, sensor interfaces, environment base classes, Utilities for configs, math, IO, rendering hooks, and simulation helpers.)
+ `isaaclab_assets`: the package holds reusable asset configurations (preconfigured robot definitions, sensor configuration objects, path and config wrappers for USD assets), so that `isaaclab` defines what is a robot or sensor, and this folder defines concrete configurations for them.
+ `isaaclab_tasks`: this package contains the actual environments and benchmark tasks (task definitions, environment configurations, Gym registration, robot-specific variants of a task, agent/training configurations, manager-based and direct environment structures depending on the task style)
+ `isaaclab_rl`: this package is the adapter layer between Isaac Lab environments and external reinforcement learning libraries (wrappers for various RL frameworks)
+ `isaaclab_mimic`: the package for imitation learning data collection pipeline. 
+ `isaaclab_contrib`: this package is for community and experimental extension (actuator models, asset types, experimental MDP terms, niche or emerging sensor integrations, etc.)
