Spring 2026 UROP

To run the keyboard control (SE(3)) for the Franka robot:
```bash
python scripts/environments/teleoperation/teleop_se3_agent.py --task Isaac-Reach-Franka-IK-Rel-v0 --teleop_device keyboard --num_envs 1
```

Isaac-Reach-Franka-IK-Rel-v0

Note: `teleop_se3_agent.py` sends relative SE(3) actions, so use `IK-Rel` tasks (e.g., `Isaac-Lift-Cube-Franka-IK-Rel-v0`).

Under `source/`, there are four submodules:
+ `isaaclab`: the core framework package that defines the reusable primitives that the rest of the repo builds on top of (i.e., simulation abstractions, scene management, robot and articulation interfaces, actuator models, sensor interfaces, environment base classes, Utilities for configs, math, IO, rendering hooks, and simulation helpers.)
+ `isaaclab_assets`: the package holds reusable asset configurations (preconfigured robot definitions, sensor configuration objects, path and config wrappers for USD assets), so that `isaaclab` defines what is a robot or sensor, and this folder defines concrete configurations for them.
+ `isaaclab_tasks`: this package contains the actual environments and benchmark tasks (task definitions, environment configurations, Gym registration, robot-specific variants of a task, agent/training configurations, manager-based and direct environment structures depending on the task style)
+ `isaaclab_rl`: this package is the adapter layer between Isaac Lab environments and external reinforcement learning libraries (wrappers for various RL frameworks)
+ `isaaclab_mimic`: the package for imitation learning data collection pipeline. 
+ `isaaclab_contrib`: this package is for community and experimental extension (actuator models, asset types, experimental MDP terms, niche or emerging sensor integrations, etc.)

We use the absolute control for the end-effector since we might want to try to reduce action space, to test the absolute action space:
```python scripts/environments/zero_agent.py --task Isaac-Reach-Franka-IK-Abs-v0 --num_env 1```

To use keyboard to control the end-effector in absolute action space:
```python scripts/environments/teleoperation/teleop_se3_agent_ik_abs.py --num_env 1```

To train a RL agent with trivial single environment:
```python scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Push-T-Franka-v0 --num_env 1 ```

To train a RL agent using parallel environment in headless mode:
```python scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Push-T-Franka-v0 --num_env 32 --headess```



