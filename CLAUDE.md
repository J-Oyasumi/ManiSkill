# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

ManiSkill is a GPU-parallelized robotics simulation and training framework built on SAPIEN. It supports CPU and GPU simulation, GPU-accelerated rendering (RGBD + segmentation at 30K+ FPS), and unified task-building APIs. The codebase is designed to seamlessly scale from single-threaded CPU to massively parallel GPU execution.

## Development Setup

### Installation from source
```bash
conda create -n "ms_dev" "python==3.9"
conda activate ms_dev
git clone https://github.com/haosulab/ManiSkill.git
cd ManiSkill
pip install -e .[dev]  # install with dev dependencies
```

### Setup pre-commit hooks
```bash
pip install pre-commit
pre-commit install
```

Pre-commit runs: black (line-length=88), isort (profile=black), autoflake, and basic checks. Excludes: `warp_maniskill/`, `docs/`, `examples/`.

## Testing

Tests use pytest with custom markers for GPU simulation and slow tests:

```bash
# Run CPU simulation tests
pytest tests/ -m "not slow and not gpu_sim"

# Run GPU simulation tests (separate session required)
pytest tests/ -m "not slow and gpu_sim"

# Run a single test
pytest tests/test_envs.py::test_env_reset -v

# Run tests with parallelization (for independent tests)
pytest tests/ -n auto -m "not slow and not gpu_sim"
```

**Important**: CPU and GPU simulation cannot run in the same pytest session. GPU simulation tests are marked with `@pytest.mark.gpu_sim`.

## Building and Publishing

```bash
# Build wheel
python -m build -s -w

# Upload to test PyPI
python -m twine upload --repository testpypi dist/*

# Upload to PyPI
python -m twine upload dist/*
```

## Core Architecture

### Three-Layer Design

1. **Physics Backend** ([scene.py](mani_skill/envs/scene.py)):
   - `ManiSkillScene` wraps SAPIEN scenes and manages `PhysxCpuSystem` or `PhysxGpuSystem`
   - For CPU: single scene (`num_envs=1`)
   - For GPU: multiple `sub_scenes` (one per parallel env), all sharing single `PhysxGpuSystem`

2. **Environment Framework** ([sapien_env.py](mani_skill/envs/sapien_env.py)):
   - `BaseEnv(gym.Env)`: Foundation for all tasks
   - Contains: `scene`, `agent`, sensors, action/observation spaces
   - Key methods: `_reconfigure()`, `_initialize_episode()`, `reset()`, `step()`

3. **Task Implementations** ([envs/tasks/](mani_skill/envs/tasks/)):
   - Tasks inherit from `BaseEnv` and implement: `_load_scene()`, `_initialize_episode()`, `evaluate()`, reward functions
   - See [template.py](mani_skill/envs/template.py) for task structure

### Struct Wrappers ([utils/structs/](mani_skill/utils/structs/))

Unified API for CPU/GPU objects:
- `Actor`: Wraps rigid bodies (`sapien.Entity`)
- `Articulation`: Wraps articulated objects like robots (`physx.PhysxArticulation`)
- `Link`: Individual links in articulation chains
- `Pose`: Position + quaternion representation

All structs maintain `_objs` list indexed by `_scene_idxs` to map to parallel environments.

### Agent System ([agents/](mani_skill/agents/))

```python
BaseAgent
├── robot: Articulation           # Physical robot body
├── controllers: Dict[str, BaseController]  # Control interfaces
├── sensors: Dict[str, BaseSensor]  # Robot-mounted sensors
└── keyframes: Dict[str, Keyframe]  # Predefined poses
```

Common controllers: `PDJointPosController`, `PDJointVelController`, `PDEEPoseController` (with IK), `PassiveController`.

### GPU Parallelization

**Key concept**: Single GPU process simulates multiple environments in parallel.

**Implementation**:
- GPU buffers store all env data contiguously: `px.cuda_rigid_body_data`, `px.cuda_articulation_qpos`
- Data flow: `gpu_apply_*()` (CPU → GPU before step), `gpu_fetch_*()` (GPU → CPU after step)
- Each struct has `_body_data_index` mapping to GPU buffer indices
- Partial reset support via `scene._reset_mask` boolean tensor

**Memory config**: Set via `SimConfig.gpu_memory_config` (contact counts, collision patches, broadphase pairs).

### Two-Stage Reset Pattern

1. **Reconfigure** (`_reconfigure()`): Rare, expensive
   - Deletes entire PhysX scene and reloads all assets
   - Called on first reset or every N resets via `reconfiguration_freq`

2. **Episode Initialize** (`_initialize_episode()`): Every reset
   - Resets poses/velocities, randomizes objects/goals
   - Fast, no memory reallocation

### Rendering and Sensors

- **Cameras**: Mount options (fixed in scene or attached to Actor/Link)
- **Shader configs**: "minimal", "default", "rt" (ray-tracing)
- **GPU-parallel rendering**: Batch cameras across parallel envs
- **Texture outputs**: RGB, depth, segmentation, normal, position, albedo

**Rendering pipeline**: `scene.update_render()` → `sensor.capture()` → `sensor.get_obs()` → `env.render()`

### Observation and Reward Modes

**Observation modes**: `"state"` (flat vector), `"state_dict"` (structured), `"sensor_data"`, `"rgb"`, `"depth"`, `"rgbd"`, `"pointcloud"`

**Reward modes**: `"sparse"` (binary 0/1), `"dense"` (continuous), `"normalized_dense"` (dense normalized to [0, 1]), `"none"` (always 0)

### Trajectory Recording and Replay

- **Recording**: Use `RecordEpisode` wrapper, saves to HDF5 (`.h5`) with `.json` metadata
- **Dataset**: `ManiSkillTrajectoryDataset` - PyTorch dataset for loading trajectories
- **Replay**: [replay_trajectory.py](mani_skill/trajectory/replay_trajectory.py) - supports action replay or exact state replay

```bash
# Replay trajectory
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path demos/trajectory.h5 \
  --save-video
```

## Task Registration

Tasks are registered via decorator:

```python
from mani_skill.utils.registration import register_env

@register_env("TaskName-v1", max_episode_steps=200)
class TaskName(BaseEnv):
    ...
```

This creates an `EnvSpec` in `REGISTERED_ENVS` and enables `gymnasium.make("TaskName-v1")`.

## Common Development Commands

### Running environments

```bash
# Basic environment interaction
python -c "import gymnasium as gym; import mani_skill; env = gym.make('PickCube-v1'); env.reset(); env.step(env.action_space.sample())"

# GPU parallelized simulation
python -c "import gymnasium as gym; import mani_skill; env = gym.make('PickCube-v1', num_envs=1024); env.reset(); env.step(env.action_space.sample())"

# With rendering
python -c "import gymnasium as gym; import mani_skill; env = gym.make('PickCube-v1', render_mode='human'); env.reset(); [env.step(env.action_space.sample()) for _ in range(100)]"
```

### Generating demonstrations

```bash
# Motion planning demos
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path demos/PickCube-v1/trajectory.h5 \
  --save-video

# RL training for demos (see scripts/data_generation/rl.sh)
python -m mani_skill.examples.benchmarking.ppo \
  --env-id="PickCube-v1" \
  --num-envs=1024 \
  --total-timesteps=10000000
```

### Baseline Training

Training scripts are in [examples/baselines/](examples/baselines/).

```bash
# PPO training
cd examples/baselines/ppo
python ppo.py --env-id="PickCube-v1" --num-envs=1024

# SAC training
cd examples/baselines/sac
python sac.py --env-id="PickCube-v1" --num-envs=128

# Behavior cloning from demos
cd examples/baselines/bc
python bc.py --env-id="PickCube-v1" --demo-path=path/to/demos.h5

# Diffusion Policy
cd examples/baselines/diffusion_policy
python train.py --env-id="PickCube-v1" --demo-path=path/to/demos.h5
```

See [examples/baselines/README.md](examples/baselines/README.md) for baseline details.

## Code Style and Conventions

### Python Style
- Black formatter with line length 88
- isort with black profile
- Type hints encouraged but not required for all functions

### Key Patterns

**Batched operations**: All operations should work with batched tensors when `num_envs > 1`.

**Device handling**: Code should be device-agnostic. Use `self.device` from environment/scene.

**RNG system**:
- `_main_rng`: Seeds episode sequences (internal)
- `_batched_episode_rng`: For task randomization (user-facing), indexed by env: `rng[env_idx]`

**State dicts**: Save/load full environment state:
```python
state_dict = {
    "actors": {name: pose/velocity for each actor},
    "articulations": {name: qpos/qvel for each articulation}
}
```

### Adding New Tasks

Follow the tutorial: https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/custom_tasks/index.html

For tasks to be added to the official repo, see: https://maniskill.readthedocs.io/en/latest/contributing/tasks.html

Key requirements:
1. Inherit from `BaseEnv`
2. Implement required methods: `_load_scene()`, `_initialize_episode()`, `evaluate()`
3. Register with `@register_env()` decorator
4. Add tests to `tests/test_envs.py`
5. Provide task documentation

## Project Structure

```
mani_skill/
├── envs/
│   ├── sapien_env.py       # BaseEnv foundation
│   ├── scene.py            # ManiSkillScene manager
│   ├── template.py         # Task template
│   ├── tasks/              # Task implementations
│   └── scenes/             # Scene configurations
├── agents/
│   ├── base_agent.py       # Agent interface
│   ├── controllers/        # Control schemes
│   └── robots/             # Robot definitions (URDF/config)
├── sensors/
│   └── camera.py           # Camera sensor
├── utils/
│   ├── structs/            # Actor/Articulation wrappers
│   ├── registration.py     # Task registration
│   ├── building/           # Asset builders
│   └── wrappers/           # Environment wrappers
├── trajectory/
│   ├── dataset.py          # Trajectory dataset
│   └── replay_trajectory.py # Replay script
└── vector/                 # Vectorization utilities

examples/
├── baselines/              # RL/IL baseline implementations
└── tutorials/              # Tutorial notebooks

tests/                      # Pytest test suite
scripts/
└── data_generation/        # Demo generation scripts
```

## Documentation

Full documentation: https://maniskill.readthedocs.io/

Key pages:
- Installation: https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html
- Quickstart: https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/quickstart.html
- Custom tasks: https://maniskill.readthedocs.io/en/latest/user_guide/tutorials/custom_tasks/index.html
- RL baselines: https://maniskill.readthedocs.io/en/latest/user_guide/reinforcement_learning/index.html
- IL baselines: https://maniskill.readthedocs.io/en/latest/user_guide/learning_from_demos/index.html

## System Requirements

Best support on Linux with NVIDIA GPU. Limited support on Windows/MacOS.

| System / GPU         | CPU Sim | GPU Sim | Rendering |
| -------------------- | ------- | ------- | --------- |
| Linux / NVIDIA GPU   | ✅      | ✅      | ✅        |
| Windows / NVIDIA GPU | ✅      | ❌      | ✅        |
| Windows / AMD GPU    | ✅      | ❌      | ✅        |
| WSL / Anything       | ✅      | ❌      | ❌        |
| MacOS / Anything     | ✅      | ❌      | ✅        |

Vulkan setup required for rendering: https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#vulkan
