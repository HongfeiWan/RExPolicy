# RExPolicy

RExPolicy is the home of the Newton + GR00T rollout-experience policy project.
The repository currently contains the validated simulator and policy runtime
that were migrated out of the Newton development tree. It does **not** yet
contain the self-improving data-flywheel trainer.

## Current baseline

The migrated baseline provides:

- the batched Newton Nero + Linker L10 environment;
- GPU-resident observations, staged rewards, action-chunk execution, and reset;
- the existing GR00T VLM + Flow-DiT RTC runtime;
- the validated 19D Newton action contract and GR00T rotation conversion;
- robot, hand, bottle, camera, collision, and physics assets required by the
  minimal scene;
- focused reward, action-target, bottle-settle, and RTC alignment tests.

The current project decision is to freeze the GR00T VLM and retain the existing
Flow-DiT as the trainable action policy. PPO, a state-policy teacher, reward
populations, MAP-Elites, and automated GPT task authoring are deliberately not
part of the first implementation.

See [the roadmap](roadmap/README.md), the [minimum flywheel
specification](roadmap/minimum_flywheel.md), and the preserved [final flywheel
design](roadmap/final_flywheel.md).

## Environment

Do not create or copy a project-local environment. Use the existing environment:

```bash
conda activate newton
cd /home/whf/Project/RExPolicy
```

The active environment must already provide Newton, Warp, NumPy, PyTorch, and
the packages required by Isaac-GR00T.

Isaac-GR00T source and model weights stay external to this repository. The RTC
runner resolves these variables first:

```bash
export ISAAC_GROOT_ROOT=/home/whf/Project/Isaac-GR00T
export GROOT_POLICY_CHECKPOINT=/path/to/checkpoint-200000
export GROOT_VLM_MODEL=/path/to/Cosmos-Reason2-2B
```

On the current machine, the runner also recognizes the existing artifacts under
`../newton/checkpoints/` as a transitional fallback. No checkpoint, Conda
environment, `.venv`, rollout data, or cached output was copied into RExPolicy.

The optional photorealistic room asset `scene/scene.glb` was also not copied.
Use `--no-scene-visuals` for a minimal smoke test, or provide the asset locally
for visual-policy experiments. See [scene/README.md](scene/README.md).

## Smoke checks

Run the clean CPU/import regression test from the repository root:

```bash
python -m unittest tools.test_newton_groot_rtc_alignment
```

The other migrated modules under `tools/test_groot_newton_*.py` exercise Warp
kernels and simulator behavior. Run them individually when changing those
paths. With the current `newton` environment, Warp 1.14 can segfault during
Python interpreter teardown after the reward assertions have completed. The
same teardown failure is reproducible in the source Newton repository and is
not introduced by this migration.

Run a minimal GPU environment step without camera rendering:

```bash
python -m tools.run_newton_groot_rl_env \
  --num-envs 1 \
  --steps 1 \
  --obs-mode state \
  --no-images \
  --no-scene-visuals \
  --no-hydroelastic \
  --no-capture-graph
```

The existing GR00T runtime remains available through:

```bash
python -m tools.run_newton_groot_rtc_control \
  --image-source sim \
  --state-source sim \
  --instruction "pick up the bottle" \
  --start-policy
```

This command performs inference and control only. It does not collect elite
rollouts or update DiT.

## Repository layout

```text
rexpolicy/              Runtime Python package
  envs/                 Newton batched environments
  ik/                   Host and Newton IK/action helpers
  policies/             GR00T action representation contract
  robots/               Newton robot runtime
  retargeting/          Linker L10 hand configuration
  teleop/               Coordinate-frame helpers used by RTC
tools/                  Runtime entry points and focused tests
debug/                  Migrated scene construction runtime
assets/                 Minimal robot, hand, camera, and bottle assets
configs/                Scene physics configuration
roadmap/                MVP and long-term flywheel design
```

## Migration boundary

The exact inventory and intentionally excluded components are recorded in
[roadmap/migration_inventory.md](roadmap/migration_inventory.md). The original
Newton files remain untouched.
