# Migration Inventory

## Source

The baseline was copied from `/home/whf/Project/newton`. Source files remain in
place. Imports were renamed from `teleop_stack` to the `rexpolicy` package.

## Included

- `rexpolicy/envs`: Newton batched environment and action-chunk wrappers.
- `rexpolicy/ik`, `rexpolicy/robots`, and `rexpolicy/teleop`: direct RTC and EEF
  control dependencies.
- `rexpolicy/policies/groot_rotation_contract.py`: validated GR00T EEF rotation
  representation.
- `rexpolicy/retargeting/hand_config.py`: Linker L10 joint and mimic contract.
- `tools/run_newton_groot_rtc_control.py`: existing GR00T VLM + Flow-DiT runtime.
- `tools/run_newton_groot_rl_env.py`: batched Newton environment smoke runner.
- Focused tests for reward, action targets, bottle settling, finger load, and RTC.
- The generated robot URDF, Nero meshes, Linker L10 URDF/meshes, bottle visual,
  camera configurations, collision boxes, and scene physics configuration.

## Initially excluded

- `.venv`, Conda files, package caches, and Python bytecode.
- GR00T and Cosmos model weights (approximately several GB). These were copied
  into Git-ignored local checkpoint directories when Phase 1 work began.
- old compact-DP checkpoints, Residual PPO checkpoints, and training outputs.
- LeRobot/smooth datasets, rollout data, debug logs, and images.
- the optional 168 MB `scene/scene.glb` room visual.
- compact Diffusion Policy and Residual PPO training code, because the current
  architecture retains the existing GR00T Flow-DiT.
- the data-flywheel collector, replay store, trainer, and promotion service.
  These were not migrated from Newton: RExPolicy now implements its own
  same-state collector, episode and Success Archive JSONL shards, FP32/DDP DiT
  trainer, atomic checkpoint/resume, operations logging, and K=1 evaluation
  gate.

## External runtime locations

The RTC runner accepts:

```text
ISAAC_GROOT_ROOT
GROOT_POLICY_CHECKPOINT
GROOT_VLM_MODEL
GROOT_SMOOTH_DATASET
```

RExPolicy-local checkpoint paths are preferred and now populated on this
machine. The sibling `Isaac-GR00T` repository supplies the Python source without
copying its `.venv`, datasets, outputs, or other large development artifacts.

## Migration verification

- All migrated Python modules compile and import in the `newton` Conda
  environment.
- Ruff static checks pass for `rexpolicy`, `tools`, and `debug`.
- The RTC alignment suite passes all 13 tests.
- A one-world Newton GPU environment loads the migrated URDF, meshes, bottle,
  collision data, and scene physics and completes a simulator step without
  camera rendering.
- The reward test completes all 25 assertions, then Warp 1.14 may segfault
  during interpreter teardown. The same behavior is reproducible in the source
  Newton repository, so it is tracked as an environment/runtime issue rather
  than a migration regression.
- `reach_green_cap/v1` has a dedicated instruction/reward/success contract,
  effective-action projection, replay fingerprint, and opt-in
  production-physics GPU reachability oracle.
- Flywheel unit suites cover globally weighted sample scheduling, successful
  reference replay, held-out gate logic, FP32 optimizer state, atomic
  checkpoint integrity, GPU monitoring, and launch preflight.
