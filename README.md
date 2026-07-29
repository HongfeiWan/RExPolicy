# RExPolicy

RExPolicy is a Newton + GR00T rollout-experience policy project. Its minimum
data flywheel keeps the pretrained GR00T visual-language backbone frozen,
samples action chunks from the existing Flow-DiT, compares those chunks from
the same simulator state, and updates Flow-DiT from simulator-selected
experience. It does not require PPO, GAE, a critic, a state-policy teacher, or
new human demonstrations.

## Current system

The repository contains:

- the batched Newton Nero + Linker L10 environment and the validated 19D action
  representation;
- the versioned `reach_green_cap/v1` task, whose instruction, reward, success,
  failure, action projection, and reachability oracle share one contract;
- deterministic reset-and-replay reconstruction of a common root state across
  K candidate worlds, with seed-stable Reach bottle XY recipes;
- success-first continuation and chunk-level advantage selection;
- a frozen BF16 GR00T VLM encoded once per verified same-state decision, then
  expanded across K independent Flow-DiT noise samples; the trainable DiT keeps
  FP32 master parameters with BF16 sampling and training forward passes;
- exact globally weighted DDP updates, including zero-weight padding for
  unequal rank-local sample counts and 100% current-generation sample coverage;
- an append-only, metadata-only Success Archive with deterministic
  cross-generation round-robin replay;
- atomic full-state checkpoints, strict resume validation, per-rank logs,
  heartbeat/status files, and periodic GPU monitoring;
- a fixed K=1 held-out non-regression gate that evaluates the raw policy without
  same-state branch search.

Only executed effective actions are training targets. Episode JSONL stores
reset recipes, action prefixes, seeds, provenance, task/reward identifiers, and
metrics. The Success Archive stores only immutable references into those
records. Neither stores video, rendered frames, frozen VLM features, bottle
trajectories, or unexecuted action tails. Replaying a recipe in Newton
reconstructs the state and renders observations when historical experience is
used again.

See [the roadmap](roadmap/README.md), the [minimum flywheel
contract](roadmap/minimum_flywheel.md), the [DDP operations
guide](roadmap/bootstrap_ddp.md), and the preserved [final flywheel
design](roadmap/final_flywheel.md).

## Environment

Do not create or copy a project-local environment. Use the existing Newton
environment:

```bash
conda activate newton
cd /home/whf/Project/RExPolicy
```

The environment must already provide Newton, Warp, NumPy, PyTorch, and the
packages required by Isaac-GR00T. The default model locations are:

```text
checkpoints/groot/checkpoint-200000/
checkpoints/nvidia/Cosmos-Reason2-2B/
```

Isaac-GR00T source remains an external runtime dependency. Paths can be
overridden without copying environments or model artifacts:

```bash
export ISAAC_GROOT_ROOT=/home/whf/Project/Isaac-GR00T
export GROOT_POLICY_CHECKPOINT=/path/to/checkpoint-200000
export GROOT_VLM_MODEL=/path/to/Cosmos-Reason2-2B
```

See [checkpoints/README.md](checkpoints/README.md) for the distinction between
immutable base weights and resumable flywheel checkpoints. The optional
photorealistic room asset `scene/scene.glb` is not required by the Reach
contract; use `--no-scene-visuals` when it is absent.

## Validation

Run the focused flywheel unit suites from the repository root:

```bash
python -m unittest \
  tools.test_flywheel_bootstrap \
  tools.test_flywheel_trainer \
  tools.test_flywheel_archive_evaluation \
  tools.test_flywheel_checkpoint_operations \
  tools.test_groot_newton_reach
```

The production-physics Reach oracle is opt-in because it needs a CUDA Newton
runtime:

```bash
REXPOLICY_RUN_GPU_ORACLE=1 \
  python -m unittest tools.test_groot_newton_reach
```

Validate the distributed launcher and local artifacts without starting a
generation:

```bash
REXPOLICY_NPROC=1 tools/launch_flywheel_ddp.sh --validate-only
```

The operational two-GPU smoke, checkpoint-resume check, and staged 10→20
generation soak are specified in
[roadmap/bootstrap_ddp.md](roadmap/bootstrap_ddp.md). A smoke run proves
engineering closure, not policy improvement; K=1 held-out metrics determine
whether a candidate is retained as `last-good`.

## Repository layout

```text
rexpolicy/
  envs/                 Newton batched environments and task contracts
  flywheel/             Collection, replay, FP32/DDP training, evaluation, ops
  ik/                   Host and Newton IK/action helpers
  policies/             GR00T action representation contract
  robots/               Newton robot runtime
  retargeting/          Linker L10 hand configuration
  teleop/               Coordinate-frame helpers used by RTC
tools/                  Runtime entry points and unittest suites
debug/                  Migrated scene construction runtime
assets/                 Robot, hand, camera, and bottle assets
configs/                Scene physics configuration
roadmap/                Runnable protocol and long-term design
```

The migration history and boundaries are recorded in
[roadmap/migration_inventory.md](roadmap/migration_inventory.md). The original
Newton source tree remains independent.
