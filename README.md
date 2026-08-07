# RExPolicy

RExPolicy is a Newton + GR00T rollout-experience policy project. It follows one
ordered delivery route defined in [the canonical roadmap](roadmap/README.md).
The only current learner is the Phase 1 `reach_green_cap/v1` flywheel: it keeps
the pretrained GR00T visual-language backbone frozen, samples action chunks
from the existing Flow-DiT, compares those chunks from the same simulator
state, and updates Flow-DiT from simulator-selected experience. It does not
require PPO, GAE, a critic, a state-policy teacher, or new human demonstrations.

Phase 1 code is implemented, but the phase is not complete until the canonical
node3 two-GPU smoke, fresh-process resume, and staged 10→20 generation soak
pass. The canonical base policy is
`checkpoints/groot/checkpoint-200000/`; runs using another base checkpoint are
experiments rather than release evidence.

`RExPolicy v2` names Phase 2 of the same roadmap, not a parallel route. Its
Success Manifold graph, encoder, selector, diagnostic projection, Flow-DiT token
conditioning, and bounded latent-memory code are implemented behind default-off
contracts. Phase 2 is neither runtime-validated nor active: it still requires a
real canonical success corpus, a trained checksum-pinned bundle, shadow-mode
evaluation, and the prescribed GPU acceptance gates. With no Phase 2 flags, the
Phase 1 path is unchanged.

A separate **SM-R** research candidate is planned, but not implemented. It
adapts progress-aligned retrieval from successful trajectories and bounded,
frequency-selective action correction to the existing Success Manifold. SM-R
is default-off and cannot change the Phase 1 path or the SM-1 through SM-5
activation gates.

The repository also contains an independent, state-only Stage 0 scientific
validation harness. It is not the canonical active learner or a substitute for
the ordered release gates above. Its reward-free Grasp-Lift close-at-step-7
corpus and train-only artifact are accepted, and its fixed no-z learner,
checkpoint/resume path, tensor-free three-run preflight, one-time validation
claim, exact Newton evaluator, and checkpoint selector are implemented. All
three 10,000-step train-only seeds completed on node1. The first one-time
validation attempt was burned by an evaluator serialization defect after its
six shards were opened but before any checkpoint, metric, or rollout was
evaluated. That cohort cannot be reused. An independent v2 six-member cohort
was subsequently authored under a pre-CUDA write-once claim, and the
claim-bound selector evaluated all 60 registered checkpoint models. The formal
gate passed: the earliest longest eligible run spans steps 2,000--5,000, and
the fixed rule selected the three-seed family at step 3,500. That family
completed 72/72 validation rollout cells with zero safety, integrity, or oracle
disagreement events. This supports fixed-cohort state-only Grasp-Lift
feasibility; it does not activate Phase 2 or establish visual or locked-test
generalization. No locked test exists or has been consumed.

The repository also contains later-phase infrastructure developed ahead of
activation: reward-free Event Ledgers, rebuildable quality-diversity views, a
fail-closed TaskSpec control plane, provider-attested authoring to static
quarantine, and signed lifecycle persistence. None of it may alter training
until the preceding roadmap phases and its own activation gates pass.

## Current status

### Active Phase 1 implementation

The runnable Phase 1 path contains:

- the batched Newton Nero + Linker L10 environment and the validated 19D action
  representation;
- the versioned `reach_green_cap/v1` task, whose instruction, reward, success,
  failure, action projection, and reachability oracle share one contract;
- deterministic reset-and-replay reconstruction of a common root state across
  K candidate worlds, with seed-stable Reach bottle XY recipes;
- success-first continuation and chunk-level advantage selection;
- a frozen BF16 GR00T VLM and a trainable FP32 Flow-DiT, with BF16 autocast for
  sampling and training forward passes;
- exact globally weighted DDP updates, including zero-weight padding for
  unequal rank-local sample counts and 100% current-generation sample coverage;
- an append-only, metadata-only Success Archive with deterministic
  cross-generation round-robin replay;
- atomic full-state checkpoints, strict resume validation, per-rank logs,
  heartbeat/status files, and periodic GPU monitoring;
- a fixed K=1 held-out non-regression gate that evaluates the raw policy without
  same-state branch search.

### Implemented but inactive

- the independent Stage 0 state-only Newton adapter, immutable trajectory
  corpus, no-z Grasp-Lift learner, and one-time selection path;
- Phase 2 metadata-only SuccessExperience graphs, future encoder, selector,
  diagnostic projection, optional success-token conditioning, and bounded
  successful-latent memory;
- Phase 4 quality-balanced archive planning bound to immutable behavior
  descriptors and external replay state, which is not yet wired into the
  trainer checkpoint;
- Phase 5 authoring, quarantine, audit, and lifecycle control-plane primitives.

### Planned, default-off research

- **SM-R:** a frozen, content-addressed successful-trajectory memory; causal
  history-to-progress alignment; and a clipped low-frequency residual on
  motion channels only. The unmodified Flow-DiT proposal remains the explicit
  abstention path, and Newton projection, safety, and the TaskSpec oracle remain
  authoritative. Validation, exposed or burned cohorts, and locked tests may
  never populate the memory or tune its hyperparameters. See the
  [Phase 2 research contract](roadmap/success_manifold_v2.md).

Only executed effective actions are training targets. Episode JSONL stores
reset recipes, action prefixes, seeds, provenance, task/reward identifiers, and
metrics. The Success Archive stores only immutable references into those
records. Neither stores video, rendered frames, frozen VLM features, bottle
trajectories, or unexecuted action tails. Replaying a recipe in Newton
reconstructs the state and renders observations when historical experience is
used again.

The [canonical roadmap](roadmap/README.md) is authoritative for phase order,
status, defaults, and release gates. Supporting details live in the [minimum
flywheel contract](roadmap/minimum_flywheel.md), [DDP operations
guide](roadmap/bootstrap_ddp.md), [Success Manifold Phase 2
contract](roadmap/success_manifold_v2.md), [automatic task authoring trust
boundary](roadmap/automatic_task_authoring.md), and preserved [final flywheel
design](roadmap/final_flywheel.md). The isolated state-only research harness is
documented in [Stage 0 Success Manifold
Validation](roadmap/stage0_success_manifold_validation.md).

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
  tools.test_groot_newton_reach \
  tools.test_rollout_recorder
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

The dated [node1 capacity report](roadmap/node1_gpu_capacity_20260804.md) is
scaling evidence only. Its K=256 candidate and historical noncanonical
checkpoint do not replace the Phase 1 defaults or node3 acceptance gates.

Record a deterministic, inference-only base or DiT-generation rollout without
creating an optimizer or writing the training archive:

```bash
CUDA_VISIBLE_DEVICES=1 python -m tools.run_rollout_recorder \
  --device cuda:0 \
  --output-dir /path/to/new-rollout-directory \
  --reset-seed 20260722 \
  --diffusion-seed 20260723 \
  --execution-horizon 2
```

The recorder writes annotated ego, wrist, and side-by-side MP4 files. See the
[rollout recorder protocol](roadmap/rollout_recorder.md) for node3 paths,
generation overlays, fixed-seed comparisons, and the output contract.

## Repository layout

```text
rexpolicy/
  envs/                 Newton batched environments and task contracts
  flywheel/             Collection, replay, FP32/DDP training, evaluation, ops
  ik/                   Host and Newton IK/action helpers
  manifold/             Optional success representation, selector, and runtime
  policies/             GR00T action representation contract
  replay/               Metadata-only successful-future graph views
  robots/               Newton robot runtime
  tasking/              TaskSpec, reward/process views, authoring, lifecycle
  retargeting/          Linker L10 hand configuration
  teleop/               Coordinate-frame helpers used by RTC
tools/                  Runtime entry points and unittest suites
debug/                  Migrated scene construction runtime
assets/                 Robot, hand, camera, and bottle assets
configs/                Scene physics configuration
roadmap/                Canonical release sequence, contracts, and evidence
```

The migration history and boundaries are recorded in
[roadmap/migration_inventory.md](roadmap/migration_inventory.md). The original
Newton source tree remains independent.
