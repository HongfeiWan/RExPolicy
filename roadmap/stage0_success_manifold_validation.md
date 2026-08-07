# RExPolicy Stage 0 — State-only Success Manifold Validation

Status: **engineering loop implemented; scientific manifold gate not yet passed**.

## Node1 Grasp-Lift authoring snapshot — 2026-08-06

The reward-free Grasp-Lift authoring gate passed on node1 CUDA0 at Git commit
`50496ad8caff07cdb91d888199f53c60679daaec`. This is a verified data-authoring
result, not a learned-policy convergence claim:

- the only intervention selected from train evidence was
  `minimum_tentative_close_step: 4 -> 7`; every other authoring-mode field
  remained fixed;
- the committed corpus contains 32/32 oracle successes, zero failures, zero
  timeouts, zero safety violations, and zero collision-buffer overflows;
- the formal denominator is 30 trajectories: train is 24/24, and the six
  previously unseen validation seeds are 6/6; all 15 baseline train successes
  were retained, all nine baseline train timeouts converted, all four eligible
  baseline validation successes were retained, and both eligible validation
  timeouts converted;
- validation seeds 7001 and 7005 were exposed by two engineering smokes before
  the formal claim existed. They remain in their original validation split but
  are permanently `formal_gate_eligible=false`, are recorded in the hash-bound
  exposure ledger, and contribute only diagnostics;
- an exclusive validation claim was consumed before Newton startup within the
  node1 Git common-dir scope. Locked test data remains uncreated and
  unconsumed;
- all trajectories formed a tentative and frozen grasp at close step 7, kept
  opposed contact through termination, and stored actions exactly matched the
  projected/executed 19D actions;
- final lift was 0.0348--0.0408 m, maximum final lateral displacement was
  0.0124 m, and maximum final bottle tilt was 0.1105 rad, all inside the fixed
  independent oracle limits;
- 192 focused Stage 0 tests passed from the clean node1 worktree before the
  formal run, and an independent post-publication strict reload passed.

The committed artifact is retained at
`/home/user/project/RExPolicy-grasp-lift-close7/outputs/stage0/grasp-lift-pilot-close7-20260806-b`.
Its manifest SHA-256 is
`7afd7bc74dc223b6101d3add4b4aac1b47959b547b6a983c906cebe09b136cf3`,
and its composite corpus SHA-256 is
`16d438d79e2fb37505e2f2b070842095d025b4ce338c1f472765d10872701620`.

The next gate is learning from this corpus: fit normalization and construct
windows from the 24 train trajectories only, train checkpoints without any
validation feedback, select once on the six eligible validation trajectories,
and keep the two engineering-smoke exclusions out of both training and model
selection. No locked test may be created until that protocol and a genuinely
new held-out cohort are pre-registered.

The pilot intentionally ran state-only with scene visuals and camera textures
disabled. The missing full `scene.glb` therefore does not weaken its mechanics
or oracle evidence, but a real scene asset is still required before any visual
rollout can be presented as camera-valid evidence; a synthetic background must
not be substituted.

## Frozen node1 no-z learning protocol — 2026-08-07

The learning and selection software is implemented, but this section records a
runtime protocol, not a completed learning result. The final checkout must be
clean and remain on one Git HEAD from the evaluator smoke through all three
training runs and one-time selection. The implementation fingerprint contains
that HEAD plus the Python, Torch, Warp, Newton, and MuJoCo-Warp versions. A
pull, commit, runtime change, different world size, or untracked file during
the cohort invalidates comparability and must fail closed.

Node1 has one visible NVIDIA RTX PRO 5000 72GB GPU. "Use all GPUs" therefore
means CUDA0 with `world_size=1`; it does not mean launching a fictitious
multi-rank job. Each process is pinned to the same first 16 available CPUs.
When the node exposes the measured 32-CPU cpuset, the union remains one half of
the CPUs even after all three seed processes start.

The immutable inputs are:

```text
pilot manifest: 7afd7bc74dc223b6101d3add4b4aac1b47959b547b6a983c906cebe09b136cf3
pilot corpus:   16d438d79e2fb37505e2f2b070842095d025b4ce338c1f472765d10872701620
train artifact: a0b2618e800b9b5a2534d979dff387025a09a910a30be21d0e1791dcb67a715e
train content:  57f5e492dc4365ef71c425231aa1c2ee4ab7ed9261ffc2e4fb08ee799ff22425
normalization:  aa2a9d9bd37637046df8ffe8a46aabb51a1ac50629705a847396aa52d2993dad
```

Before touching validation, activate the existing Newton environment and run
the evaluator parity smoke against six train resets and four fixed policy-noise
variants. The output path must be new. This exercises the exact 24-world,
72-control-step, state-only Newton rollout stack with a fixed random policy;
success and ordinary task-safety failures are deliberately not gates. The gate
requires exact budget binding, zero execution-integrity faults (including any
nonzero reward), zero oracle-input disagreements, and zero collision-buffer
overflow. It never opens a validation shard or creates a claim.

```bash
cd /home/user/project/RExPolicy-grasp-lift-training
test -z "$(git status --porcelain --untracked-files=all)" || {
  echo "refuse dirty scientific worktree" >&2
  exit 1
}
git rev-parse HEAD
python - <<'PY'
import os
available = sorted(os.sched_getaffinity(0))
assert len(available) == 32, available
print(available)
PY
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader

CUDA_VISIBLE_DEVICES=0 python -u -m tools.smoke_stage0_grasp_lift_no_z_evaluator \
  --config /home/user/project/RExPolicy-grasp-lift-training/configs/stage0/grasp_lift_no_z_bc.json \
  --pilot-directory /home/user/project/RExPolicy-grasp-lift-close7/outputs/stage0/grasp-lift-pilot-close7-20260806-b \
  --training-artifact /home/user/project/RExPolicy-grasp-lift-training/outputs/stage0/grasp-lift-training-close7-20260806-a \
  --output /home/user/project/RExPolicy-grasp-lift-training/outputs/stage0/grasp-lift-no-z-evaluator-smoke-20260807-a.json \
  --device cuda:0
```

Proceed only when the persisted smoke record is self-hash valid, has
`state="passed"`, reports 24 episodes, and records the same Git HEAD intended
for training. A failed smoke may be diagnosed and repaired using train data;
because the repair changes HEAD, it requires a new smoke output path. It still
must not consume validation.

Formal training uses seeds `31001`, `31002`, and `31003`, global batch 2,048,
10,000 optimizer steps, a checkpoint at step zero and every 500 steps, and no
validation feedback. Console logs and PID files live outside each run root so
the runner can create that root exclusively. Start seed 31001 first. After its
step-50 heartbeat proves the memory/runtime contract, start 31002 and 31003 as
independent processes on the same 72GB GPU to improve utilization without
changing the registered `world_size=1` semantics.

```bash
cd /home/user/project/RExPolicy-grasp-lift-training
PYTHON="$(command -v python)"
PILOT=/home/user/project/RExPolicy-grasp-lift-close7/outputs/stage0/grasp-lift-pilot-close7-20260806-b
ARTIFACT=/home/user/project/RExPolicy-grasp-lift-training/outputs/stage0/grasp-lift-training-close7-20260806-a
RUNROOT=/home/user/project/RExPolicy-grasp-lift-training/outputs/stage0
LOGROOT="$RUNROOT/grasp-lift-no-z-launch-logs-20260807-a"
mkdir -p "$LOGROOT"

launch_seed() {
  seed="$1"
  out="$RUNROOT/grasp-lift-no-z-seed-${seed}-20260807-a"
  test ! -e "$out" && test ! -L "$out" || {
    echo "refuse existing output: $out" >&2
    return 1
  }
  nohup env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
    "$PYTHON" -u -m tools.run_stage0_grasp_lift_no_z \
    --config configs/stage0/grasp_lift_no_z_bc.json \
    --pilot-directory "$PILOT" \
    --training-artifact "$ARTIFACT" \
    --output-directory "$out" \
    --training-seed "$seed" \
    >"$LOGROOT/seed-${seed}.console.log" 2>&1 &
  printf '%s\n' "$!" >"$LOGROOT/seed-${seed}.pid"
}

launch_seed 31001
```

Inspect seed 31001's `status.json`, `heartbeat.jsonl`, console, GPU
memory/utilization, and `checkpoint-000000000000`. Only after a valid step-50
heartbeat, deliberately run the second launch block in the same shell:

```bash
launch_seed 31002
launch_seed 31003
```

If measured concurrent memory plus operating margin approaches 60 GiB, or a
run exits, do not delete or recreate its directory. Preserve the error record
and use `--resume` on that exact seed/output only; the manager resumes from the
latest complete 500-step checkpoint (step zero is also complete). Changing
HEAD, Python/runtime versions, CUDA visibility, world size, CPU affinity,
config, pilot, or artifact makes resume illegal.

Only after every run has exactly the 21 expected checkpoints and a complete
status may the tensor-free preflight run. The selection runner then persists a
global exclusive claim before opening the six validation shards and evaluates
the fixed 60-model by 24-rollout grid once. That claim is irreversible:

```bash
cd /home/user/project/RExPolicy-grasp-lift-training
PYTHON="$(command -v python)"
RUNROOT=/home/user/project/RExPolicy-grasp-lift-training/outputs/stage0
PILOT=/home/user/project/RExPolicy-grasp-lift-close7/outputs/stage0/grasp-lift-pilot-close7-20260806-b
ARTIFACT="$RUNROOT/grasp-lift-training-close7-20260806-a"
LOGROOT="$RUNROOT/grasp-lift-no-z-launch-logs-20260807-a"
SELECTOUT="$RUNROOT/grasp-lift-no-z-selection-20260807-a"
test ! -e "$SELECTOUT" && test ! -L "$SELECTOUT" || {
  echo "refuse existing selection output: $SELECTOUT" >&2
  exit 1
}

nohup env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  "$PYTHON" -u -m tools.run_stage0_grasp_lift_no_z_selection \
  --config configs/stage0/grasp_lift_no_z_bc.json \
  --pilot-directory "$PILOT" \
  --training-artifact "$ARTIFACT" \
  --seed-31001-run "$RUNROOT/grasp-lift-no-z-seed-31001-20260807-a" \
  --seed-31002-run "$RUNROOT/grasp-lift-no-z-seed-31002-20260807-a" \
  --seed-31003-run "$RUNROOT/grasp-lift-no-z-seed-31003-20260807-a" \
  --output-directory "$SELECTOUT" \
  --device cuda:0 \
  >"$LOGROOT/selection.console.log" 2>&1 &
printf '%s\n' "$!" >"$LOGROOT/selection.pid"
```

The selection output is intentionally non-resumable. After the claim receipt
has been persisted, any interruption burns this model-selection attempt and
must not be bypassed by deleting the output or claim registry.

- if selection passes, it is a development/model-selection result only. A new
  locked-test cohort and its own pre-registered one-time protocol are still
  required before a generalization claim;
- if selection fails, do not create a locked test. The next authoring proposal
  may use train evidence and the now-consumed validation audit, but the locked
  test must remain absent and cannot be used for tuning.

During the formal training and selection cohort the repository is
fingerprint-bound and may not be edited. The remote main branch is eligible for
a fast-forward only after the runtime gate and retained artifacts have been
verified.

## Node1 validation snapshot — 2026-08-05

The first independent Reach loop is now implemented under `rexpolicy/stage0/`
and was exercised on node1 CUDA without images or GR00T:

- the 64-world environment gate returned contiguous finite CUDA state
  `[64, 79]` and action `[64, 19]` tensors;
- the standalone oracle agreed exactly with Newton for every world: 64/64
  success, zero failure, with at most `1.42e-7 m` object displacement;
- the environment-only run completed 448 world-steps at about 432
  world-steps/s;
- the integrated run retained 64 successful trajectories and produced
  322/45/38 grouped train/validation/test windows;
- FutureTrajectoryEncoder, selector, and Flow-DiT each completed 20 optimizer
  steps, followed by a verified multi-component checkpoint save/load;
- all 57 focused Stage 0 tests, Ruff, formatting, and diff checks passed on
  node1.

This is an engineering validation, not a convergence claim. The short run had
an active latent-dimension fraction of 1.0 but effective rank 1.09, below the
minimum gate of 2.0. Oracle-z changed generated actions and reduced offline
action MSE from 0.976 (no-z) to 0.961, but this diagnostic is explicitly not a
rollout-success comparison. The next scientific gate requires repeated
same-reset successes with deliberately different safe approach modes, longer
encoder training, and equal-budget no-z/oracle-z/selector-z Newton rollouts.

Stage 0 is a fast scientific validation platform, not a smaller claim of full
VLA capability. It asks whether parallel robot self-interaction can produce a
non-collapsed, multimodal space of successful futures and whether conditioning
an action generator on that space improves behavior. Newton supplies dynamics
and the authoritative success/failure oracle. Images, language, GR00T, the
GR00T processor, and the frozen VLM are outside this stage.

The existing `rexpolicy/manifold`, `rexpolicy/replay`, and
`rexpolicy/flywheel` v1/v2/full paths remain intact. Default-off successor,
occupancy, shadow scoring, temperature, and active-sampling code is retained as
future Stage 3 work; it is not activated or treated as Stage 0 evidence.

## Scientific hypotheses

Stage 0 must distinguish three claims:

1. successful future windows form repeatable high-dimensional structure rather
   than a collapsed embedding or simulator-layout artifact;
2. current state predicts a calibrated distribution over reachable success
   latents;
3. a state-only Flow-DiT uses the latent, rather than ignoring it, to generate
   distinct successful action modes from the same initial-state group.

The first end-to-end acceptance task is Reach. Push and Pick enter only after
their reset distributions and binary oracles have independent tests. The
existing `bottle_transfer` task is grasp→lift→transport→release→settle and must
not be relabeled as a simple Pick oracle. No Push oracle currently exists.

## Architecture and isolation boundary

```text
Newton batched simulator
  -> canonical state-only adapter
  -> raw trajectory corpus (success, failure, timeout)
  -> verified-success future windows
  -> future-trajectory encoder -> z_success
  -> state-conditioned selector -> p(z_success | state)
  -> state token + success token
  -> lightweight conditional Flow-DiT -> action chunk
  -> Newton rollout -> authoritative oracle -> corpus expansion
```

Stage 0 lives under `rexpolicy/stage0/`. Its model path may import PyTorch but
must not import `gr00t`, `transformers`, `diffusers`, or the GR00T policy
wrapper. A strict `system_path` config enum distinguishes `stage0_state` from
`full_groot`; Stage 0 entry points require the former and never silently fall
back to the latter. The lightweight policy consumes a generic condition
contract:

```text
tokens:         float [batch, token_count, model_dim]
attention_mask: bool  [batch, token_count]
```

The Stage 0 policy owns and trains both `state_projector` and
`latent_projector`. It must not reuse the v2 encoder `condition_head`: that
head is not part of the current manifold loss and therefore is not a valid
trained policy-conditioning boundary.

## Canonical state contract

The first state schema is an ordered, versioned, fingerprinted vector of about
79 dimensions:

| Group | Field | Width |
| --- | --- | ---: |
| robot | arm joint position / velocity | 7 + 7 |
| robot | hand joint position / velocity | 10 + 10 |
| robot | end-effector position + rotation-6D | 3 + 6 |
| object | object position + rotation-6D | 3 + 6 |
| object | object linear + angular velocity | 3 + 3 |
| environment | goal, EEF→object, object→goal | 3 + 3 + 3 |
| contact | per-finger contact counts | 5 |
| contact | hand-contact and grasp flags | 1 + 1 |
| task | phase one-hot | 5 |

Exact fields may change before schema v1 is sealed, but these invariants may
not:

- all poses, twists, goals, and relative vectors use the right robot base
  frame, not replicated-world coordinates;
- quaternion sign is canonicalized or converted to rotation-6D before packing;
- raw values remain SI-unit physical state; normalization statistics are a
  separate fingerprinted training artifact;
- field names, offsets, units, frames, dtype, and dimension are included in
  the schema hash;
- packing is finite, CUDA-resident, batch-major, and does not render cameras.

This frame rule is a scientific gate. Flattening world-frame object and goal
poses would let the embedding cluster by Newton world placement and create
false success modes.

## Stage 0 trajectory and window data

Stage 0 intentionally does not reuse `SuccessExperienceGraph`. That graph is a
metadata-only locator over successful v2/full Event Ledgers and forbids stored
state/action tensors. Its signals also do not contain the Stage 0 state
contract.

Each immutable Stage 0 trajectory shard stores materialized state-only data:

```text
trajectory identity and provenance
task/oracle/simulator/config/state/action schema hashes
generation, rank, world, reset seed
states       [T + 1, state_dim]
actions      [T, action_dim]
terminated / truncated
outcome      success | failure | timeout
failure reason
```

All outcomes are retained for corpus accounting, oracle diagnostics, and
coverage analysis. Only simulator-verified successful trajectories may produce
success-manifold targets. Failures are not artificial “different success
mode” labels and do not enter the success encoder, selector targets, or
conditional-policy target set by default.

For every successful trajectory, a separate locator schema yields transition-
aligned windows:

```text
current_state       s[t]
action_chunk        a[t:t+H] with action mask
future_actions      a[t:t+K]
future_states       s[t+1:t+K+1]
shared future mask; H <= K
trajectory_id, start offset, schema and corpus hashes
```

Each future action is paired with its post-action state. Tail padding, masks,
stride, horizon, and future length are fixed by one fingerprinted extraction
policy. Dataset split is by trajectory and reset-state group before window
extraction so neighboring windows cannot leak across train/validation/test.

## Models and objectives

The Stage 0 future encoder receives state/action transitions and learns with:

- masked future reconstruction;
- temporal-neighborhood multi-positive contrastive alignment;
- variance and covariance collapse prevention.

Different trajectory IDs are not automatically different-mode negatives. A
trajectory-neighborhood positive policy and held-out clustering metrics avoid
forcing equivalent successful trajectories apart. A DDP batch must provide at
least two compatible windows per positive group or fail before InfoNCE silently
degenerates to zero.

The selector predicts a diagonal Gaussian `p(z_success | current_state)` and
is evaluated with held-out NLL, cosine alignment, retrieval/mode hit rate, and
calibration. The encoder is frozen before selector and policy training.

The conditional Flow-DiT uses standard flow matching:

```text
x0 ~ Normal(0, I)
x1 = normalized executed action chunk
xt = (1 - t) * x0 + t * x1
target velocity = x1 - x0
```

It predicts velocity from noisy actions, time, and the state/success condition
tokens. Inference uses a fixed-step Euler sampler. Required controls are
`no-z`, oracle future-z, and selector-z. Latent zeroing/permutation tests and a
toy same-state two-mode overfit test must prove that the policy uses z.

## Latent analysis and evaluation

Clustering runs in the original latent space. PCA/UMAP is diagnostic only and
must never define a training label, reward, or cluster assignment. Reports
include:

- success rate and outcome counts;
- cluster count, assignment entropy, effective mode count, and stability;
- per-dimension variance, effective rank, covariance, and pairwise cosine;
- cluster trajectory locators and a mask-aware average trajectory;
- same-initial-state successful mode count under multiple z/noise seeds;
- selector NLL, cosine alignment, calibration, and mode retrieval;
- no-z versus oracle-z versus selector-z policy results under equal simulator
  budgets and identical reset groups.

Reward shaping remains out of scope. The Newton oracle is authoritative and
latent diversity is logging only.

## Implemented Stage 0 surface

```text
rexpolicy/stage0/
  __init__.py
  config.py
  types.py
  seed.py
  distributed.py
  checkpoint.py
  runtime.py
  envs/
    __init__.py
    state_schema.py
    state_only.py
    oracles.py
    reset_sampler.py
  data/
    __init__.py
    trajectory.py
    store.py
    windows.py
    splits.py
  models/
    __init__.py
    _validation.py
    conditioning.py
    future_encoder.py
    selector.py
    flow_dit.py
  trainers/
    __init__.py
    batch.py
    common.py
    losses.py
    manifold.py
    selector.py
    policy.py
  evaluation/
    __init__.py
    latent_analysis.py
    metrics.py
    evaluator.py

configs/stage0/
  base.json
  reach_smoke.json

tools/
  run_stage0.py
  run_stage0_env.py
  test_stage0_*.py
```

Push/Pick configs, task-specific oracles, richer analysis entry points, and a
multi-GPU launcher remain phase-gated additions rather than placeholders.

Files are added phase by phase; placeholder modules are not created in bulk.

## Existing files with bounded changes

- `README.md` and `roadmap/README.md`: research priority and links;
- `roadmap/success_manifold_v2.md`: cross-link only, preserving v2/full history;
- `rexpolicy/envs/groot_newton_env.py`: at most one additive public GPU state
  hook for base-frame object twist/components; existing observations, reward,
  replay, and task behavior remain schema compatible;
- no Stage 0 change to `rexpolicy/replay/success_graph.py`,
  `rexpolicy/flywheel/groot_policy.py`, `rexpolicy/flywheel/trainer.py`,
  `rexpolicy/flywheel/checkpoint.py`, or `tools/run_flywheel_ddp.py`.

## Implementation phases and gates

### Phase A — state-only Newton interface

Add strict config/state schemas, canonical base-frame packing, deterministic
per-world reset recipes, and a camera-free single-control-step adapter. Do not
use the existing chunk wrapper for data capture after a world terminates. Gate
on GR00T-free import, shape/dtype/device/finite checks, frame invariance across
replicated worlds, masked partial reset, same-seed replay, different-seed
coverage, and a Reach GPU oracle smoke. The minimal demo is 64 worlds returning
state `[64, D]`, action `[64, 19]`, and oracle outcome tensors.

### Phase B — trajectory corpus and success windows

First add immutable success/failure/timeout tensor shards with descriptors and
checksums. Then add successful-window extraction, masks, grouped splits, and a
deterministic sampler. Gate on shard tamper rejection, no cross-trajectory
windows, exact tail semantics, split non-leakage, and resume-stable order.

### Phase C — future encoder

Add the independent encoder, reconstruction/contrastive/collapse objectives,
DDP trainer, and atomic checkpoint. Gate on nonzero gradients for every
trainable branch, two-rank synchronization, uninterrupted-versus-resume
equivalence, and held-out non-collapse metrics. Do not proceed without enough
verified Reach successes.

### Phase D — latent diagnostics

Add latent artifacts, high-dimensional clustering, collapse statistics,
cluster-average trajectory views, and deterministic PCA/optional UMAP. Gate on
permutation-stable results, diagnostic-only authority, trajectory-grouped
evaluation, and fixed seeds.

### Phase E — mode selector

Freeze the accepted encoder and train `state -> p(z_success)`. Gate on held-out
NLL/alignment/calibration, initial-state-group retrieval, exact checkpoint
resume, and a baseline comparison against a global latent distribution.

### Phase F — lightweight state-only Flow-DiT

Add generic condition tokens, trained state/latent projections, flow matching,
and Euler sampling without GR00T dependencies. Gate on toy two-mode overfit,
latent-use interventions, deterministic sampling, DDP synchronization, and
no-z/oracle-z/selector-z controls.

### Phase G — integrated flywheel

Close Newton rollout→oracle→corpus→encoder/selector/policy updates with a
multi-component atomic checkpoint. Bind model/optimizer/scheduler state,
phase/step/cursor, all RNG states, config/schema/normalization/dataset/split
hashes, and per-rank state. Gate on crash recovery, exact staged continuation,
fixed-budget held-out evaluation, and the runnable Reach demo before enabling
Push or Pick experiments.

Every phase is a sequence of small commits. Each commit receives focused unit
tests on node1 and is pushed to the remote work branch. The remote main branch
is updated only by a verified fast-forward after the phase gate passes.
