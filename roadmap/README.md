# RExPolicy Roadmap

This file is the single source of truth for delivery order, activation status,
and release gates. Documents elsewhere in `roadmap/` provide contracts,
operations, historical evidence, or the intended end state; they do not define
independent product routes.

## Canonical status vocabulary

- **Implemented** means the repository contains the code and focused contract
  tests.
- **Runtime-validated** means the phase-specific GPU, resume, determinism, and
  held-out gates have passed with archived evidence on the canonical artifacts.
- **Active** means the capability is enabled in the training path at a clean
  generation boundary.
- **Complete** means all exit gates for that phase have passed. Code being
  implemented ahead of its phase does not make that phase active or complete.

Phases activate in the order below. Later-phase code may be developed and
tested in isolation, but it must remain default-off and cannot change the active
learner until all preceding activation gates pass.

```text
Phase 0 baseline
    -> Phase 1 minimum Reach flywheel
    -> Phase 2 Success Manifold conditioning
    -> Phase 3 visual-language grounding
    -> Phase 4 task, reward, and replay expansion
    -> Phase 5 autonomous task authoring
    -> final multi-task system
```

## Canonical baseline and operating configuration

- Base policy: `checkpoints/groot/checkpoint-200000/`.
- Frozen VLM: `checkpoints/nvidia/Cosmos-Reason2-2B/`.
- Active task: `reach_green_cap/v1`.
- Current learner: the unconditioned v1 Flow-DiT flywheel.
- Default local/single-rank functional configuration: `K=2`, `H=2`.
- Phase 1 hardware acceptance: the node3 two-rank `K=2`, `H=1` smoke,
  fresh-process resume, and staged `K=4`, `H=2`, 10→20 generation soak defined in
  [bootstrap_ddp.md](bootstrap_ddp.md).

Any run that uses another base checkpoint is an explicitly named experiment. It
cannot provide canonical release or before/after evidence without being rerun
against `checkpoint-200000`.

The node1 `K=256` measurements are capacity evidence for a 72 GB single-GPU
configuration. `K=256` is not the repository default, does not replace node3
acceptance, and cannot become active until its own H=2 release ladder passes.
See [node1_gpu_capacity_20260804.md](node1_gpu_capacity_20260804.md).

## Phase 0 - migrated baseline

Status: **complete**.

- Newton batched simulator, GPU observations, action chunks, and deterministic
  reset.
- Frozen-capable GR00T VLM plus the original Flow-DiT inference path.
- Validated 19D EEF + Linker L10 action representation.
- Required compact assets and focused runtime tests.

Migration history is recorded in
[migration_inventory.md](migration_inventory.md).

## Phase 1 - minimum Reach flywheel

Status: **implemented and active, but not complete; node3 two-GPU,
fresh-process resume, and staged soak acceptance remain open**.

The fixed `reach_green_cap/v1` task tests whether the existing Flow-DiT can
improve using its own Newton-scored chunks. The implementation provides:

- same-state K-way action sampling, within-state advantages, and success-first
  continuation;
- all-success retention and successful-path back-indexing;
- FP32 DiT/AdamW updates with BF16 autocast and DDP-global weight
  normalization;
- guaranteed once-per-generation coverage of every newly selected sample;
- metadata-only Success Archive shards and deterministic historical
  round-robin replay;
- atomic full-state checkpoint/resume, run logs, heartbeat, GPU monitoring, and
  stop-at-generation-boundary control;
- sealed K=1 held-out evaluation and non-regression gating.

No PPO, GAE, critic, full-action teacher, compact replacement policy, or new
human pretraining dataset is required. The normative learning contract is
[minimum_flywheel.md](minimum_flywheel.md); the only Phase 1 acceptance protocol
is [bootstrap_ddp.md](bootstrap_ddp.md). Read-only base-versus-generation video
evidence follows [rollout_recorder.md](rollout_recorder.md).

Phase 1 exits only after the canonical node3 smoke, resume check, and staged
soak pass with `checkpoint-200000` and archived manifests. A successful node1
capacity probe or unit-test run does not close this gate.

## Phase 2 - Success Manifold conditioning

Status: **SM-1 through SM-5 code is implemented and default-off; the phase is
not runtime-validated or active**.

`RExPolicy v2` is the architecture name for this phase, not a second roadmap.
It adds an optional successful-future conditioning path to the Phase 1 learner:

- **SM-1:** metadata-only `SuccessExperienceGraph` construction and fixed-offset
  future-window materialization;
- **SM-2:** future encoder and state-to-success-mode selector training code;
- **SM-3:** diagnostic-only latent projection;
- **SM-4:** one-token Flow-DiT conditioning with mixed v1/v2 batch support;
- **SM-5:** simulator-verified bounded latent-memory expansion.

The active v1 path remains unchanged when the feature flags are absent. Phase 2
cannot activate until Phase 1 is complete and the following gates pass:

1. build and audit a real successful-window corpus from canonical Phase 1 data;
2. train and evaluate the encoder and selector on held-out real data;
3. export and checksum-pin a deployment bundle;
4. pass conditioning shadow mode, single-GPU smoke, resume equivalence, node3
   two-GPU smoke, and the staged non-regression soak;
5. enable bounded latent memory only after selector-only metrics are stable.

Diversity/progress reward shaping, successor representation, milestone windows,
automatic bundle promotion, and corpus-scale acceptance evidence remain
deferred. The detailed contract is
[success_manifold_v2.md](success_manifold_v2.md).

### State-only Stage 0 research workstream

The repository also retains a state-only Newton validation harness as isolated
Phase 2 pre-activation research. It does not alter the active learner or the
ordered phase sequence. It removes images, language, GR00T, and the frozen VLM
to test success-oracle data generation and lightweight conditional-policy
hypotheses.

Its reward-free Grasp-Lift authoring corpus and train-only artifact are
accepted. Three 10,000-step no-z training seeds completed, and an independent
write-once validation cohort produced a formal model-selection pass across the
fixed 60-model inventory. The registered rule selected the three-seed step-3,500
family after a seven-checkpoint eligible run from step 2,000 through 5,000;
the selected family achieved 72/72 rollout successes with zero safety,
integrity, or oracle-disagreement events. This remains supporting state-only
research evidence: it cannot close or activate Phase 2, prove visual grounding,
or substitute for a new locked-test protocol. See
[stage0_success_manifold_validation.md](stage0_success_manifold_validation.md).

## Phase 3 - visual-language grounding

Status: **not implemented**.

- Add two visually distinguishable targets in the same scene.
- Make the instruction, rather than simulator metadata exposed to the policy,
  select the target.
- Add equivalent paraphrases and counterfactual target instructions.
- Test whether the frozen VLM preserves identity and pose before introducing an
  adapter or limited unfreezing.

Phase 3 exits only after multi-target K=1 held-out grounding passes without
regressing the completed Phase 1 and Phase 2 gates.

## Phase 4 - task, reward, and replay expansion

Status: **supporting ledger and offline replay-control code is implemented;
active multi-task training is not implemented**.

Implemented ahead of activation:

- reward-free Event Ledgers and rebuildable reward/process views;
- exact success behavior descriptors and immutable quality-diversity indexes;
- deterministic Success Archive planning by behavior cell, reward profile, and
  initial-state group.

Still required:

- extend the curriculum from Reach to pre-grasp, grasp, lift, transport, and
  place;
- add deliberately diverse reward profiles for the same canonical success
  predicate;
- persist quality-balanced replay state in trainer checkpoints and make the
  online trainer consume it;
- add physics, camera, object, and scene randomization only after fixed-task
  replay remains deterministic.

Until Phase 4 activates, Phase 1 historical replay remains round-robin and no
offline QD plan may silently affect training.

## Phase 5 - autonomous task authoring

Status: **authoring-to-quarantine and lifecycle primitives are implemented;
provider-authored tasks are not active or consumed by training**.

Implemented ahead of activation:

- pinned, sandboxed, provider-attested proposal execution;
- strict compilation and repair into `quarantined_static` candidates;
- separate collection, promotion, and sealed-audit instruction commitments;
- signed admission and generation-boundary activation events with CAS heads and
  an independently located monotonic anchor.

Still required:

- archive a real approved provider job rather than a local fixture;
- produce independent dynamic-runtime, capability, episode-result, and sealed
  audit certifications;
- deploy independently administered lifecycle and generation-boundary signing
  authorities;
- integrate the active task registry and authored runtime with the Phase 4
  multi-task trainer;
- exercise a real `ADMITTED_DORMANT` to `ACTIVE` transition at a clean
  generation boundary.

The exact trust boundary is in
[automatic_task_authoring.md](automatic_task_authoring.md).

`TaskSpec v2` is a task-contract schema version. It does not name another
RExPolicy delivery route.

The v2/full implementation remains governed by the canonical phases above.
No isolated Stage 0 module may silently import or mutate the active runtime,
replay graph, or checkpoint state.

## Final system

The intended end state combines the completed phases above: unified success and
safety semantics, intentionally non-uniform rewards, metadata-only replay,
quality-balanced sampling, and held-out model promotion. It remains a target,
not an independently executable roadmap. See
[final_flywheel.md](final_flywheel.md).

## Document authority

| Document | Role |
| --- | --- |
| `README.md` | Short repository overview; must mirror this roadmap's active status. |
| `minimum_flywheel.md` | Normative Phase 1 learning contract. |
| `bootstrap_ddp.md` | Normative Phase 1 hardware acceptance protocol. |
| `success_manifold_v2.md` | Normative Phase 2 design and activation gates. |
| `stage0_success_manifold_validation.md` | Supporting Phase 2 research evidence and node1 execution protocol. |
| `automatic_task_authoring.md` | Normative Phase 5 trust boundary. |
| `node1_gpu_capacity_20260804.md` | Dated capacity evidence; not a release route or default configuration. |
| `rollout_recorder.md` | Read-only evidence procedure. |
| `migration_inventory.md` | Historical migration record. |
| `final_flywheel.md` | Aspirational end-state design. |

When a phase, default, or release gate changes, update this file first and then
update the affected supporting document and top-level README in the same change.
