# RExPolicy Roadmap

The roadmap separates the runnable minimum flywheel from the intended
multi-task, multi-reward system. The repository now contains the Phase 1 Reach
flywheel plus fail-closed control and data planes for TaskSpec v2, reward-free
ledgers and rebuildable views, sandboxed provider-attested automatic authoring,
signed lifecycle transitions, and offline quality-diversity replay planning.
The active learner remains the fixed Reach path: authored candidates do not
enter the registry or training until independent runtime and sealed-audit
evidence, signed activation at a generation boundary, and runtime integration
gates pass.

An additional, default-off v2 track now implements the first Success Manifold
MVP: successful-future graph construction, representation learning, a
state-conditioned mode selector, diagnostic latent projection, Flow-DiT token
conditioning, and bounded online latent-memory expansion. It is intentionally
isolated from the v1 learner until a trained bundle is checksum-pinned at
launch. See [success_manifold_v2.md](success_manifold_v2.md).

## Current research sequence

The scientific validation sequence is now explicitly separated from the
production flywheel phases below:

1. **Stage 0 — State-only Success Manifold Validation (current priority).**
   Use Newton state, a simulator oracle, future-trajectory latents, a
   state-conditioned selector, and a lightweight conditional Flow-DiT. Do not
   load images, language, GR00T, Transformers, or the frozen VLM. First close
   Reach end to end; Push and Pick require their own verified oracles and reset
   diversity before they are acceptance tasks.
2. **Stage 1 — GR00T visual embedding.** Replace the Stage 0 state condition
   adapter with frozen visual embeddings while keeping the validated manifold,
   data, evaluation, and conditional-policy contracts.
3. **Stage 2 — Full GR00T VLA + Success Manifold Guided Flow-DiT.** Add visual
   and language grounding only after Stage 0 and Stage 1 pass held-out gates.
4. **Stage 3 — Successor Representation + Active Exploration.** Activate
   successor calibration, latent occupancy, active sampling, and curriculum
   only after the base representation is shown to be non-collapsed and useful.

The recently added successor, occupancy, temperature, shadow-scoring, and
active-sampling components remain strict default-off Stage 3 candidates. They
are not evidence that the Stage 0 hypothesis has been validated. See
[stage0_success_manifold_validation.md](stage0_success_manifold_validation.md)
for the implementation order and acceptance gates.

Stage 0's Reach engineering path is now implemented and has completed a
64-world node1 CUDA smoke, success-only window construction, three component
training phases, and exact checkpoint recovery. Its first short-run latent
effective rank was 1.09, so the representation-quality gate remains open; the
next work is same-reset multi-mode data and fixed-budget conditional rollouts,
not Stage 1 activation.

## Phase 0 — migrated baseline

Status: **complete**.

- Newton batched simulator, GPU observations, action chunks, and deterministic
  reset.
- Frozen-capable GR00T VLM plus the original Flow-DiT inference path.
- Validated 19D EEF + Linker L10 action representation.
- Required compact assets and focused runtime tests.

## Phase 1 — minimum Reach flywheel

Status: **implemented; node3 two-GPU and soak acceptance remain runtime gates**.

The fixed `reach_green_cap/v1` task establishes whether the existing Flow-DiT
can improve using its own Newton-scored chunks. Its instruction asks the open
right hand to approach a safe pre-grasp point beside the green bottle cap
without touching or moving the bottle. Only EEF XYZ is effective; orientation,
hand, and arm targets remain fixed.

The implementation provides:

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

No PPO, GAE, critic, full-action teacher, compact replacement policy, or human
pretraining dataset is required. See [minimum_flywheel.md](minimum_flywheel.md)
for the learning contract and [bootstrap_ddp.md](bootstrap_ddp.md) for the
node3 validation protocol.

## Phase 2 — visual-language grounding

Status: **not implemented; multi-target frozen-VLM grounding remains a
prerequisite gate**.

- Add two visually distinguishable targets in the same scene.
- Make the instruction, not simulator metadata exposed to the policy, select
  the target.
- Add equivalent paraphrases and counterfactual target instructions.
- Test whether the frozen VLM preserves identity and pose before introducing an
  adapter or limited unfreezing.

## Phase 3 — task and behavior expansion

Status: **partial; ledger and offline replay-control infrastructure are
implemented, but active multi-task training remains pending**.

- Extend the curriculum from Reach to pre-grasp, grasp, lift, transport, and
  place.
- Add a deliberately diverse set of reward profiles for the same canonical
  success predicate.
- Implemented: compile reward-free Event Ledgers into exact success behavior
  descriptors, persist immutable quality-diversity indexes, explicitly rebase
  their cursors, and plan deterministic Success Archive replay by behavior
  cell, reward profile, and initial-state group without making valid successes
  ineligible.
- Pending: persist that quality-balanced state in trainer checkpoints and make
  the online trainer consume it. The active Phase 1 path remains round-robin.
- Add physics, camera, object, and scene randomization only after fixed-task
  replay remains deterministic.

## Phase 4 — autonomous task authoring

Status: **partial; production-safe authoring to static quarantine and signed
lifecycle primitives are implemented, but no authored candidate is active or
consumed by training**.

Implemented:

- bind a public brief, curriculum snapshot, pre-authoring audit plan, allowed
  capabilities/processes, parent contracts, provider/model policy, and all
  compiler policies before a proposal is requested;
- execute a pinned proposer in Bubblewrap, require an Ed25519 provider receipt,
  persist one resumable job chain, and compile/repair only strict canonical
  responses into `quarantined_static` candidates;
- keep collection/training, promotion, and sealed-audit instruction
  commitments separate;
- persist signed admission and activation lifecycle events with CAS heads and
  an independently located monotonic anchor;
- rebuild reward, process-label, and quality-diversity views from immutable raw
  ledger facts without deleting the Success Archive.

Pending operational gates:

- run and archive a real approved provider job rather than the local signed
  integration fixture;
- generate independent dynamic-runtime, capability-claim, episode-result, and
  sealed-audit certifications for a candidate;
- deploy independently administered lifecycle and generation-boundary signing
  authorities, then exercise a real `ADMITTED_DORMANT` to `ACTIVE` transition;
- integrate the active registry, authored task runtime, reward populations,
  quality-balanced checkpoint state, and online reward-gap discovery with the
  trainer.

See [automatic_task_authoring.md](automatic_task_authoring.md) for the exact
safety boundary and execution contract.

## RExPolicy v2 — Success Manifold Guided Flow-DiT

Status: **MVP phases 1–5 implemented behind strict opt-in flags; corpus-scale
training and node3 acceptance remain runtime gates**.

- compile multiple verified terminal paths into metadata-only
  SuccessExperience graphs and fixed-offset future windows;
- train a Transformer future encoder with masked reconstruction,
  multi-positive InfoNCE, variance, and covariance objectives;
- train a Gaussian state-to-success-mode selector and export a checksum-pinned
  frozen runtime bundle;
- inspect the learned geometry with deterministic PCA or optional UMAP/t-SNE,
  explicitly marked diagnostic-only;
- append one projected success token to Flow-DiT context and preserve mixed
  historical v1 samples;
- sample selector and bounded-memory modes online, admit only simulator-verified
  successful modes, and checkpoint per-rank memory state.

Diversity/progress reward shaping and successor representation are deferred
until the representation and selector pass held-out real-corpus gates. This is
an explicit staging boundary, not an implicit activation of new rewards.

The v2/full implementation is preserved while Stage 0 becomes the active
validation path. No Stage 0 module may silently import or mutate the v2/full
runtime, replay graph, or checkpoint state.

## Final system

The stable end-state decisions—unified success semantics, intentionally
non-uniform rewards, replay instead of stored video, and held-out model
promotion—are preserved in [final_flywheel.md](final_flywheel.md).
