# Automatic Task Authoring

This document defines the implemented production boundary for automatic
TaskSpec authoring. The boundary ends at a statically validated,
`quarantined_static` candidate. Authoring never grants admission, activation,
reward authority, or permission to train.

## Trust boundary

One canonical production bundle fixes every input that may affect a proposal:

- the sealed curriculum snapshot and its 0.5-success learning frontier;
- the public brief, AuthoringIntent, and precommitted sealed-audit plan;
- the capability catalog, allowed ProcessSpecs, and optional parent task;
- task, property, process, authoring, and proposer-execution policies;
- the exact proposer command and Bubblewrap sandbox profile;
- the provider, model, public verification key, and Ed25519 receipt policy.

The proposer receives only the public brief. Its response is accepted only when
the signed receipt binds the request bytes, response bytes, provider request
ID, provider/model versions, wrapper command, sandbox session, and verifier.
HMAC exists only in test fixtures; production bundle loading requires Ed25519.

The Bubblewrap runner uses a fixed filesystem and network policy, bounded input
and output sizes, a deadline, a sanitized environment, and verified regular
executables. A failed or interrupted attempt remains in the durable single
chain and can be resumed. An attested provider attempt is not silently rerun.

## Production invocation

The bundle must be a regular file containing canonical ASCII JSON. Its expected
SHA-256 is supplied out of band:

```bash
python tools/run_task_authoring.py \
  --bundle /absolute/path/authoring-bundle.json \
  --expected-bundle-sha256 <64-lowercase-hex> \
  --run-dir /absolute/path/to/authoring-run
```

The command prints one canonical result record. The run directory is the
durable source of truth for proposal attempts, validation/repair attempts, and
the final quarantine artifact. Reusing the same bundle and run directory
resumes the chain; changing a pinned input is rejected as drift.

Credentials belong behind the pinned provider wrapper. Do not place them in the
bundle, command-line arguments, candidate output, or repository.

## Candidate lifecycle

Lifecycle transitions are explicit and append exactly one immutable event:

```text
QUARANTINED_STATIC
        │ independent runtime + sealed-audit certifications
        ▼
ADMISSION_READY
        │ lifecycle-authority signature
        ▼
ADMITTED_DORMANT
        │ separate generation-boundary signature
        ▼
ACTIVE
```

Every event binds the exact candidate, intent, audit plan, runtime policy and
evidence, sealed suite, capability policy/report, and episode results. The
lifecycle store uses an immutable manifest, a contiguous hash-chained journal,
an expected-head CAS, and an independently located monotonic anchor. The
lifecycle and generation-boundary signer sets must be disjoint.

Deployment must put the anchor under a different administrative persistence
domain. A second file in the lifecycle directory is not an independent anchor.

## Relation to VLA-RL

The design adopts three useful ideas from
[VLA-RL](https://arxiv.org/abs/2505.18719) without turning an LLM or learned
reward model into an oracle:

- curriculum mass peaks at the learning frontier near 50% measured success;
- immutable raw transition facts support autonomous process-label and reward
  views, so labels can be rebuilt instead of becoming source-of-truth data;
- ProcessSpecs can describe milestone-oriented progress signals for later
  reward-model training and diagnostics.

Process-reward artifacts are currently `shadow_only` with
`oracle_authority=none`. They cannot redefine success or safety, admit a task,
activate a candidate, select a model for promotion, or mutate the Event Ledger.
Activation requires deterministic simulator certification and sealed K=1 audit
evidence even if a future process reward model is added.

## Operational acceptance checklist

A real authored task is not active until all of the following are archived and
verified:

1. canonical production bundle and Ed25519 provider receipt;
2. statically rebuilt candidate with no hidden simulator coordinates or audit
   leakage in train/promotion instructions;
3. dynamic runtime certification against the exact candidate and policy;
4. precommitted sealed-audit suite, paired episode results across independent
   training seeds, and an accepted capability-claim report;
5. signed `ADMITTED_DORMANT` event and independently anchored head;
6. clean generation boundary, separate activation signature, and exact active
   registry/runtime integration;
7. trainer checkpoint coverage for any new curriculum or quality-balanced
   replay state.

The repository implements the verification and persistence primitives for this
sequence. Running a real approved provider, producing hardware evidence,
deploying independent signers/anchor storage, and consuming an active authored
task remain operational gates.
