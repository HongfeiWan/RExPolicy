"""In-memory training samples and replayable episode metadata."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

DIRECT_SUCCESS_ROLE = "direct_branch"
SUCCESS_PATH_ROLE = "selected_success_path"
SUCCESS_ROLES = frozenset((DIRECT_SUCCESS_ROLE, SUCCESS_PATH_ROLE))


@dataclass
class BranchOutcomeAccumulator:
    """Sticky oracle outcomes observed over an executed candidate prefix."""

    failure: bool = False
    contact_violation: bool = False
    displacement_violation: bool = False

    def observe(
        self,
        *,
        executed: bool,
        failure: bool,
        contact_violation: bool,
        displacement_violation: bool,
    ) -> None:
        """Accumulate only frames that belong to this candidate prefix."""
        if not executed:
            return
        self.failure = self.failure or bool(failure)
        self.contact_violation = (
            self.contact_violation or bool(contact_violation)
        )
        self.displacement_violation = (
            self.displacement_violation or bool(displacement_violation)
        )

    @property
    def safety_violation(self) -> bool:
        """Whether any trusted safety predicate fired during the prefix."""
        return self.contact_violation or self.displacement_violation


@dataclass
class TrainingSample:
    """One executed action window conditioned on frozen VLM features."""

    backbone_features: Any
    backbone_attention_mask: Any
    image_mask: Any | None
    state: Any
    embodiment_id: int
    action: Any
    action_mask: Any
    valid_steps: int
    advantage: float = 0.0
    sample_weight: float = 1.0
    sample_id: str = ""
    generation: int = -1
    rank: int = -1
    episode: int = -1
    decision: int = -1
    world: int = -1
    task_id: str = "unknown"
    reward_profile_id: str = "unknown"
    source: str = "current"
    success_roles: tuple[str, ...] = ()

    @property
    def is_success(self) -> bool:
        """Whether this sample is a direct or successful-path experience."""
        return bool(self.success_roles)

    def add_success_role(self, role: str) -> None:
        """Attach one stable success provenance role."""
        if role not in SUCCESS_ROLES:
            raise ValueError(f"Unsupported success role {role!r}")
        if role not in self.success_roles:
            self.success_roles = (*self.success_roles, role)

    def set_provenance(
        self,
        *,
        generation: int,
        rank: int,
        episode: int,
        decision: int,
        world: int,
        task_id: str,
        reward_profile_id: str,
        source: str = "current",
    ) -> None:
        """Set the deterministic identity used by coverage and replay."""
        self.sample_id = (
            f"g{generation:06d}-r{rank:05d}-e{episode:05d}"
            f"-d{decision:05d}-w{world:05d}"
        )
        self.generation = int(generation)
        self.rank = int(rank)
        self.episode = int(episode)
        self.decision = int(decision)
        self.world = int(world)
        self.task_id = str(task_id)
        self.reward_profile_id = str(reward_profile_id)
        self.source = str(source)


@dataclass
class ChunkCandidate:
    """One same-state Newton branch and its comparative training metadata."""

    world: int
    score: float
    success: bool
    terminated: bool
    truncated: bool
    valid_steps: int
    rewards: list[float]
    action: dict[str, Any]
    sample: TrainingSample
    failure: bool = False
    safety_violation: bool = False
    failure_reasons: tuple[str, ...] = ()
    advantage: float = 0.0
    sample_weight: float = 0.0
    selected_for_training: bool = False
    chosen_for_continuation: bool = False

    def archive_record(self) -> dict[str, Any]:
        """Return the candidate without its transient VLM condition."""
        record = {
            "sample_id": self.sample.sample_id,
            "success_roles": list(self.sample.success_roles),
            "world": self.world,
            "score": self.score,
            "success": self.success,
            "failure": self.failure,
            "safety_violation": self.safety_violation,
            "failure_reasons": list(self.failure_reasons),
            "terminated": self.terminated,
            "truncated": self.truncated,
            "valid_steps": self.valid_steps,
            "rewards": self.rewards,
            "advantage": self.advantage,
            "sample_weight": self.sample_weight,
            "selected_for_training": self.selected_for_training,
            "chosen_for_continuation": self.chosen_for_continuation,
            "action": self.action,
        }
        validate_candidate_archive_record(record)
        return record


def validate_candidate_archive_record(record: dict[str, Any]) -> None:
    """Enforce the schema-v5 branch outcome and numeric invariants."""
    required = {
        "score",
        "success",
        "failure",
        "safety_violation",
        "failure_reasons",
        "terminated",
        "truncated",
        "valid_steps",
        "rewards",
        "selected_for_training",
        "chosen_for_continuation",
    }
    missing = sorted(required.difference(record))
    if missing:
        raise ValueError(
            "Candidate archive record is missing schema-v5 fields: "
            + ", ".join(missing)
        )
    boolean_fields = (
        "success",
        "failure",
        "safety_violation",
        "terminated",
        "truncated",
        "selected_for_training",
        "chosen_for_continuation",
    )
    if any(type(record[name]) is not bool for name in boolean_fields):
        raise ValueError("Candidate archive outcome flags must be booleans")
    success = bool(record["success"])
    failure = bool(record["failure"])
    unsafe = bool(record["safety_violation"])
    if success and failure:
        raise ValueError("Candidate cannot be both success and failure")
    if success and unsafe:
        raise ValueError("Candidate cannot be both success and unsafe")
    if unsafe and not failure:
        raise ValueError("Unsafe candidate must also be marked as failure")
    raw_reasons = record["failure_reasons"]
    if not isinstance(raw_reasons, (tuple, list)) or any(
        not isinstance(reason, str) or not reason
        for reason in raw_reasons
    ):
        raise ValueError("Candidate failure reasons must be non-empty strings")
    reasons = tuple(raw_reasons)
    if failure and not reasons:
        raise ValueError("Failed candidate requires a failure reason")
    if not failure and reasons:
        raise ValueError("Safe candidate cannot carry failure reasons")
    if (failure or unsafe) and (
        bool(record["selected_for_training"])
        or bool(record["chosen_for_continuation"])
    ):
        raise ValueError("Failed or unsafe candidate cannot be selected")
    if type(record["valid_steps"]) is not int:
        raise ValueError("Candidate valid_steps must be an integer")
    valid_steps = record["valid_steps"]
    rewards = tuple(float(reward) for reward in record["rewards"])
    if valid_steps < 1 or len(rewards) != valid_steps:
        raise ValueError("Candidate reward length must equal positive valid_steps")
    if not math.isfinite(float(record["score"])) or not all(
        math.isfinite(reward) for reward in rewards
    ):
        raise ValueError("Candidate archive contains non-finite scores")


@dataclass(frozen=True)
class ChunkSelection:
    """Result of one same-state comparative selection."""

    baseline: float
    mode: str
    selected: list[ChunkCandidate]
    continuation: ChunkCandidate | None


@dataclass
class EpisodeExperience:
    """One root trajectory assembled from success-constrained chunk branches."""

    generation: int
    sampling_policy_generation: int
    rank: int
    episode: int
    reset_seed: int
    instruction: str
    task_id: str
    reward_profile_id: str
    score: float
    success: bool
    initial_state: dict[str, Any]
    simulator_fingerprint: str = ""
    decisions: list[dict[str, Any]] = field(default_factory=list)
    selected_actions: list[dict[str, Any]] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    samples: list[TrainingSample] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        """Enforce the collect-then-update policy version convention."""
        if self.generation < 1:
            raise ValueError("Episode data generation must be positive")
        expected = self.generation - 1
        if self.sampling_policy_generation != expected:
            raise ValueError(
                "Episode sampling policy must be the preceding generation: "
                f"data={self.generation}, "
                f"sampling_policy={self.sampling_policy_generation}"
            )

    @property
    def success_samples(self) -> list[TrainingSample]:
        """Return unique direct and successful-path samples."""
        unique: dict[str, TrainingSample] = {}
        for sample in self.samples:
            if sample.is_success:
                unique[sample.sample_id] = sample
        return list(unique.values())

    def mark_success_path(self, path_samples: list[TrainingSample]) -> None:
        """Promote every chosen root chunk after the episode succeeds."""
        if not self.success:
            return
        for sample in path_samples:
            sample.add_success_role(SUCCESS_PATH_ROLE)

    def archive_record(self) -> dict[str, Any]:
        """Return replay metadata without images or frozen VLM features."""

        for decision in self.decisions:
            for candidate in decision.get("candidates", ()):
                validate_candidate_archive_record(candidate)

        def serializable(value: Any) -> Any:
            if hasattr(value, "tolist"):
                return value.tolist()
            if isinstance(value, dict):
                return {key: serializable(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [serializable(item) for item in value]
            return value

        return {
            "schema_version": 5,
            "generation": self.generation,
            "data_generation": self.generation,
            "sampling_policy_generation": self.sampling_policy_generation,
            "policy_version": (
                f"generation-{self.sampling_policy_generation:06d}"
            ),
            "rank": self.rank,
            "episode": self.episode,
            "task_id": self.task_id,
            "reward_profile_id": self.reward_profile_id,
            "simulator_fingerprint": self.simulator_fingerprint,
            "reset_recipe": {
                "environment": "GrootNewtonEnv",
                "seed": self.reset_seed,
                "initial_state": serializable(self.initial_state),
            },
            "instruction": self.instruction,
            "selected_path": {
                "score": self.score,
                "success": self.success,
                "rewards": self.rewards,
                "actions": serializable(self.selected_actions),
            },
            "decisions": serializable(self.decisions),
            "success_samples": [
                {
                    "sample_id": sample.sample_id,
                    "generation": sample.generation,
                    "rank": sample.rank,
                    "episode": sample.episode,
                    "decision": sample.decision,
                    "world": sample.world,
                    "roles": list(sample.success_roles),
                    "weight": sample.sample_weight,
                }
                for sample in self.success_samples
            ],
        }


def select_advantage_chunks(
    candidates: list[ChunkCandidate],
    *,
    fraction: float,
    temperature: float,
) -> ChunkSelection:
    """Retain every success, otherwise select positive within-state advantages."""
    if not candidates:
        raise ValueError("At least one chunk candidate is required")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("chunk selection fraction must be in (0, 1]")
    if temperature <= 0.0:
        raise ValueError("advantage temperature must be positive")

    for candidate in candidates:
        if candidate.success and candidate.failure:
            raise ValueError(
                f"Candidate world {candidate.world} cannot be both success and failure"
            )
        if candidate.success and candidate.safety_violation:
            raise ValueError(
                f"Candidate world {candidate.world} cannot be both success and unsafe"
            )

    eligible = [
        candidate
        for candidate in candidates
        if not candidate.failure and not candidate.safety_violation
    ]
    successful = [candidate for candidate in eligible if candidate.success]
    comparison_pool = successful or eligible
    if not comparison_pool:
        scores = sorted(candidate.score for candidate in candidates)
        midpoint = len(scores) // 2
        baseline = (
            scores[midpoint]
            if len(scores) % 2
            else 0.5 * (scores[midpoint - 1] + scores[midpoint])
        )
        for candidate in candidates:
            candidate.advantage = candidate.score - baseline
            candidate.sample.advantage = candidate.advantage
        return ChunkSelection(
            baseline=baseline,
            mode="no_safe_candidate",
            selected=[],
            continuation=None,
        )

    scores = sorted(candidate.score for candidate in comparison_pool)
    midpoint = len(scores) // 2
    if len(scores) % 2:
        baseline = scores[midpoint]
    else:
        baseline = 0.5 * (scores[midpoint - 1] + scores[midpoint])

    for candidate in candidates:
        candidate.advantage = candidate.score - baseline
        candidate.sample.advantage = candidate.advantage

    ordered = sorted(
        comparison_pool,
        key=lambda candidate: (candidate.score, -candidate.world),
        reverse=True,
    )
    continuation = ordered[0]
    continuation.chosen_for_continuation = True

    if successful:
        mode = "all_success"
        selected = ordered
    else:
        mode = "positive_advantage"
        limit = max(1, math.ceil(len(comparison_pool) * fraction))
        positive = [candidate for candidate in ordered if candidate.advantage > 0.0]
        selected = positive[:limit] or [continuation]
    raw_weights = [
        math.exp(min(math.log(20.0), max(0.0, candidate.advantage) / temperature))
        for candidate in selected
    ]
    mean_weight = sum(raw_weights) / len(raw_weights)
    for candidate, raw_weight in zip(selected, raw_weights):
        candidate.selected_for_training = True
        candidate.sample_weight = raw_weight / mean_weight
        candidate.sample.sample_weight = candidate.sample_weight
    return ChunkSelection(
        baseline=baseline,
        mode=mode,
        selected=selected,
        continuation=continuation,
    )


def collate_training_samples(
    samples: list[TrainingSample],
    *,
    device: Any,
    dtype: Any,
) -> tuple[Any, Any]:
    """Pad cached VLM sequences and form GR00T action-head inputs."""
    if not samples:
        raise ValueError("At least one training sample is required")

    import torch
    from transformers.feature_extraction_utils import BatchFeature

    batch_size = len(samples)
    sequence_length = max(int(sample.backbone_features.shape[0]) for sample in samples)
    feature_dim = int(samples[0].backbone_features.shape[1])
    backbone_features = torch.zeros(
        (batch_size, sequence_length, feature_dim),
        dtype=dtype,
        device=device,
    )
    backbone_attention_mask = torch.zeros(
        (batch_size, sequence_length),
        dtype=torch.bool,
        device=device,
    )
    has_image_mask = any(sample.image_mask is not None for sample in samples)
    image_mask = (
        torch.zeros((batch_size, sequence_length), dtype=torch.bool, device=device)
        if has_image_mask
        else None
    )

    for index, sample in enumerate(samples):
        length = int(sample.backbone_features.shape[0])
        backbone_features[index, :length].copy_(
            sample.backbone_features.to(device=device, dtype=dtype)
        )
        backbone_attention_mask[index, :length].copy_(
            sample.backbone_attention_mask.to(device=device, dtype=torch.bool)
        )
        if image_mask is not None and sample.image_mask is not None:
            image_mask[index, :length].copy_(
                sample.image_mask.to(device=device, dtype=torch.bool)
            )

    backbone_data = {
        "backbone_features": backbone_features,
        "backbone_attention_mask": backbone_attention_mask,
    }
    if image_mask is not None:
        backbone_data["image_mask"] = image_mask

    action_input = BatchFeature(
        data={
            "state": torch.stack([sample.state for sample in samples]).to(
                device=device, dtype=dtype
            ),
            "embodiment_id": torch.tensor(
                [sample.embodiment_id for sample in samples],
                dtype=torch.long,
                device=device,
            ),
            "action": torch.stack([sample.action for sample in samples]).to(
                device=device, dtype=dtype
            ),
            "action_mask": torch.stack([sample.action_mask for sample in samples]).to(
                device=device, dtype=dtype
            ),
        }
    )
    return BatchFeature(data=backbone_data), action_input
