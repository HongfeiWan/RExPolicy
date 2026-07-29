"""Frozen GR00T VLM inference and trainable Flow-DiT utilities."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .experience import TrainingSample

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")


@dataclass(frozen=True)
class RawPolicyObservation:
    """One world observation in the checkpoint's raw modality layout."""

    images: dict[str, np.ndarray]
    state: dict[str, np.ndarray]
    instruction: str


@dataclass(frozen=True)
class CachedCondition:
    """Frozen VLM output and normalized state for one action window."""

    backbone_features: Any
    backbone_attention_mask: Any
    image_mask: Any | None
    state: Any
    embodiment_id: int


@dataclass(frozen=True)
class PolicyBatch:
    """Decoded candidate chunks and their reusable training conditions."""

    decoded_action: dict[str, np.ndarray]
    conditions: list[CachedCondition]


def _processor_path(model_path: Path) -> Path:
    if (model_path / "processor_config.json").is_file():
        return model_path
    if (model_path / "processor" / "processor_config.json").is_file():
        return model_path / "processor"
    raise FileNotFoundError(
        f"GR00T processor_config.json is missing under {model_path}"
    )


class GrootFlowDitPolicy:
    """Own the frozen VLM, processor, and DiT updated by DDP."""

    DEFAULT_TRAINED_ACTION_KEYS = ("eef_9d", "hand_joint_target")

    def __init__(
        self,
        *,
        isaac_groot_root: Path,
        checkpoint: Path,
        vlm_model: Path,
        device: Any,
        dit_overlay: Path | None = None,
    ) -> None:
        root = isaac_groot_root.expanduser().resolve()
        checkpoint = checkpoint.expanduser().resolve()
        vlm_model = vlm_model.expanduser().resolve()
        self._validate_layout(root, checkpoint, vlm_model)
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        import gr00t.model  # noqa: F401
        import torch
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.data.types import MessageType, VLAStepData
        from transformers import AutoConfig, AutoModel, AutoProcessor

        self.torch = torch
        self.device = device
        self.compute_dtype = torch.bfloat16
        # Retain the old attribute for callers while making its meaning explicit:
        # model compute remains BF16 even though DiT master parameters are FP32.
        self.dtype = self.compute_dtype
        self.embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
        self.message_type = MessageType
        self.vla_step_data = VLAStepData

        config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
        config.model_name = str(vlm_model)
        self.model = AutoModel.from_pretrained(
            checkpoint,
            config=config,
            dtype=self.compute_dtype,
            local_files_only=True,
            transformers_loading_kwargs={"local_files_only": True},
        )
        self.model.requires_grad_(False)
        self.model.action_head.set_trainable_parameters(False, True, False)
        self.model.action_head.state_dropout_prob = 0.0
        self.model.to(device=device, dtype=self.compute_dtype)
        self.dit.to(device=device, dtype=torch.float32)
        self.model.eval()
        self.assert_precision_contract()

        self.processor = AutoProcessor.from_pretrained(
            _processor_path(checkpoint),
            model_name=str(vlm_model),
            transformers_loading_kwargs={"local_files_only": True},
            local_files_only=True,
        )
        self.processor.eval()
        self.collate_fn = self.processor.collator
        modality_configs = self.processor.get_modality_configs()[
            self.embodiment_tag.value
        ]
        self.action_keys = tuple(modality_configs["action"].modality_keys)
        self.processor_action_horizon = len(modality_configs["action"].delta_indices)
        self.state_keys = tuple(
            self.processor.get_modality_configs()[self.embodiment_tag.value][
                "state"
            ].modality_keys
        )
        self._backbone_offloaded = False

        if dit_overlay is not None:
            self.load_dit_overlay(dit_overlay)

    @staticmethod
    def _validate_layout(root: Path, checkpoint: Path, vlm_model: Path) -> None:
        required = (
            root / "gr00t" / "__init__.py",
            checkpoint / "config.json",
            checkpoint / "model.safetensors.index.json",
            checkpoint / "processor_config.json",
            vlm_model / "config.json",
            vlm_model / "model.safetensors",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Missing GR00T runtime artifacts:\n" + "\n".join(missing)
            )

    @property
    def action_head(self) -> Any:
        """Return the GR00T action head wrapped by DDP for updates."""
        return self.model.action_head

    @property
    def dit(self) -> Any:
        """Return the trainable Flow-DiT module."""
        return self.model.action_head.model

    @property
    def backbone_offloaded(self) -> bool:
        """Whether the frozen VLM currently resides on CPU."""
        return self._backbone_offloaded

    def parameter_counts(self) -> dict[str, int]:
        """Report full-model and trainable DiT parameter counts."""
        return {
            "model": sum(parameter.numel() for parameter in self.model.parameters()),
            "dit": sum(parameter.numel() for parameter in self.dit.parameters()),
            "trainable": sum(
                parameter.numel()
                for parameter in self.model.parameters()
                if parameter.requires_grad
            ),
        }

    def assert_precision_contract(self) -> None:
        """Require FP32 CUDA master parameters only inside the Flow-DiT."""
        trainable_names = {
            name
            for name, parameter in self.action_head.named_parameters()
            if parameter.requires_grad
        }
        dit_names = {f"model.{name}" for name, _ in self.dit.named_parameters()}
        unexpected = sorted(trainable_names - dit_names)
        if unexpected:
            raise RuntimeError(
                "Only action_head.model may be trainable; unexpected parameters: "
                + ", ".join(unexpected[:8])
            )
        if not trainable_names:
            raise RuntimeError("Flow-DiT has no trainable parameters")
        violations = []
        for name, parameter in self.action_head.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.dtype != self.torch.float32:
                violations.append(f"{name}: dtype={parameter.dtype}")
            if parameter.device != self.device:
                violations.append(f"{name}: device={parameter.device}")
        if violations:
            raise RuntimeError(
                "Flow-DiT FP32 master-parameter contract failed: "
                + "; ".join(violations[:8])
            )

    def encode_conditions(
        self, observations: list[RawPolicyObservation]
    ) -> list[CachedCondition]:
        """Encode images, language, and state without sampling a DiT action."""
        if self._backbone_offloaded:
            raise RuntimeError("Move the frozen VLM back to CUDA before collection")
        if not observations:
            raise ValueError("At least one policy observation is required")

        processed = []
        for observation in observations:
            step = self.vla_step_data(
                images=observation.images,
                states=observation.state,
                actions={},
                text=observation.instruction,
                embodiment=self.embodiment_tag,
            )
            messages = [
                {
                    "type": self.message_type.EPISODE_STEP.value,
                    "content": step,
                }
            ]
            processed.append(self.processor(messages))

        collated = self.collate_fn(processed)["inputs"]
        with (
            self.torch.inference_mode(),
            self.torch.autocast(
                device_type="cuda",
                dtype=self.compute_dtype,
                enabled=self.device.type == "cuda",
            ),
        ):
            backbone_input, action_input = self.model.prepare_input(collated)
            backbone_output = self.model.backbone(backbone_input)

        conditions = []
        attention = backbone_output["backbone_attention_mask"]
        optional_image_mask = backbone_output.get("image_mask")
        for index in range(len(observations)):
            mask = attention[index].to(dtype=self.torch.bool)
            nonzero = self.torch.nonzero(mask, as_tuple=False)
            length = (
                int(nonzero[-1].item()) + 1 if nonzero.numel() else int(mask.shape[0])
            )
            image_mask = None
            if optional_image_mask is not None:
                image_mask = optional_image_mask[index, :length].detach().to("cpu")
            conditions.append(
                CachedCondition(
                    backbone_features=backbone_output["backbone_features"][
                        index, :length
                    ]
                    .detach()
                    .to("cpu"),
                    backbone_attention_mask=attention[index, :length]
                    .detach()
                    .to("cpu"),
                    image_mask=image_mask,
                    state=action_input["state"][index].detach().to("cpu"),
                    embodiment_id=int(action_input["embodiment_id"][index].item()),
                )
            )
        return conditions

    def _collate_conditions(
        self, conditions: list[CachedCondition]
    ) -> tuple[Any, Any]:
        """Move cached frozen features back to CUDA for action sampling."""
        from transformers.feature_extraction_utils import BatchFeature

        if not conditions:
            raise ValueError("At least one cached condition is required")
        batch_size = len(conditions)
        sequence_length = max(
            int(condition.backbone_features.shape[0]) for condition in conditions
        )
        feature_dim = int(conditions[0].backbone_features.shape[1])
        features = self.torch.zeros(
            (batch_size, sequence_length, feature_dim),
            device=self.device,
            dtype=self.compute_dtype,
        )
        attention = self.torch.zeros(
            (batch_size, sequence_length),
            device=self.device,
            dtype=self.torch.bool,
        )
        has_image_mask = any(condition.image_mask is not None for condition in conditions)
        image_mask = (
            self.torch.zeros_like(attention) if has_image_mask else None
        )
        for index, condition in enumerate(conditions):
            length = int(condition.backbone_features.shape[0])
            features[index, :length].copy_(
                condition.backbone_features.to(
                    device=self.device, dtype=self.compute_dtype
                )
            )
            attention[index, :length].copy_(
                condition.backbone_attention_mask.to(
                    device=self.device, dtype=self.torch.bool
                )
            )
            if image_mask is not None and condition.image_mask is not None:
                image_mask[index, :length].copy_(
                    condition.image_mask.to(
                        device=self.device, dtype=self.torch.bool
                    )
                )
        backbone_data = {
            "backbone_features": features,
            "backbone_attention_mask": attention,
        }
        if image_mask is not None:
            backbone_data["image_mask"] = image_mask
        action_input = BatchFeature(
            data={
                "state": self.torch.stack(
                    [condition.state for condition in conditions]
                ).to(device=self.device, dtype=self.compute_dtype),
                "embodiment_id": self.torch.tensor(
                    [condition.embodiment_id for condition in conditions],
                    dtype=self.torch.long,
                    device=self.device,
                ),
            }
        )
        return BatchFeature(data=backbone_data), action_input

    def sample_from_conditions(
        self,
        *,
        conditions: list[CachedCondition],
        observations: list[RawPolicyObservation],
    ) -> PolicyBatch:
        """Sample independent DiT noise from already encoded conditions."""
        if len(conditions) != len(observations):
            raise ValueError("Conditions and observations must have equal length")
        backbone_output, action_input = self._collate_conditions(conditions)
        with (
            self.torch.inference_mode(),
            self.torch.autocast(
                device_type="cuda",
                dtype=self.compute_dtype,
                enabled=self.device.type == "cuda",
            ),
        ):
            prediction = self.action_head.get_action(backbone_output, action_input)

        normalized_action = prediction["action_pred"].float().cpu().numpy()
        batched_states = {
            key: np.stack(
                [observation.state[key] for observation in observations], axis=0
            )
            for key in self.state_keys
        }
        decoded = self.processor.decode_action(
            normalized_action,
            self.embodiment_tag,
            batched_states,
        )
        return PolicyBatch(
            decoded_action={
                key: np.asarray(value, dtype=np.float32)
                for key, value in decoded.items()
            },
            conditions=conditions,
        )

    def sample(self, observations: list[RawPolicyObservation]) -> PolicyBatch:
        """Encode observations and sample independent DiT noise per world."""
        conditions = self.encode_conditions(observations)
        return self.sample_from_conditions(
            conditions=conditions,
            observations=observations,
        )

    def sample_same_state(
        self,
        observation: RawPolicyObservation,
        *,
        candidate_count: int,
    ) -> PolicyBatch:
        """Encode one shared state and sample independent DiT noise per world.

        The caller must first verify that every candidate world represents the
        same simulator state. The frozen VLM runs only for the canonical
        reference world; the condition and raw state are then expanded across
        the candidate batch before Flow-DiT samples independent noise.
        """
        if candidate_count < 1:
            raise ValueError("candidate_count must be positive")
        condition = self.encode_conditions([observation])[0]
        return self.sample_from_conditions(
            conditions=[condition] * candidate_count,
            observations=[observation] * candidate_count,
        )

    def make_training_sample(
        self,
        *,
        condition: CachedCondition,
        raw_state: dict[str, np.ndarray],
        executed_action: dict[str, np.ndarray],
        action_dimension_masks: dict[str, tuple[bool, ...]] | None = None,
        sample_metadata: dict[str, Any] | None = None,
    ) -> TrainingSample:
        """Normalize only an actually executed action prefix for Flow Matching."""
        valid_steps = min(int(value.shape[0]) for value in executed_action.values())
        if valid_steps < 1:
            raise ValueError("An executed action window must contain at least one step")

        padded_action = {}
        for key, value in executed_action.items():
            array = np.asarray(value, dtype=np.float32)
            if array.shape[0] < self.processor_action_horizon:
                padding = np.repeat(
                    array[-1:],
                    self.processor_action_horizon - array.shape[0],
                    axis=0,
                )
                array = np.concatenate((array, padding), axis=0)
            padded_action[key] = array[: self.processor_action_horizon]

        _, normalized = self.processor.state_action_processor.apply(
            state=raw_state,
            action=padded_action,
            embodiment_tag=self.embodiment_tag.value,
        )
        action_dim = int(self.model.config.max_action_dim)
        action_horizon = int(self.model.config.action_horizon)
        if valid_steps > action_horizon:
            raise ValueError(
                f"Executed prefix {valid_steps} exceeds model horizon {action_horizon}"
            )

        action = self.torch.zeros(
            (action_horizon, action_dim), dtype=self.torch.float32
        )
        action_mask = self.torch.zeros_like(action)
        offset = 0
        for key in self.action_keys:
            values = self.torch.from_numpy(
                np.asarray(normalized[key], dtype=np.float32)
            )
            width = int(values.shape[1])
            action[:valid_steps, offset : offset + width].copy_(values[:valid_steps])
            if action_dimension_masks is None:
                if key in self.DEFAULT_TRAINED_ACTION_KEYS:
                    action_mask[:valid_steps, offset : offset + width] = 1.0
            else:
                configured_mask = action_dimension_masks.get(key)
                if configured_mask is not None:
                    if len(configured_mask) != width:
                        raise ValueError(
                            f"Action mask for {key!r} has {len(configured_mask)} "
                            f"dimensions, expected {width}"
                        )
                    dimension_mask = self.torch.tensor(
                        configured_mask,
                        dtype=action_mask.dtype,
                    )
                    action_mask[
                        :valid_steps, offset : offset + width
                    ] = dimension_mask
            offset += width

        sample = TrainingSample(
            backbone_features=condition.backbone_features,
            backbone_attention_mask=condition.backbone_attention_mask,
            image_mask=condition.image_mask,
            state=condition.state,
            embodiment_id=condition.embodiment_id,
            action=action,
            action_mask=action_mask,
            valid_steps=valid_steps,
        )
        if sample_metadata:
            sample.set_provenance(**sample_metadata)
        return sample

    def offload_backbone(self) -> None:
        """Move the frozen VLM to CPU while cached features train DiT."""
        if self._backbone_offloaded:
            return
        self.model.backbone.to("cpu")
        self._backbone_offloaded = True
        self.torch.cuda.empty_cache()

    def restore_backbone(self) -> None:
        """Move the frozen VLM back to this rank's CUDA device."""
        if not self._backbone_offloaded:
            return
        self.model.backbone.to(device=self.device, dtype=self.compute_dtype)
        self.model.backbone.eval()
        self._backbone_offloaded = False

    def load_dit_overlay(self, directory: Path) -> None:
        """Load a previously saved DiT-only candidate checkpoint."""
        directory = directory.expanduser().resolve()
        candidates = (
            directory / "diffusion_pytorch_model.safetensors",
            directory / "model.safetensors",
        )
        weight_path = next((path for path in candidates if path.is_file()), None)
        if weight_path is None:
            raise FileNotFoundError(f"No DiT safetensors file found in {directory}")
        from safetensors.torch import load_file

        state = load_file(str(weight_path), device="cpu")
        missing, unexpected = self.dit.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"DiT overlay mismatch: missing={list(missing)} unexpected={list(unexpected)}"
            )

    def save_dit(self, directory: Path, *, metadata: dict[str, Any]) -> None:
        """Save only the synchronized DiT rather than another full VLM."""
        from safetensors.torch import save_model

        directory.mkdir(parents=True, exist_ok=True)
        save_model(
            self.dit,
            str(directory / "model.safetensors"),
            metadata={"format": "pt", "component": "gr00t_flow_dit"},
        )
        (directory / "flywheel_state.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
