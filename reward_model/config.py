from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass(frozen=True)
class EncoderConfig:
    kind: str
    params: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class EmbeddingFieldConfig:
    key: str
    dim: Optional[int] = None


@dataclass(frozen=True)
class RewardBackboneConfig:
    name: str
    visual_embedding: EmbeddingFieldConfig
    language_embedding: Optional[EmbeddingFieldConfig] = None
    visual_config: EncoderConfig = field(default_factory=EncoderConfig)
    language_config: Optional[EncoderConfig] = None


DINOV2_MINILM_BACKBONE = RewardBackboneConfig(
    name="dinov2_minilm",
    visual_embedding=EmbeddingFieldConfig(key="dino_embeddings", dim=768),
    language_embedding=EmbeddingFieldConfig(key="minlm_task_embedding", dim=384),
    visual_config=EncoderConfig(
        kind="dinov2",
        params={
            "img_size": 518,
            "patch_size": 14,
            "init_values": 1.0,
            "ffn_layer": "mlp",
            "block_chunks": 0,
            "num_register_tokens": 4,
            "interpolate_antialias": True,
            "interpolate_offset": 0.0,
            "dino_ckpt_path": "/data/yuwenye/reward_modeling/dinov2_vitb14_reg4_pretrain.pth"
        },
    ),
    language_config=EncoderConfig(
        kind="minilm",
        params={
            "model_name": "sentence-transformers/all-MiniLM-L12-v2",
        },
    ),
)

QWEN3_VL_BACKBONE = RewardBackboneConfig(
    name="qwen3_vl",
    visual_embedding=EmbeddingFieldConfig(key="qwen3_vl_embeddings", dim=None),
    language_embedding=None,
    visual_config=EncoderConfig(
        kind="qwen3_vl",
        params={
            "model_name": "Qwen/Qwen3-VL-8B-Thinking",
            "attn_implementation": "flash_attention_2",
            "device_map": "auto",
            "image_patch_size": 16,
        },
    ),
)


PI0_INTERNAL_BACKBONE = RewardBackboneConfig(
    name="pi0_internal",
    visual_embedding=EmbeddingFieldConfig(key="pi0_internal_embeddings", dim=None),
    language_embedding=None,
    visual_config=EncoderConfig(
        kind="pi0_internal",
        params={
            "policy_config": "pi05_flexiv",
            "policy_dir": "/data/yuwenye/openpi/checkpoints/pi05_flexiv/kitchen_100/39999",
            "enable_multi_device": True,
        },
    ),
)


REWARD_BACKBONE_REGISTRY: Dict[str, RewardBackboneConfig] = {
    DINOV2_MINILM_BACKBONE.name: DINOV2_MINILM_BACKBONE,
    QWEN3_VL_BACKBONE.name: QWEN3_VL_BACKBONE,
    PI0_INTERNAL_BACKBONE.name: PI0_INTERNAL_BACKBONE,
}


def get_reward_backbone_config(name: str) -> RewardBackboneConfig:
    """Return the registered backbone configuration identified by ``name``."""
    try:
        return REWARD_BACKBONE_REGISTRY[name]
    except KeyError as exc:  # pragma: no cover - defensive programming
        available = ", ".join(REWARD_BACKBONE_REGISTRY)
        raise ValueError(f"Unknown backbone '{name}'. Available options: {available}.") from exc

