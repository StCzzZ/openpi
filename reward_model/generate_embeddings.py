from __future__ import annotations

import dataclasses
import logging
from typing import Optional

import h5py
import numpy as np
import torch
import tyro
from tqdm import tqdm

from reward_model.config import RewardBackboneConfig, get_reward_backbone_config
from reward_model.embedding_extractors import (
    ExtractorInitParams,
    build_embedding_extractor,
)

LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    data_path: str
    output_path: str
    backbone: str = "dinov2_minilm"
    batch_size: int = 32
    prompt: Optional[str] = None
    device: str = "cuda"


def _resolve_device(device_str: str) -> torch.device:
    device = torch.device(device_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return device


def _write_dataset(group: h5py.Group, name: str, data: np.ndarray) -> None:
    if name in group:
        del group[name]
    group.create_dataset(name, data=data)


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO)
    device = _resolve_device(args.device)
    backbone_config = get_reward_backbone_config(args.backbone)
    prompt = args.prompt

    extractor = build_embedding_extractor(
        backbone_config,
        ExtractorInitParams(device=device),
    )
    language_embedding = extractor.get_language_embedding(prompt)

    with h5py.File(args.data_path, "r") as dataset, h5py.File(args.output_path, "w") as output_file:
        episode_ids = list(dataset.keys())
        for key in tqdm(episode_ids, desc="Processing episodes"):
            episode = dataset[key]
            group = output_file.require_group(key)
            frames = episode["side_cam"]
            other_frames = episode["wrist_cam"]
            proprio_state = episode["tcp_pose"]
            visual_embeddings = extractor.extract_visual_embeddings(
                frames=frames,
                batch_size=args.batch_size,
                prompt=prompt,
                other_frames=other_frames,
                proprio_state=proprio_state,
            )
            # Prevent bfloat16 precision issue with hdf5 datasets
            visual_embeddings = visual_embeddings.astype(np.float32)
            _write_dataset(group, backbone_config.visual_embedding.key, visual_embeddings)

            if language_embedding is not None and backbone_config.language_embedding is not None:
                _write_dataset(group, backbone_config.language_embedding.key, language_embedding)

            num_steps = int(episode["action"].shape[0])
            progress = np.arange(1, num_steps + 1, dtype=np.float32) / float(num_steps)
            _write_dataset(group, "progress", progress)

    LOGGER.info("Saved embeddings for %d episodes to %s", len(episode_ids), args.output_path)


if __name__ == "__main__":
    main(tyro.cli(Args))
