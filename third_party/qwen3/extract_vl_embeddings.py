"""Minimal Qwen3-VL embedding extraction example for debugging new backbones."""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import List

import numpy as np
import torch
import tyro
from PIL import Image

from reward_model.config import get_reward_backbone_config
from reward_model.embedding_extractors import ExtractorInitParams, build_embedding_extractor

LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    image_dir: str
    prompt: str = "Put the items in the pot."
    batch_size: int = 32


def _load_frames(image_dir: Path) -> np.ndarray:
    frames: List[np.ndarray] = []
    for image_path in sorted(image_dir.glob("*.png")):
        image = Image.open(image_path).convert("RGB")
        frames.append(np.asarray(image))
    if not frames:
        raise ValueError(f"No PNG images found under {image_dir}.")
    return np.stack(frames, axis=0)


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO)
    image_dir = Path(args.image_dir)
    frames = _load_frames(image_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = get_reward_backbone_config("qwen3_vl")
    extractor = build_embedding_extractor(
        backbone,
        ExtractorInitParams(device=device),
    )

    embeddings = extractor.extract_visual_embeddings(frames, args.batch_size, args.prompt)
    LOGGER.info(
        "Extracted embeddings with shape %s from %d frames located at %s",
        embeddings.shape,
        frames.shape[0],
        image_dir,
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
