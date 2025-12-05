import dataclasses
import json
import logging
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import h5py
import numpy as np
import tyro

from reward_model.config import RewardBackboneConfig, get_reward_backbone_config

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    data_path: str
    stage_annotation_path: str
    num_stages: int
    backbone: str = "dinov2_minilm"

def _load_stage_annotations(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
    if not isinstance(data, dict):
        raise ValueError("Stage annotation file must contain a JSON object keyed by episode id.")
    return data


def _sanitize_stage_boundaries(num_steps: int, raw_boundaries: Iterable[Any]) -> List[int]:
    sanitized: List[int] = []
    sorted_boundaries = sorted({int(boundary) for boundary in raw_boundaries})
    for boundary in sorted_boundaries:
        if 0 <= boundary < num_steps:
            if not sanitized or boundary > sanitized[-1]:
                sanitized.append(boundary)
    if not sanitized or sanitized[0] != 0:
        sanitized.insert(0, 0)
    return sanitized


def _build_stage_annotations(
    num_steps: int, raw_boundaries: Sequence[Any]
) -> Tuple[np.ndarray, np.ndarray]:
    stage_starts = _sanitize_stage_boundaries(num_steps, raw_boundaries)
    stage_ids = np.zeros(num_steps, dtype=np.int_)
    subtask_progress = np.zeros(num_steps, dtype=np.float32)

    boundaries = stage_starts + [num_steps]
    for stage_idx, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        end = min(end, num_steps)
        if start >= end:
            continue
        stage_length = end - start
        stage_ids[start:end] = stage_idx
        subtask_progress[start:end] = np.arange(
            1, stage_length + 1, dtype=np.float32
        ) / np.float32(stage_length)

    return stage_ids, subtask_progress


def _write_dataset(group: h5py.Group, name: str, data: np.ndarray) -> None:
    if name in group:
        del group[name]
    group.create_dataset(name, data=data, dtype=data.dtype)


def main(args: Args) -> None:
    stage_annotations = _load_stage_annotations(args.stage_annotation_path)
    backbone_config = get_reward_backbone_config(args.backbone)

    with h5py.File(args.data_path, "r+") as dataset:
        updated = 0
        total = len(dataset.keys())

        for key in dataset.keys():
            if key not in stage_annotations:
                LOGGER.warning("Missing stage annotations for episode %s; skipping.", key)
                continue

            episode = dataset[key]
            if backbone_config.visual_embedding.key not in episode:
                LOGGER.error(
                    "Embedding key '%s' not found in episode %s.", backbone_config.visual_embedding.key, key
                )
                continue
            num_steps = int(episode[backbone_config.visual_embedding.key].shape[0])
            raw_boundaries = stage_annotations[key]
            assert (len(raw_boundaries) == args.num_stages - 1), "The number of stage boundaries must be the number of stages - 1"

            if isinstance(raw_boundaries, (str, bytes)) or not isinstance(raw_boundaries, Sequence):
                LOGGER.error(
                    "Stage annotations for episode %s must be a sequence of frame indices.", key
                )
                continue

            stage_ids, subtask_progress = _build_stage_annotations(num_steps, raw_boundaries)
            _write_dataset(episode, "stage", stage_ids.astype(np.int_))
            _write_dataset(episode, "subtask_progress", subtask_progress.astype(np.float32))
            updated += 1

        LOGGER.info("Annotated %d/%d episodes with stage metadata.", updated, total)


if __name__ == '__main__':
    main(tyro.cli(Args))