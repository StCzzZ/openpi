from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
import tyro
from easydict import EasyDict
from tqdm import tqdm

from reward_model.config import get_reward_backbone_config
from reward_model.embedding_extractors import ExtractorInitParams, build_embedding_extractor
from reward_model.multi_stage_dataset import MultiStageDataset
from reward_model.reward_transformer import RewardTransformer
from reward_model.util import save_episode_reward_video

LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    dataset_path: str
    output_path: str
    reward_model_path: str
    dino_ckpt_path: Optional[str] = None
    batch_size: int = 32
    device: str = "cuda"
    video_fps: int = 10


def _resolve_device(device_str: str) -> torch.device:
    device = torch.device(device_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return device


def _build_sliding_windows(
    dataset: MultiStageDataset,
    embeddings: np.ndarray,
    max_seq_len: int,
) -> np.ndarray:
    padded_sequences = [
        dataset.padding_sequence(embeddings[:-i]) for i in range(embeddings.shape[0] - 1, 0, -1)
    ]
    padded_sequences.append(dataset.padding_sequence(embeddings))
    return np.stack(padded_sequences, axis=0)


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO)
    device = _resolve_device(args.device)
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_dict = torch.load(args.reward_model_path, map_location="cpu")
    train_config = EasyDict(**saved_dict["args"])
    backbone_name = getattr(train_config, "backbone", "dinov2_minilm")
    backbone_config = get_reward_backbone_config(backbone_name)

    reward_model = RewardTransformer(
        args=train_config,
        video_dim=train_config.video_dim,
        text_dim=train_config.text_dim,
        hidden_dim=train_config.hidden_dim,
        num_heads=train_config.num_heads,
        num_layers=train_config.num_layers,
        num_stages=train_config.num_stages,
    )
    reward_model.load_state_dict(saved_dict["model_state_dict"])
    reward_model.to(device)
    reward_model.eval()

    visual_key = getattr(train_config, "visual_embedding_key", backbone_config.visual_embedding.key)
    default_language_key = (
        backbone_config.language_embedding.key if backbone_config.language_embedding else None
    )
    language_key = getattr(train_config, "language_embedding_key", default_language_key)
    train_dataset = MultiStageDataset(
        dataset_path=train_config.dataset_path,
        num_stages=train_config.num_stages,
        max_seq_len=train_config.max_seq_len,
        video_rewind=train_config.video_rewind,
        visual_embedding_key=visual_key,
        language_embedding_key=language_key,
    )
    stage_prior = torch.from_numpy(train_dataset.stage_prior).to(device)
    cumulative_stage_prior = torch.from_numpy(train_dataset.cumulative_stage_prior).to(device)

    extractor = build_embedding_extractor(
        backbone_config,
        ExtractorInitParams(device=device, dino_ckpt_path=args.dino_ckpt_path),
    )
    prompt = getattr(train_config, "prompt")
    language_embedding = extractor.get_language_embedding(prompt)

    with h5py.File(args.dataset_path, "r") as dataset:
        for key in tqdm(dataset.keys(), desc="Processing episodes"):
            episode = dataset[key]
            frames = episode["side_cam"]
            visual_embeddings = extractor.extract_visual_embeddings(
                frames=frames,
                batch_size=args.batch_size,
                prompt=prompt,
            )

            padded_visual_embeddings = _build_sliding_windows(
                train_dataset,
                visual_embeddings,
                train_config.max_seq_len,
            )
            padded_visual_embeddings = torch.from_numpy(padded_visual_embeddings).to(device)

            language_tensor = None
            if language_embedding is not None and train_config.text_dim:
                repeated = np.repeat(
                    np.expand_dims(language_embedding, axis=0),
                    padded_visual_embeddings.shape[0],
                    axis=0,
                )
                language_tensor = torch.from_numpy(repeated).to(device)

            pred_mask = np.zeros(
                (padded_visual_embeddings.shape[0], train_config.max_seq_len),
                dtype=bool,
            )
            offset_mask = np.zeros(
                (padded_visual_embeddings.shape[0], padded_visual_embeddings.shape[0]),
                dtype=bool,
            )
            for idx in range(padded_visual_embeddings.shape[0]):
                pred_mask[idx, min(idx, train_config.max_seq_len - 1)] = 1
                offset_mask[idx, max(idx - train_config.max_seq_len + 1, 0)] = 1
            pred_mask_tensor = torch.from_numpy(pred_mask).to(device)
            offset_mask_tensor = torch.from_numpy(offset_mask).to(device)

            with torch.no_grad():
                stage_preds, progress_preds = reward_model(padded_visual_embeddings, language_tensor)
                if stage_preds is not None:
                    stage_preds = torch.argmax(stage_preds, dim=-1)
                    progress_preds = progress_preds.squeeze(-1)

                    stage_preds = stage_preds[pred_mask_tensor]
                    progress_preds = progress_preds[pred_mask_tensor]

                    prior_progress = cumulative_stage_prior[stage_preds]
                    total_progress_pred = prior_progress + progress_preds * stage_prior[stage_preds]
                else:
                    progress_preds = progress_preds.squeeze(-1)
                    total_progress_pred = progress_preds[pred_mask_tensor] # (num_frames, )
                    offset_progress_pred = total_progress_pred.unsqueeze(0).repeat(padded_visual_embeddings.shape[0], 1)
                    total_progress_pred = offset_progress_pred[offset_mask_tensor] \
                        + (1. - offset_progress_pred[offset_mask_tensor]) * total_progress_pred # (num_frames, )

            reward_sequence = total_progress_pred.detach().cpu().numpy()
            episode_output_path = output_dir / f"{key}.mp4"
            save_episode_reward_video(
                episode_name=str(key),
                frames=episode["side_cam"][:],
                rewards=reward_sequence,
                output_path=episode_output_path,
                fps=args.video_fps,
            )
            LOGGER.info("Saved visualization for episode %s to %s", key, episode_output_path)


if __name__ == "__main__":
    main(tyro.cli(Args))
