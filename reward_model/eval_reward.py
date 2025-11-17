import dataclasses
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple

import cv2
import h5py
import numpy as np
import torch
import tyro
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from easydict import EasyDict

from reward_model.multi_stage_dataset import MultiStageDataset
from reward_model.reward_transformer import RewardTransformer
from reward_model.util import dino_load_image, mean_pooling, save_episode_reward_video
from third_party.dinov2.model import vit_base


PROMPT = "Put the items in the pot."


@dataclasses.dataclass
class Args:
    # dataset parameters
    dataset_path: str
    output_path: str
    reward_model_path: str
    vision_encoder_path: str
    # training parameters
    batch_size: int = 32
    device: str = "cuda"
    # visualization parameters
    video_fps: int = 10


def load_model(args: Args, device: torch.device) -> Tuple[torch.nn.Module, AutoTokenizer, AutoModel]:
    # load pretrained DINOv2 encoder
    dino_encoder = vit_base(
        img_size=518,
        patch_size=14,
        init_values=1.0,
        ffn_layer="mlp",
        block_chunks=0,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
    ).to(device)
    dino_encoder.load_state_dict(torch.load(args.vision_encoder_path), strict=True)

    # load pretrained all-MiniLM-L12-v2 encoder following ReWIND
    minilm_tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/all-MiniLM-L12-v2"
    )
    minilm_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L12-v2").to(
        device
    )

    return dino_encoder, minilm_tokenizer, minilm_model


def main(args: Args) -> None:
    # Load trained model and dataset
    saved_dict = torch.load(args.reward_model_path)
    train_config = saved_dict['args']
    train_config = EasyDict(**train_config)
    device = torch.device(args.device)
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    reward_model = RewardTransformer(
        args=train_config,
        video_dim=train_config.video_dim,
        text_dim=train_config.text_dim,
        hidden_dim=train_config.hidden_dim,
        num_heads=train_config.num_heads,
        num_layers=train_config.num_layers,
        num_stages=train_config.num_stages
    )
    reward_model.load_state_dict(saved_dict['model_state_dict'])
    reward_model.to(device)
    reward_model.eval()

    # Load stage prior from the pretrained dataset
    train_dataset = MultiStageDataset(
        dataset_path=train_config.dataset_path,
        num_stages=train_config.num_stages,
        max_seq_len=train_config.max_seq_len
    )
    stage_prior = torch.from_numpy(train_dataset.stage_prior).to(device)
    cumulative_stage_prior = torch.from_numpy(train_dataset.cumulative_stage_prior).to(device)

    # Load pretrained vision and language encoder
    dino_encoder, minilm_tokenizer, minilm_model = load_model(args, device)
    with torch.no_grad():
        encoded_input = minilm_tokenizer(
                [PROMPT], padding=False, truncation=True, return_tensors="pt"
            ).to(device)
        model_output = minilm_model(**encoded_input)
        minlm_task_embedding = (
            mean_pooling(model_output, encoded_input["attention_mask"])
            .cpu()
            .detach()
            .numpy()
        )

    # Read the dataset to be processed
    with h5py.File(args.dataset_path, 'r') as f:
        for key in tqdm(f, desc="Processing episodes"):
            episode = f[key]
            side_cam_frames = episode['side_cam'][:]
            num_episode_steps = side_cam_frames.shape[0]
            # ================ Extract DINO embeddings ================
            dino_embeddings = []
            for step in tqdm(range(0, num_episode_steps, args.batch_size), desc="Extracting DINO embeddings"):
                batch_images = side_cam_frames[step:min(step+args.batch_size, num_episode_steps)]
                batch_images = [dino_load_image(img) for img in batch_images]
                batch_images = torch.cat(batch_images, dim=0)
                batch_images = batch_images.to(device)
                with torch.no_grad():
                    batch_embeddings = dino_encoder(batch_images)
                    batch_embeddings = batch_embeddings.cpu().detach().numpy()
                    dino_embeddings.append(batch_embeddings)
            
            dino_embeddings = np.concatenate(dino_embeddings, axis=0)
            # ================ Extract DINO embeddings ================

            # ================= Evaluate pretrained reward model ================
            padded_dino_embeddings = [train_dataset.padding_sequence(dino_embeddings[:-i]) \
                for i in range(train_config.max_seq_len - 1, 0, -1)] + [train_dataset.padding_sequence(dino_embeddings)]
            padded_dino_embeddings = np.stack(padded_dino_embeddings, axis=0)
            padded_dino_embeddings = torch.from_numpy(padded_dino_embeddings).to(device)

            curr_minlm_task_embedding = np.repeat(np.expand_dims(minlm_task_embedding, axis=0), padded_dino_embeddings.shape[0], axis=0)
            curr_minlm_task_embedding = torch.from_numpy(curr_minlm_task_embedding).to(device)

            pred_mask = np.zeros((padded_dino_embeddings.shape[0], train_config.max_seq_len), dtype=bool)
            for i in range(padded_dino_embeddings.shape[0]):
                pred_mask[i, min(i, train_config.max_seq_len - 1)] = 1
            pred_mask = torch.from_numpy(pred_mask).to(device)

            with torch.no_grad():
                stage_preds, progress_preds = reward_model(padded_dino_embeddings, curr_minlm_task_embedding)
                stage_preds = torch.argmax(stage_preds, dim=-1)
                progress_preds = progress_preds.squeeze(-1)

                stage_preds = stage_preds[pred_mask] # (num_steps, )
                progress_preds = progress_preds[pred_mask] # (num_steps,)

                prior_progress = cumulative_stage_prior[stage_preds]
                total_progress_pred = prior_progress + progress_preds * stage_prior[stage_preds]
            
            # ================= Evaluate pretrained reward model ================
            reward_sequence = total_progress_pred.detach().cpu().numpy()
            episode_output_path = output_dir / f"{key}.mp4"
            save_episode_reward_video(
                episode_name=str(key),
                frames=side_cam_frames,
                rewards=reward_sequence,
                output_path=episode_output_path,
                fps=args.video_fps
            )


if __name__ == "__main__":
    main(tyro.cli(Args))