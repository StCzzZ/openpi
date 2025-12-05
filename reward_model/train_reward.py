from __future__ import annotations

import dataclasses
import logging
import os
from typing import Optional

import torch
import tyro
import wandb
from torch.nn.functional import cross_entropy, mse_loss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from reward_model.config import get_reward_backbone_config
from reward_model.multi_stage_dataset import MultiStageDataset
from reward_model.reward_transformer import RewardTransformer
from reward_model.util import CosineWithMinLRScheduler, dict_apply

LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    # dataset parameters
    dataset_path: str
    output_path: str
    num_stages: int
    max_seq_len: int
    backbone: str = "dinov2_minilm"
    # training parameters
    batch_size: int = 256
    learning_rate: float = 1e-4
    num_epochs: int = 100
    num_workers: int = 4
    clip_grad: bool = True
    video_rewind: bool = False
    device: str = "cuda"
    eval_every: int = 10
    save_every: int = 20
    # model parameters
    discrete: bool = False
    video_dim: Optional[int] = None
    text_dim: Optional[int] = None
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 4
    # wandb parameters
    wandb_project: str = "reward_model"
    exp_name: str = "rewind_sarm_test_code"


def _resolve_device(device_str: str) -> torch.device:
    device = torch.device(device_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return device


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO)
    device = _resolve_device(args.device)
    backbone_config = get_reward_backbone_config(args.backbone)

    visual_key = backbone_config.visual_embedding.key
    language_key = backbone_config.language_embedding.key if backbone_config.language_embedding else None
    args.visual_embedding_key = visual_key
    args.language_embedding_key = language_key

    train_dataset = MultiStageDataset(
        dataset_path=args.dataset_path,
        num_stages=args.num_stages,
        max_seq_len=args.max_seq_len,
        video_rewind=args.video_rewind,
        visual_embedding_key=visual_key,
        language_embedding_key=language_key,
        discrete=args.discrete,
    )
    val_dataset = MultiStageDataset(
        dataset_path=args.dataset_path,
        num_stages=args.num_stages,
        max_seq_len=args.max_seq_len,
        video_rewind=args.video_rewind,
        visual_embedding_key=visual_key,
        language_embedding_key=language_key,
        discrete=args.discrete,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    video_dim = args.video_dim if args.video_dim is not None else train_dataset.visual_dim
    text_dim = args.text_dim if args.text_dim is not None else train_dataset.language_dim
    args.video_dim = video_dim
    args.text_dim = text_dim

    wandb.init(
        project=args.wandb_project,
        name=args.exp_name,
        config=dataclasses.asdict(args),
    )

    os.makedirs(os.path.join(args.output_path, args.exp_name), exist_ok=True)

    model = RewardTransformer(
        args=args,
        video_dim=video_dim,
        text_dim=text_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_stages=args.num_stages,
    )
    model.to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = CosineWithMinLRScheduler(
        optimizer,
        max_steps=args.num_epochs * len(train_loader),
        max_lr=args.learning_rate,
        min_lr=1e-5,
    )

    LOGGER.info("Training start with backbone '%s'.", backbone_config.name)
    for epoch in range(args.num_epochs):
        model.train()

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} Training"):
            optimizer.zero_grad()
            batch = dict_apply(batch, lambda x: x.to(device))
            language_inputs = batch.get("language_embeddings")

            if args.discrete:
                stage_preds, progress_preds = model(batch["visual_embeddings"], language_inputs)
                progress_loss = mse_loss(progress_preds.squeeze(-1), batch["subtask_progress"])
                stage_loss = cross_entropy(
                    stage_preds.reshape(-1, args.num_stages),
                    batch["stage"].reshape(-1),
                )
                loss = progress_loss + stage_loss
            else:
                _, progress_preds = model(batch["visual_embeddings"], language_inputs)
                loss = mse_loss(progress_preds.squeeze(-1), batch["progress"])
            
            loss.backward()
            if args.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            if args.discrete:
                wandb.log(
                    {
                        "train/progress_loss": progress_loss.item(),
                        "train/stage_loss": stage_loss.item(),
                        "train/loss": loss.item(),
                        "train/lr": scheduler.get_last_lr()[0],
                    }
                )
            else:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/lr": scheduler.get_last_lr()[0],
                    }
                )

        LOGGER.info("Epoch %d completed; Train loss: %.4f", epoch + 1, loss.item())

        if epoch % args.eval_every == 0:
            model.eval()
            val_loss = 0.0
            val_progress_loss = 0.0
            val_stage_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1} Validation"):
                    batch = dict_apply(batch, lambda x: x.to(device))
                    language_inputs = batch.get("language_embeddings")
                    if args.discrete:
                        stage_preds, progress_preds = model(batch["visual_embeddings"], language_inputs)
                        progress_loss = mse_loss(progress_preds.squeeze(-1), batch["subtask_progress"])
                        stage_loss = cross_entropy(
                            stage_preds.reshape(-1, args.num_stages),
                            batch["stage"].reshape(-1),
                        )
                        curr_val_loss = progress_loss + stage_loss
                        val_loss += curr_val_loss.item()
                        val_progress_loss += progress_loss.item()
                        val_stage_loss += stage_loss.item()
                    else:
                        _, progress_preds = model(batch["visual_embeddings"], language_inputs)
                        progress_loss = mse_loss(progress_preds.squeeze(-1), batch["progress"])
                        val_loss += progress_loss.item()

            if args.discrete:
                wandb.log(
                    {
                        "val/progress_loss": val_progress_loss / len(val_loader),
                        "val/stage_loss": val_stage_loss / len(val_loader),
                        "val/loss": val_loss / len(val_loader),
                    }
                )
            else:
                wandb.log(
                    {
                        "val/loss": val_loss / len(val_loader),
                    }
                )
            LOGGER.info(
                "Epoch %d completed; Validation loss: %.4f",
                epoch + 1,
                val_loss / len(val_loader),
            )

        if epoch % args.save_every == 0 or epoch == args.num_epochs - 1:
            save_dict = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "args": dataclasses.asdict(args),
            }
            save_path = os.path.join(args.output_path, args.exp_name, f"reward_model_{epoch}.pt")
            torch.save(save_dict, save_path)
            LOGGER.info("Epoch %d completed; Model saved to %s", epoch + 1, save_path)


if __name__ == "__main__":
    main(tyro.cli(Args))
