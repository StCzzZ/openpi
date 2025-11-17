import torch
import os
import wandb
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn.functional import mse_loss, cross_entropy
import dataclasses
import tyro
from tqdm import tqdm


from reward_model.multi_stage_dataset import MultiStageDataset
from reward_model.reward_transformer import RewardTransformer
from reward_model.util import CosineWithMinLRScheduler, dict_apply

@dataclasses.dataclass
class Args:
    # dataset parameters
    dataset_path: str
    output_path: str
    num_stages: int
    max_seq_len: int
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
    video_dim: int = 768
    text_dim: int = 384
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 4
    # wandb parameters
    wandb_project: str = "reward_model"
    exp_name: str = "simple_sarm"


def main(args: Args):
    train_dataset = MultiStageDataset(
        dataset_path=args.dataset_path,
        num_stages=args.num_stages,
        max_seq_len=args.max_seq_len,
        video_rewind=args.video_rewind
    )
    val_dataset = MultiStageDataset(
        dataset_path=args.dataset_path,
        num_stages=args.num_stages,
        max_seq_len=args.max_seq_len,
        video_rewind=args.video_rewind
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    wandb.init(
        project=args.wandb_project,
        name=args.exp_name,
        config=dataclasses.asdict(args)
    )

    os.makedirs(os.path.join(args.output_path, args.exp_name), exist_ok=True)
    
    model = RewardTransformer(
        args=args,
        video_dim=args.video_dim,
        text_dim=args.text_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_stages=args.num_stages
    )
    device = torch.device(args.device)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = CosineWithMinLRScheduler(optimizer, max_steps=args.num_epochs * len(train_loader), max_lr=args.learning_rate, min_lr=1e-5)

    print("Training start...")
    for epoch in range(args.num_epochs):
        model.train()

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            optimizer.zero_grad()
            batch = dict_apply(batch, lambda x: x.to(device))
            stage_preds, progress_preds = model(batch['dino_embeddings'], batch['minlm_task_embedding'])
            progress_loss = mse_loss(progress_preds.squeeze(-1), batch['subtask_progress'])
            stage_loss = cross_entropy(stage_preds.reshape(-1, args.num_stages), batch['stage'].reshape(-1))
            loss = progress_loss + stage_loss
            loss.backward()
            if args.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            wandb_log = {
                "train/progress_loss": progress_loss.item(),
                "train/stage_loss": stage_loss.item(),
                "train/loss": loss.item(),
                "train/lr": scheduler.get_last_lr()[0],
            }
            wandb.log(wandb_log)

        print(f"Epoch {epoch+1} completed; Train loss: {loss.item()}")

        if epoch % args.eval_every == 0:
            model.eval()
            val_loss = 0
            val_progress_loss = 0
            val_stage_loss = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
                    batch = dict_apply(batch, lambda x: x.to(device))
                    stage_preds, progress_preds = model(batch['dino_embeddings'], batch['minlm_task_embedding'])
                    progress_loss = mse_loss(progress_preds.squeeze(-1), batch['subtask_progress'])
                    stage_loss = cross_entropy(stage_preds.reshape(-1, args.num_stages), batch['stage'].reshape(-1))
                    curr_val_loss = progress_loss + stage_loss
                    val_loss += curr_val_loss.item()
                    val_progress_loss += progress_loss.item()
                    val_stage_loss += stage_loss.item()
            wandb_log = {
                "val/progress_loss": val_progress_loss / len(val_loader),
                "val/stage_loss": val_stage_loss / len(val_loader),
                "val/loss": val_loss / len(val_loader),
            }
            wandb.log(wandb_log)

            print(f"Epoch {epoch+1} completed; Validation loss: {val_loss / len(val_loader)}")

        if epoch % args.save_every == 0 or epoch == args.num_epochs - 1:
            save_dict = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "args": dataclasses.asdict(args),
            }
            torch.save(save_dict, os.path.join(args.output_path, args.exp_name, f"reward_model_{epoch}.pt"))
            print(f"Epoch {epoch+1} completed; Model saved to {os.path.join(args.output_path, args.exp_name, f'reward_model_{epoch}.pt')}")


if __name__ == "__main__":
    main(tyro.cli(Args))