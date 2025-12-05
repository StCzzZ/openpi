import torch
from typing import Callable, Dict, Any
import math
from torch.optim import Optimizer
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple
from tqdm import tqdm


def dict_apply(dict_data: Dict[str, Any], func: Callable[[Any], Any]) -> Dict[str, Any]:
    for key, value in dict_data.items():
        if isinstance(value, dict):
            dict_data[key] = dict_apply(value, func)
        else:
            dict_data[key] = func(value)
    return dict_data

dino_transform_image = T.Compose(
    [T.ToTensor(), T.Normalize([0.5], [0.5])]
)

def dino_load_image(img: np.ndarray) -> torch.Tensor:
    """
    Load an image and return a tensor that can be used as an input to DINOv2.
    """
    img = Image.fromarray(img)

    transformed_img = dino_transform_image(img)[:3].unsqueeze(0)

    return transformed_img


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def save_episode_reward_video(
    episode_name: str,
    frames: np.ndarray,
    rewards: np.ndarray,
    output_path: Path,
    fps: int
) -> None:
    """
    Create a side-by-side visualization of the episode frames and cumulative rewards.
    """
    rewards_np = np.asarray(rewards)
    frames_np = np.asarray(frames)

    if frames_np.ndim == 4 and frames_np.shape[1] in (1, 3) and frames_np.shape[-1] not in (1, 3):
        frames_np = np.transpose(frames_np, (0, 2, 3, 1))

    frame_indices = _select_frame_indices(frames_np.shape[0], rewards_np.shape[0])

    reward_min = float(np.min(rewards_np))
    reward_max = float(np.max(rewards_np))
    if np.isclose(reward_min, reward_max):
        reward_min -= 0.1
        reward_max += 0.1
    reward_range = reward_max - reward_min
    reward_bounds = (reward_min - 0.05 * reward_range, reward_max + 0.05 * reward_range)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        filelist_path = temp_path / "filelist.txt"
        frame_paths = []
        for viz_idx, frame_idx in enumerate(
            tqdm(frame_indices, desc=f"Rendering video for {episode_name}")
        ):
            base_frame = _ensure_uint8_bgr(frames_np[frame_idx])
            # Increase plot width to accommodate captions on the rightmost curve
            # Ensure width is even for H.264 encoding compatibility
            plot_width = int(base_frame.shape[1] * 1.3)  # 30% wider
            if plot_width % 2 == 1:  # Make sure width is even
                plot_width += 1
            plot_frame = _render_reward_plot_frame(
                rewards=rewards_np,
                current_step=viz_idx,
                height=base_frame.shape[0],
                width=plot_width,
                reward_bounds=reward_bounds,
            )
            combined_frame = np.concatenate((base_frame, plot_frame), axis=1)
            frame_path = temp_path / f"frame_{viz_idx:06d}.png"
            cv2.imwrite(str(frame_path), combined_frame)
            frame_paths.append(frame_path)

        with open(filelist_path, "w", encoding="utf-8") as file_handle:
            for frame_path in frame_paths:
                img_path = str(frame_path.absolute()).replace("'", "'\\''")
                file_handle.write(f"file '{img_path}'\n")

        cmd = [
            "ffmpeg",
            "-f",
            "concat",
            "-safe",
            "0",
            "-r",
            str(fps),
            "-i",
            str(filelist_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(output_path),
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            print("✓ Video saved to %s for episode '%s'", output_path, episode_name)
        else:
            print("✗ Failed to create video %s for episode '%s'", output_path, episode_name)
            print("FFmpeg error: %s", result.stderr)


def _render_reward_plot_frame(
    rewards: np.ndarray,
    current_step: int,
    height: int,
    width: int,
    reward_bounds: Tuple[float, float]
) -> np.ndarray:
    plot_img = np.full((height, width, 3), 255, dtype=np.uint8)
    margin_left, margin_right = 60, 80  # Increased right margin for captions
    margin_top, margin_bottom = 20, 40
    axis_width = max(width - margin_left - margin_right, 1)
    axis_height = max(height - margin_top - margin_bottom, 1)

    origin = (margin_left, height - margin_bottom)
    x_axis_end = (width - margin_right, height - margin_bottom)
    y_axis_end = (margin_left, margin_top)

    cv2.line(plot_img, origin, x_axis_end, color=(0, 0, 0), thickness=2)
    cv2.line(plot_img, origin, y_axis_end, color=(0, 0, 0), thickness=2)

    total_steps = len(rewards) - 1 if len(rewards) > 1 else 1
    y_denominator = reward_bounds[1] - reward_bounds[0]
    y_denominator = y_denominator if y_denominator != 0 else 1e-6

    points = []
    for idx in range(current_step + 1):
        normalized_x = idx / total_steps
        normalized_y = (rewards[idx] - reward_bounds[0]) / y_denominator
        normalized_y = np.clip(normalized_y, 0.0, 1.0)

        px = margin_left + int(normalized_x * axis_width)
        py = origin[1] - int(normalized_y * axis_height)
        points.append((px, py))

    if points:
        polyline = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(plot_img, [polyline], isClosed=False, color=(235, 99, 37), thickness=3)
        cv2.circle(plot_img, points[-1], radius=6, color=(38, 38, 220), thickness=-1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        plot_img,
        "Timestep",
        (int((margin_left + x_axis_end[0]) / 2) - 40, height - 5),
        font,
        0.5,
        (80, 80, 80),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        plot_img,
        "Predicted Reward",
        (5, margin_top - 5 if margin_top > 5 else 15),
        font,
        0.5,
        (80, 80, 80),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        plot_img,
        f"{reward_bounds[0]:.2f}",
        (5, height - margin_bottom),
        font,
        0.5,
        (100, 100, 100),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        plot_img,
        f"{reward_bounds[1]:.2f}",
        (5, margin_top + 5),
        font,
        0.5,
        (100, 100, 100),
        1,
        cv2.LINE_AA,
    )
    if points:
        cv2.putText(
            plot_img,
            f"{rewards[current_step]:.2f}",
            (points[-1][0] + 5, points[-1][1] - 5),
            font,
            0.5,
            (38, 38, 220),
            1,
            cv2.LINE_AA,
        )

    return plot_img


def _ensure_uint8_bgr(frame: np.ndarray) -> np.ndarray:
    frame_np = np.asarray(frame)
    if frame_np.ndim == 2:
        frame_np = np.expand_dims(frame_np, axis=-1)
    if frame_np.shape[-1] == 1:
        frame_np = np.repeat(frame_np, 3, axis=-1)
    if frame_np.dtype != np.uint8:
        frame_max = float(frame_np.max()) if frame_np.size else 1.0
        if frame_max <= 1.0:
            frame_np = (frame_np * 255.0).clip(0, 255).astype(np.uint8)
        else:
            frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
    if frame_np.shape[-1] > 3:
        frame_np = frame_np[..., :3]
    if frame_np.shape[-1] == 3:
        return cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(frame_np, cv2.COLOR_GRAY2BGR)


def _select_frame_indices(num_frames: int, num_steps: int) -> np.ndarray:
    if num_frames == 0 or num_steps == 0:
        return np.array([], dtype=int)
    if num_steps <= num_frames:
        return np.linspace(0, num_frames - 1, num_steps).astype(int)
    repeated_indices = np.concatenate(
        (np.arange(num_frames, dtype=int), np.full(num_steps - num_frames, num_frames - 1, dtype=int))
    )
    return repeated_indices


class CosineWithMinLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer: Optimizer, max_steps: int, max_lr: float, min_lr: float, last_epoch: int = -1):
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch <= self.max_steps:
            # Cosine decay for the first max_steps
            cos_decay = 0.5 * (1 + math.cos(math.pi * self.last_epoch / self.max_steps))
            return [self.min_lr + (self.max_lr - self.min_lr) * cos_decay for _ in self.base_lrs]
        else:
            # Keep the minimum learning rate
            return [self.min_lr for _ in self.base_lrs]