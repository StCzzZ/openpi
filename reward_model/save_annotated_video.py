import dataclasses
import logging
import subprocess
import tempfile
from pathlib import Path

import cv2
import h5py
import numpy as np
import tyro
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    data_path: str  # Path to the h5py dataset file
    output_dir: str  # Directory to save the annotated videos
    fps: int = 30  # Frames per second for the output videos
    font_scale: float = 1.0  # Font scale for frame number annotation
    font_thickness: int = 2  # Font thickness for frame number annotation
    font_color: tuple[int, int, int] = (0, 255, 0)  # Font color in BGR format (green)
    text_position: tuple[int, int] = (10, 30)  # Text position (x, y) in pixels


def annotate_frame(
    frame: np.ndarray,
    frame_number: int,
    font_scale: float,
    font_thickness: int,
    font_color: tuple[int, int, int],
    text_position: tuple[int, int]
) -> np.ndarray:
    """
    Annotate a frame with its frame number.
    
    Args:
        frame: Input frame as a numpy array
        frame_number: Frame number to annotate
        font_scale: Font scale for the text
        font_thickness: Font thickness
        font_color: Font color in BGR format
        text_position: Position of the text (x, y)
    
    Returns:
        Annotated frame
    """
    annotated_frame = frame.copy()
    text = f"Frame: {frame_number}"
    
    cv2.putText(
        annotated_frame,
        text,
        text_position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        font_color,
        font_thickness,
        cv2.LINE_AA
    )
    
    return annotated_frame


def save_episode_video(
    episode_name: str,
    frames: np.ndarray,
    output_path: Path,
    args: Args
) -> None:
    """
    Save an episode as an annotated video using ffmpeg with H.264 encoding.
    
    Args:
        episode_name: Name of the episode
        frames: Array of frames with shape (num_frames, height, width, channels)
        output_path: Path to save the output video
        args: Arguments containing video parameters
    """
    num_frames = frames.shape[0]
    height, width = frames.shape[1:3]
    
    # Determine if the frames are grayscale or color
    if len(frames.shape) == 3:
        # Grayscale images
        is_color = False
        channels = 1
    else:
        is_color = True
        channels = frames.shape[3]
    
    logger.info(f"Saving video for episode '{episode_name}' with {num_frames} frames ({height}x{width}, {channels} channels)")
    
    # Create a temporary directory to store annotated frames
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        filelist_path = temp_path / 'filelist.txt'
        
        # Save annotated frames to temporary directory
        logger.info("Saving annotated frames to temporary directory...")
        frame_paths = []
        for frame_idx in tqdm(range(num_frames), desc=f"Annotating frames for {episode_name}"):
            frame = frames[frame_idx]
            
            # Ensure frame is in the correct format (uint8)
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
            
            # Convert grayscale to BGR if needed
            if not is_color or len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            # Convert RGB to BGR if needed (h5py images are typically RGB)
            elif frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Annotate the frame
            annotated_frame = annotate_frame(
                frame,
                frame_idx,
                args.font_scale,
                args.font_thickness,
                args.font_color,
                args.text_position
            )
            
            # Save frame to temporary file
            frame_path = temp_path / f"frame_{frame_idx:06d}.png"
            cv2.imwrite(str(frame_path), annotated_frame)
            frame_paths.append(frame_path)
        
        # Create file list for ffmpeg
        logger.info("Creating video with ffmpeg...")
        with open(filelist_path, 'w') as f:
            for frame_path in frame_paths:
                # Use absolute path and escape single quotes
                img_path = str(frame_path.absolute()).replace("'", "'\\''")
                f.write(f"file '{img_path}'\n")
        
        # Use ffmpeg to create video with H.264 codec and yuv420p pixel format
        cmd = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-r', str(args.fps),
            '-i', str(filelist_path),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-y',  # Overwrite output file if it exists
            str(output_path)
        ]
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode == 0:
            logger.info(f"✓ Video saved to {output_path}")
        else:
            logger.error(f"✗ Failed to create video {output_path}")
            logger.error(f"FFmpeg error: {result.stderr}")


def main(args: Args) -> None:
    """
    Main function to read h5py dataset and save annotated videos.
    
    Args:
        args: Command-line arguments
    """
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Open the h5py dataset
    logger.info(f"Opening dataset: {args.data_path}")
    with h5py.File(args.data_path, 'r') as f:
        logger.info(f"Found {len(f.keys())} episodes in the dataset")
        
        for key in tqdm(f.keys(), desc="Processing episodes"):
            episode = f[key]
            
            # Check if 'side_cam' exists in the episode
            if 'side_cam' not in episode:
                logger.warning(f"Episode '{key}' does not have 'side_cam' data. Skipping.")
                continue
            
            # Extract side_cam frames
            side_cam_frames = episode['side_cam'][:]
            logger.info(f"Episode '{key}': {side_cam_frames.shape[0]} frames")
            
            # Define output video path
            output_path = output_dir / f"{key}.mp4"
            
            # Save the video
            save_episode_video(key, side_cam_frames, output_path, args)
    
    logger.info("All videos saved successfully!")


if __name__ == '__main__':
    main(tyro.cli(Args))

