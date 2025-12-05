import dataclasses
import logging
from pathlib import Path
from typing import Optional

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
    output_dir: str  # Directory to save the episode images
    camera_key: str = "side_cam"  # Camera key to extract from episodes


def save_episode_images(
    episode_name: str,
    frames: np.ndarray,
    episode_dir: Path,
    camera_key: str
) -> None:
    """
    Save episode frames as individual PNG images.
    
    Args:
        episode_name: Name of the episode
        frames: Array of frames with shape (num_frames, height, width, channels)
        episode_dir: Directory to save the episode images
        camera_key: Camera key being processed
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
    
    logger.info(f"Saving {num_frames} images for episode '{episode_name}' ({height}x{width}, {channels} channels)")
    
    # Create episode directory
    episode_dir.mkdir(parents=True, exist_ok=True)
    
    # Save each frame as PNG
    for frame_idx in tqdm(range(num_frames), desc=f"Saving images for {episode_name}"):
        frame = frames[frame_idx]
        
        # Ensure frame is in the correct format (uint8)
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
        
        # Handle different image formats
        if not is_color or len(frame.shape) == 2:
            # Grayscale image - keep as grayscale for PNG
            processed_frame = frame
        elif frame.shape[2] == 3:
            # RGB image - convert to BGR for OpenCV
            processed_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif frame.shape[2] == 4:
            # RGBA image - convert to BGRA for OpenCV
            processed_frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA)
        else:
            # Use frame as-is
            processed_frame = frame
        
        # Save frame as PNG
        image_path = episode_dir / f"{frame_idx}.png"
        success = cv2.imwrite(str(image_path), processed_frame)
        
        if not success:
            logger.error(f"Failed to save image {image_path}")
    
    logger.info(f"✓ Saved {num_frames} images to {episode_dir}")


def main(args: Args) -> None:
    """
    Main function to read h5py dataset and save images as PNG files.
    
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
            
            # Check if the specified camera key exists in the episode
            if args.camera_key not in episode:
                logger.warning(f"Episode '{key}' does not have '{args.camera_key}' data. Skipping.")
                continue
            
            # Extract camera frames
            camera_frames = episode[args.camera_key][:]
            logger.info(f"Episode '{key}': {camera_frames.shape[0]} frames from '{args.camera_key}'")
            
            # Create episode directory
            episode_dir = output_dir / f"episode_{key}"
            
            # Save the images
            save_episode_images(key, camera_frames, episode_dir, args.camera_key)
    
    logger.info("All images saved successfully!")


if __name__ == '__main__':
    main(tyro.cli(Args))
