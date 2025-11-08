import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tyro
import h5py
import numpy as np

REPO_NAME = "Virlus/flexiv_fold_towel_twice"

def main(data_dir: str, *, push_to_hub: bool = False):
    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="single_flexiv_rizon4",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (9,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (10,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )
    
    with h5py.File(data_dir, 'r') as f:
        for key in f:
            episode = f[key]
            for step in range(episode["action"].shape[0]):
                dataset.add_frame(
                    {
                        "image": episode['side_cam'][step],
                        "wrist_image": episode['wrist_cam'][step],
                        "state": episode['tcp_pose'][step].astype(np.float32),
                        "actions": episode['action'][step].astype(np.float32),
                        "task": 'fold the towel twice by grabbing its corners',
                    }
                )
            dataset.save_episode()


if __name__ == "__main__":
    tyro.cli(main)