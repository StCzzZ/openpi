import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tyro
import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R

REPO_NAME = "Virlus/kitchen_rollout"

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
                "shape": (6,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    with h5py.File(data_dir, "r") as f:
        for key in f:
            episode = f[key]
            num_steps = episode["action"].shape[0]
            for step in range(num_steps):
                dataset.add_frame(
                    {
                        "image": episode["side_cam"][step],
                        "wrist_image": episode["wrist_cam"][step],
                        "state": episode["tcp_pose"][step].astype(np.float32),
                        "actions": episode["action"][step].astype(np.float32),
                        "task": "Put the items in the pot",
                    }
                )
            dataset.save_episode()


if __name__ == "__main__":
    tyro.cli(main)