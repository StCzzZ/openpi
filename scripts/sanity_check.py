import time
import os
import numpy as np
from dataclasses import dataclass
from typing import List
from scipy.spatial.transform import Rotation as R
import h5py
import tyro
from PIL import Image

from hardware.robot_env import RobotEnv
from hardware.my_device.macros import CAM_SERIAL


@dataclass
class Args:
    save_path: str
    resolution: List[int]
    fps: float = 10.0


def main(args: Args):
    robot_env = RobotEnv(camera_serial=CAM_SERIAL, img_shape=[3]+args.resolution, fps=args.fps)
    
    with h5py.File(args.save_path, 'r') as f:
        # Find the highest episode_id in the existing file
        episode_keys = [key for key in f.keys() if key.startswith('episode_')]
        
        for episode_key in episode_keys:
            episode = f[episode_key]
            robot_env.reset_robot()
            robot_env.deploy_action(episode['tcp_pose'][0], episode['action'][0][6])
            for step in range(episode['action'].shape[0]):
                start_time = time.time()
                action = episode['action'][step]
                robot_env.deploy_action(action[:6], action[6])
                time.sleep(max(1 / args.fps - (time.time() - start_time), 0))
            for step in range(episode['side_cam'].shape[0]):
                assert (f[episode_key]['side_cam'][step].shape[:2] == tuple(args.resolution))
                Image.fromarray(f[episode_key]['side_cam'][step]).save(f"/mnt/workspace/pi05/image/side_img_{step}.png")
                Image.fromarray(f[episode_key]['wrist_cam'][step]).save(f"/mnt/workspace/pi05/image/wrist_img_{step}.png")
            
            print("Finished replaying ", episode_key)


if __name__ == '__main__':
    main(tyro.cli(Args))