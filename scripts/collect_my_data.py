import time
import os
import numpy as np
from dataclasses import dataclass
from typing import List
from scipy.spatial.transform import Rotation as R
import h5py
import tyro

from hardware.robot_env import RobotEnv
from hardware.my_device.macros import CAM_SERIAL, CANONICAL_EULER_ANGLES


@dataclass
class Args:
    save_path: str
    resolution: List[int]
    fps: float = 10.0


def main(args: Args):
    robot_env = RobotEnv(camera_serial=CAM_SERIAL, img_shape=[3]+args.resolution, fps=args.fps)
    
    # Determine starting episode_id based on existing file
    episode_id = 0
    if os.path.exists(args.save_path):
        with h5py.File(args.save_path, 'r') as f:
            # Find the highest episode_id in the existing file
            episode_keys = [key for key in f.keys() if key.startswith('episode_')]
            if episode_keys:
                episode_ids = [int(key.split('_')[1]) for key in episode_keys]
                episode_id = max(episode_ids) + 1
                print(f"Appending to existing file. Starting from episode {episode_id}")
    
    with h5py.File(args.save_path, 'a') as f:
        while not robot_env.keyboard.quit:
            print("start recording...")
            tcp_pose = []
            joint_pos = []
            action = []
            wrist_cam = []
            side_cam = []

            robot_env.keyboard.start = False
            robot_env.keyboard.discard = False
            robot_env.keyboard.finish = False
            cnt = 0

            robot_env.reset_robot()

            seed = int(time.time())
            np.random.seed(seed)

            while not robot_env.keyboard.quit and not robot_env.keyboard.discard and not robot_env.keyboard.finish:
                transition_data = robot_env.human_teleop_step()
                if not robot_env.keyboard.start or transition_data is None:
                    continue

                # Initialize at the beginning of the episode
                if cnt == 0:
                    random_init_pose = robot_env.robot.init_pose + np.random.uniform(-0.1, 0.1, size=7)
                    random_init_pose[2] = max(random_init_pose[2], 0.15)
                    robot_env.reset_robot(random_init=True, random_init_pose=random_init_pose)
                    cnt += 1
                    print("Episode start!")
                    continue
                
                wrist_cam.append(transition_data['wrist_img'])
                side_cam.append(transition_data['side_img'])
                tcp_pose.append(transition_data['tcp_pose'])
                joint_pos.append(transition_data['joint_pos'])
                action.append(transition_data['action'])

            if not robot_env.keyboard.start or robot_env.keyboard.quit or robot_env.keyboard.discard:
                print('WARNING: discard the demo!')
                robot_env.gripper.move(robot_env.gripper.max_width)
                time.sleep(0.5)
                continue
            
            episode = dict()
            episode['wrist_cam'] = np.stack(wrist_cam, axis=0)
            episode['side_cam'] = np.stack(side_cam, axis=0)
            episode['tcp_pose'] = np.stack(tcp_pose, axis=0)
            episode['joint_pos'] = np.stack(joint_pos, axis=0)
            episode['action'] = np.stack(action, axis=0)
            
            # ================ Prevent Gimbal Lock problem of Euler angles =================
            concat_tcp_r = np.concatenate((np.expand_dims(CANONICAL_EULER_ANGLES, axis=0), episode['tcp_pose'][:, 3:6]), axis=0)
            unwrapped_tcp_r = np.unwrap(concat_tcp_r, axis=0)
            episode['tcp_pose'] = np.concatenate((episode['tcp_pose'][:, :3], unwrapped_tcp_r[1:, :]), axis=-1)
            # ================ Prevent Gimbal Lock problem of Euler angles =================

            # Create a group for this episode
            episode_group = f.create_group(f'episode_{episode_id}')
            # Save all key-value pairs for this episode
            for key, value in episode.items():
                # Convert to numpy array if needed
                if not isinstance(value, np.ndarray):
                    value = np.array(value)
                
                # Create dataset in the episode group
                episode_group.create_dataset(key, data=value, compression='gzip')
                print(f"Saved {key} with shape {value.shape}")
            print('Saved episode ', episode_id)
            episode_id += 1
            robot_env.gripper.move(robot_env.gripper.max_width)
            time.sleep(0.5)
            if not robot_env.keyboard.quit:
                print("reset the environment...")
                time.sleep(10)


if __name__ == '__main__':
    main(tyro.cli(Args))
    