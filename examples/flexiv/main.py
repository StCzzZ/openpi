import dataclasses
import logging
import tyro
import numpy as np
from typing import Union, Tuple
import pathlib
import time
import collections
import h5py
import os

from hardware.robot_env import RobotEnv
from hardware.my_device.macros import CAM_SERIAL, HUMAN, ROBOT, CANONICAL_EULER_ANGLES
from openpi_client import websocket_client_policy as _websocket_client_policy
from reward_model.util import save_episode_reward_video

KEY_MAPPING = {
    'wrist_cam': 'wrist_img',
    'side_cam': 'side_img',
    'tcp_pose': 'tcp_pose',
    'joint_pos': 'joint_pos',
    'action': 'action',
    'action_mode': 'action_mode'
}

def init_robot_env(img_shape: tuple[int, int, int], fps: float) -> RobotEnv:
    return RobotEnv(
        camera_serial=CAM_SERIAL, 
        img_shape=img_shape, 
        fps=fps
    )

@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: Union[int, Tuple[int, int]] = (224, 224)
    replan_steps: int = 10

    #################################################################################################################
    # Flexiv environment-specific parameters
    #################################################################################################################
    fps: float = 10.0
    random_init: bool = True
    max_steps: int = 600
    num_rollouts: int = 20

    #################################################################################################################
    # Utils
    #################################################################################################################
    output_dir: str = "data/flexiv/rollout_data"  # Path to save rollout data
    output_name: str = "1201_kitchen_100_value"
    seed: int = 7  # Random Seed (for reproducibility)


def main(args: Args) -> None:
    """Main entry point"""
    # set random seed
    np.random.seed(args.seed)

    # Task description
    task_description = "Put the items in the pot."
    
    # Initialize robot environment
    if isinstance(args.resize_size, int):
        robot_env_img_shape = (3, args.resize_size, args.resize_size)
    else:
        robot_env_img_shape = (3, *args.resize_size)
    robot_env = init_robot_env(img_shape=robot_env_img_shape, fps=args.fps)

    # Setup output file
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.output_name}.hdf5")
    episode_id = 0
    if os.path.exists(output_path):
        with h5py.File(output_path, 'r') as f:
            # Find the highest episode_id in the existing file
            episode_keys = [key for key in f.keys() if key.startswith('episode_')]
            if episode_keys:
                episode_ids = [int(key.split('_')[1]) for key in episode_keys]
                episode_id = max(episode_ids) + 1
                print(f"Appending to existing file. Starting from episode {episode_id}")

    # Run rollout
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    for i in range(args.num_rollouts):
        if robot_env.keyboard.quit:
            break

        # Reset keyboard states
        robot_env.keyboard.finish = False
        robot_env.keyboard.help = False
        robot_env.keyboard.infer = False
        robot_env.keyboard.discard = False

        # Initialize episode buffers
        episode_buffers = {
            'tcp_pose': [],
            'joint_pos': [],
            'action': [],
            'action_mode': [],
            'wrist_cam': [],
            'side_cam': []
        }

        logging.info(f"Running rollout {i+1}/{args.num_rollouts}")
        logging.info(f"Task description: {task_description}")

         # Reset robot
        random_init_pose = None
        if args.random_init:
            random_init_pose = robot_env.robot.init_pose + np.random.uniform(-0.1, 0.1, size=7)
            random_init_pose[2] = max(random_init_pose[2], 0.15)
        
        robot_state = robot_env.reset_robot(args.random_init, random_init_pose)
        time.sleep(10)
        action_plan = collections.deque()
        tcp_rot_history = [CANONICAL_EULER_ANGLES] # to prevent gimbal lock problem of euler angles in the observation space
        t = 0

        while t < args.max_steps:
            if robot_env.keyboard.finish:
                # Finalize episode
                for key in episode_buffers.keys():
                    episode_buffers[key] = np.stack(episode_buffers[key], axis=0)
                # Save to file
                with h5py.File(output_path, 'a') as f:
                    episode_group = f.create_group(f'episode_{episode_id}')
                    for key, value in episode_buffers.items():
                        episode_group.create_dataset(key, data=value)
                    episode_id += 1
                logging.info(f"Rollout {i+1}/{args.num_rollouts} finished")
                logging.info(f"Episode {episode_id} saved")
                # Save reward video
                if "value" in episode_buffers:
                    save_episode_reward_video(
                        episode_name=f"episode_{episode_id}",
                        frames=episode_buffers['side_cam'],
                        rewards=episode_buffers['value'],
                        output_path=pathlib.Path(os.path.join(args.output_dir, args.output_name, f"episode_{episode_id}_reward.mp4")),
                        fps=args.fps,
                    )
                    logging.info(f"Reward video saved to {os.path.join(args.output_dir, args.output_name, f'episode_{episode_id}_reward.mp4')}")
                break

            if robot_env.keyboard.discard:
                logging.info(f"Rollout {i+1}/{args.num_rollouts} discarded")
                break

            if not robot_env.keyboard.help:
                start_time = time.time()

                # Get observations
                robot_state = robot_env.get_robot_state()
                standard_tcp_rot = np.unwrap(np.stack((tcp_rot_history[-1], robot_state['tcp_pose'][3:6]), axis=0), axis=0)[1, :]
                tcp_rot_history.append(standard_tcp_rot) # Prevent gimbal lock
                if not action_plan:
                    element = {
                        "observation/image": robot_state['side_img'],
                        "observation/wrist_image": robot_state['wrist_img'],
                        "observation/state": np.concatenate((robot_state['tcp_pose'][:3], standard_tcp_rot), axis=0),
                        "prompt": task_description,
                    }
                    model_output = client.infer(element)
                    action_chunk = model_output["actions"]
                    # Save predicted value to buffer
                    if "value" in model_output:
                        value = model_output["value"]
                        if "value" not in episode_buffers:
                            episode_buffers["value"] = []
                        episode_buffers["value"].append(value)
                    assert(
                        len(action_chunk) >= args.replan_steps
                    ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                    action_plan.extend(action_chunk[:args.replan_steps])
                # Execute action
                action = action_plan.popleft()
                robot_env.deploy_action(action[:6], action[6])
                # Save to buffer
                episode_buffers['wrist_cam'].append(robot_state['wrist_img'])
                episode_buffers['side_cam'].append(robot_state['side_img'])
                episode_buffers['tcp_pose'].append(robot_state['tcp_pose'])
                episode_buffers['joint_pos'].append(robot_state['joint_pos'])
                episode_buffers['action'].append(action)
                episode_buffers['action_mode'].append(ROBOT)

                # Sleep to maintain the desired fps
                time.sleep(max(1 / args.fps - (time.time() - start_time), 0))
                t += 1
            else:
                if len(action_plan):
                    action_plan.clear()
                teleop_data = robot_env.human_teleop_step()
                if teleop_data is None:
                    continue
                
                standard_tcp_rot = np.unwrap(np.stack((tcp_rot_history[-1], teleop_data['tcp_pose'][3:6]), axis=0), axis=0)[1, :]
                tcp_rot_history.append(standard_tcp_rot) # Prevent gimbal lock
                teleop_data['tcp_pose'] = np.concatenate((teleop_data['tcp_pose'][:3], standard_tcp_rot), axis=0)
                # Save to buffer
                for buffer_key, teleop_data_key in KEY_MAPPING.items():
                    episode_buffers[buffer_key].append(teleop_data[teleop_data_key])
                # Reset keyboard states
                if robot_env.keyboard.infer:
                    robot_env.keyboard.help = False
                    robot_env.keyboard.infer = False
                
                t += 1

        logging.info(f"Rollout {i+1}/{args.num_rollouts} timeout")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))