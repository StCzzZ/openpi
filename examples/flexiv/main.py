import dataclasses
import logging
import tyro
import numpy as np
from typing import Union, Tuple
import pathlib
import time
import collections

from hardware.robot_env import RobotEnv
from hardware.my_device.macros import CAM_SERIAL, HUMAN, ROBOT
from openpi_client import websocket_client_policy as _websocket_client_policy

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
    num_rollouts: int = 10

    #################################################################################################################
    # Utils
    #################################################################################################################
    output_dir: str = "data/flexiv/rollout_data"  # Path to save rollout data

    seed: int = 7  # Random Seed (for reproducibility)


def main(args: Args) -> None:
    """Main entry point"""
    # set random seed
    np.random.seed(args.seed)

    # Task description
    task_description = "Fold the towel twice by grabbing its corners"
    
    # Initialize robot environment
    if isinstance(args.resize_size, int):
        robot_env_img_shape = (3, args.resize_size, args.resize_size)
    else:
        robot_env_img_shape = (3, *args.resize_size)
    robot_env = init_robot_env(img_shape=robot_env_img_shape, fps=args.fps)

    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Run rollout
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    for i in range(args.num_rollouts):
        logging.info(f"Running rollout {i+1}/{args.num_rollouts}")
        logging.info(f"Task description: {task_description}")

         # Reset robot
        random_init_pose = None
        if args.random_init:
            random_init_pose = robot_env.robot.init_pose + np.random.uniform(-0.1, 0.1, size=7)
        
        robot_state = robot_env.reset_robot(args.random_init, random_init_pose)
        action_plan = collections.deque()

        t = 0
        while t < args.max_steps:
            start_time = time.time()

            # Get observations
            robot_state = robot_env.get_robot_state()
            if not action_plan:
                element = {
                    "observation/image": robot_state['side_img'].permute(1, 2, 0).cpu().numpy().astype(np.uint8),
                    "observation/wrist_image": robot_state['wrist_img'].permute(1, 2, 0).cpu().numpy().astype(np.uint8),
                    "observation/state": robot_state['tcp_pose'],
                    "prompt": task_description,
                }
                action_chunk = client.infer(element)["actions"]
                assert(
                    len(action_chunk) >= args.replan_steps
                ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                action_plan.extend(action_chunk[:args.replan_steps])
            
            action = action_plan.popleft()
            robot_env.deploy_action(action[:6], action[6])
            # Sleep to maintain the desired fps
            time.sleep(max(1 / args.fps - (time.time() - start_time), 0))
            t += 1

        logging.info(f"Rollout {i+1}/{args.num_rollouts} completed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(main)