# import os
# os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''  # 禁用OpenCV的Qt插件
# os.environ['QT_QPA_PLATFORM'] = 'xcb'  # 强制使用系统X11
# os.environ['MPLBACKEND'] = 'Qt5Agg'  # 强制matplotlib使用PyQt5

import matplotlib
matplotlib.use('Agg')  # 或 'GTK3Agg'
import matplotlib.pyplot as plt
import cv2
import collections
import dataclasses
import logging
import math
import pathlib

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import socket
import threading
import queue
import tqdm
import tyro

# ignore warnings
import warnings
warnings.filterwarnings("ignore")
from termcolor import cprint
import json
import ast
import time
import h5py  # uv pip linstall h5py==3.7.0 to match numpy 1.22.4 TODO

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

def _client_handler(conn, addr, q, lock, stop_evt):
    conn.settimeout(1.0)
    buffer = b""
    try:
        while not stop_evt.is_set():
            try:
                data = conn.recv(4096)
                # cprint(f"data received: {data}", "yellow")
            except socket.timeout:
                cprint(f"socket timeout from {addr}, continuing...", "yellow")
                continue
            if not data:
                cprint(f"no data recieved.", "red")
                continue
            buffer += data
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                text = line.decode().strip()
                if not text:
                    continue
                if text.lower() == "stop":
                    cprint(f"Received STOP from {addr}", "yellow")
                    stop_evt.set()
                    return
                # try JSON first, fallback to python literal
                parsed = None
                try:
                    parsed = json.loads(text)
                except Exception:
                    try:
                        parsed = ast.literal_eval(text)
                    except Exception:
                        parsed = None
                if parsed is None:
                    cprint("Ignoring HTTP/WebSocket handshake header", "yellow")
                    continue
                if isinstance(parsed, list) and len(parsed) == len(LIBERO_DUMMY_ACTION):
                    try:
                        arr = [float(x) for x in parsed]
                        with lock:
                            if q.full():
                                q.get()
                            q.put(arr)
                    except Exception:
                        cprint(f"Received data but failed to convert to floats: {parsed}", "red")
                else:
                    pass
            time.sleep(0.01) # prevent busy loop
    finally:
        try:
            conn.close()
        except Exception:
            pass

def _server_thread(host, port, q, lock, stop_evt):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    srv.settimeout(1.0)
    cprint(f"Manual-control server listening on {host}:{port}", "green")
    try:
        while not stop_evt.is_set():
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue
            cprint(f"Manual-control client connected from {addr}", "green")
            _client_handler(conn, addr, q, lock, stop_evt)
            cprint(f"Manual-control client from {addr} disconnected", "yellow")
            if stop_evt.is_set():
                break
    finally:
        try:
            srv.close()
        except Exception:
            pass
    cprint("Manual-control server stopped", "yellow")

@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8001
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 5  # Number of rollouts per task
    num_envs: int = 10  # Number of environments to run in parallel

    #################################################################################################################
    # SpaceMouse parameters
    #################################################################################################################
    spm_host = "0.0.0.0"
    spm_port = 18002
    stop_str = "STOP"  # String to signal stopping manual control
    hil_fps: int = 2  # Frequency to poll SpaceMouse for actions during human-in-the-loop control

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos
    seed: int = 7  # Random Seed (for reproducibility)
    data_record_path: str = "data/libero/h5py_records"  # Path to save h5py records

def eval_libero_parallel(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id) # initial_states: Optional[Union[int, List[int], np.ndarray]]

        # Initialize LIBERO environment and task description
        env, task_description = _get_parallel_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed, args.num_envs)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            indices = (np.arange(args.num_envs) + episode_idx) % initial_states.shape[0]
            obs = env.set_init_state(initial_states[indices])  # use a "_get_observations()" function built-in the env to get obs after setting initial state

            cprint(f"indices: {indices}", "blue")  # [1~19]
            cprint(f"type(initial_states): {type(initial_states)}", "blue")   #  <class 'numpy.ndarray'>
            cprint(f"initial_states.shape: {initial_states.shape}", "blue")   # initial_states.shape: (50, 77)

            # Setup
            t = 0
            replay_images = []
            dones = np.array([False] * args.num_envs)

            # IMG visualization
            policy_rollout_window_name = "policy rollout view"
            policy_rollout_wrist_window_name = "policy rollout wrist view"
            cv2.namedWindow(policy_rollout_window_name, cv2.WINDOW_NORMAL)
            cv2.namedWindow(policy_rollout_wrist_window_name, cv2.WINDOW_NORMAL)

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                # try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                if t < args.num_steps_wait:
                    dummy_actions = np.array([LIBERO_DUMMY_ACTION] * args.num_envs)
                    obs, reward, done, info = env.step(dummy_actions)
                    t += 1
                    continue

                # Get preprocessed image
                # IMPORTANT: rotate 180 degrees to match train preprocessing
                img = np.stack([np.ascontiguousarray(x["agentview_image"][::-1, ::-1]) for x in obs], axis=0)
                wrist_img = np.stack([np.ascontiguousarray(x["robot0_eye_in_hand_image"][::-1, ::-1]) for x in obs], axis=0)
                img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                )
                wrist_img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                )

                # Visualize img[0] using opencv
                cv2.imshow(policy_rollout_window_name, cv2.cvtColor(img[0], cv2.COLOR_RGB2BGR))
                cv2.imshow(policy_rollout_wrist_window_name, cv2.cvtColor(wrist_img[0], cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

                # print the current step
                cprint(f"Current step: {t}, dones: {dones}", "yellow")
                # Save preprocessed image for replay video
                replay_images.append(img[0])

                if not action_plan:
                # Finished executing previous action chunk -- compute new chunk
                # Prepare observations dict
                    element = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": np.stack([np.concatenate(
                            (
                                x["robot0_eef_pos"],
                                _quat2axisangle(x["robot0_eef_quat"]),
                                x["robot0_gripper_qpos"],
                            )
                        ) for x in obs], axis=0),
                        "prompt": np.array([str(task_description) for _ in range(args.num_envs)]),
                    }

                    # Query model to get action
                    action_chunk = client.infer(element)["actions"]
                    assert (
                        len(action_chunk[0]) >= args.replan_steps
                    ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk[0])} steps."
                    action_plan.extend(action_chunk[:, :args.replan_steps].transpose(1,0,2))

                action = action_plan.popleft() # action.shape: (num_envs, 7) -> (6dof+openness)

                # Execute action in environment
                obs, reward, done, info = env.step(action)  # type(obs): <class 'numpy.ndarray'>, obs.shape: (20,), so does reward, done, info

                # get the current state
                # sim_state_now = env.get_sim_state()
                # cprint(f"type(sim_state_now): {type(sim_state_now)}", "yellow") # list
                # sim_state_now_array = np.array(sim_state_now)
                # cprint(f"sim_state_now_array.shape: {sim_state_now_array.shape}", "yellow") # (20, 77), TODO: check what the 77-dim stands for, or just leave it, record and reset the state when Armada feels wrong

                dones = dones | done
                if dones.all():
                    break
                t += 1

            cv2.destroyWindow(policy_rollout_window_name)
            cv2.destroyWindow(policy_rollout_wrist_window_name)
            # failure detection
            # sry, raise some failure cases here, just for debugging
            dones[-3:-1] = 0

            not_dones_indices = np.where(dones == False)[0]
            cprint(f"not_dones_indices: {not_dones_indices}", "red")
            if len(not_dones_indices) > 0:
                cprint(f"{len(not_dones_indices)} envs failed, failed envs: {not_dones_indices}", "red")

                for idx in not_dones_indices:
                    human_correct_step = 0
                    ans = input(f"Env {idx} failed. Intervene and reset to init state? (y/n): ").strip().lower()
                    if ans != "y":
                        continue
                    # Reset all envs and set the initial states (keeps behavior consistent with earlier reset+set_init)
                    env.reset()
                    initial_state_idx = np.expand_dims(initial_states[idx], axis=0)
                    cprint(f"initial_state_idx.shape: {initial_state_idx.shape}", "blue")  # (1, 77)
                    obs_idx = env.set_init_state(initial_state_idx, id=[idx])

                    # queue to receive actions from network thread
                    action_queue = queue.Queue(maxsize=1) # always use the newest action
                    stop_event = threading.Event()
                    lock_spacemouse = threading.Lock()

                    # start server thread
                    server_thread = threading.Thread(target=_server_thread, args=(args.spm_host, args.spm_port, action_queue, lock_spacemouse, stop_event), daemon=True)
                    server_thread.start()

                    hil_window_name = f"HIL Control Env {idx}"
                    hil_wrist_window_name = f"HIL Control Wrist Env {idx}"
                    win_w = int(args.resize_size * 2)
                    win_h = int(args.resize_size * 2)
                    cv2.namedWindow(hil_wrist_window_name, cv2.WINDOW_NORMAL)
                    cv2.namedWindow(hil_window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(hil_window_name, win_w, win_h)
                    cv2.resizeWindow(hil_wrist_window_name, win_w, win_h)

                    waiting_cnt = 1
                    while not stop_event.is_set(): # use stop_event.set() to stop the loop
                        try:
                            action = action_queue.get(timeout=0.4)
                            cprint(f"action: {action}", "blue")
                        except queue.Empty:
                            # cprint(f"failed to get action from queue, continue waiting...", "yellow")
                            waiting_cnt += 1
                            if waiting_cnt % 50 == 0:
                                cprint(f"Waiting for action for env {idx}...", "yellow")
                                waiting_cnt = 1
                            continue

                        # validate and create action_arr for single env
                        try:
                            if not (isinstance(action, (list, tuple, np.ndarray)) and action != args.stop_str):
                                raise ValueError("action must be a list-like")
                            if len(action) != len(LIBERO_DUMMY_ACTION):
                                raise ValueError(f"action length must be {len(LIBERO_DUMMY_ACTION)}")
                            action_arr = np.array([action], dtype=float)
                        except Exception as e:
                            cprint(f"Ignoring invalid action from client: {e}", "red")
                            continue
                        # perform the single-env step by specifying id
                        human_correct_step += 1
                        obs_idx, reward, done, info = env.step(action_arr, id=[idx])
                        cprint(f"done: {done}, human_correct_step: {human_correct_step}", "green")

                        # update the preview window and wrist view for this env
                        hil_img = np.ascontiguousarray(obs_idx[0]["agentview_image"][::-1, ::-1])
                        hil_img = image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(hil_img[None, ...], args.resize_size, args.resize_size)
                        )[0]
                        hil_wrist_img = np.ascontiguousarray(obs_idx[0]["robot0_eye_in_hand_image"][::-1, ::-1])
                        hil_wrist_img = image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(hil_wrist_img[None, ...], args.resize_size, args.resize_size)
                        )[0]

                        cv2.imshow(hil_window_name, cv2.cvtColor(hil_img, cv2.COLOR_RGB2BGR))
                        cv2.imshow(hil_wrist_window_name, cv2.cvtColor(hil_wrist_img, cv2.COLOR_RGB2BGR))
                        cv2.waitKey(1)

                        if done[0]:
                            cprint(f"Env {idx} finished (done).", "cyan")
                        
                        # sleep to match HIL fps
                        time.sleep(1.0 / args.hil_fps)
                        
                    # destroy cv2 window and stop server thread
                    cv2.destroyWindow(hil_window_name)
                    cv2.destroyWindow(hil_wrist_window_name)
                    stop_event.set()
                    # stopping the event loop is handled by thread exit when client disconnects
                    # but we'll attempt to join thread
                    server_thread.join(timeout=1.0)

            task_successes += np.sum(dones)
            total_successes += np.sum(dones)

            task_episodes += args.num_envs
            total_episodes += args.num_envs

            # Save a replay video of the episode
            suffix = "success" if dones[0] else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_parallel_libero_env(task, resolution, seed, num_envs):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = SubprocVectorEnv(
            [lambda: OffScreenRenderEnv(**env_args) for _ in range(num_envs)]
        )
    env.seed(np.arange(num_envs))  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero_parallel)