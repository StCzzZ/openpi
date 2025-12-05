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
from typing import Dict, List, Optional
import zarr  # uv pip linstall h5py==3.7.0 to match numpy 1.22.4 TODO
import copy

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

        #     task_successes += np.sum(dones)
        #     total_successes += np.sum(dones)

        #     task_episodes += args.num_envs
        #     total_episodes += args.num_envs

        #     # Save a replay video of the episode
        #     suffix = "success" if dones[0] else "failure"
        #     task_segment = task_description.replace(" ", "_")
        #     imageio.mimwrite(
        #         pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
        #         [np.asarray(x) for x in replay_images],
        #         fps=10,
        #     )

        #     # Log current results
        #     logging.info(f"Success: {done}")
        #     logging.info(f"# episodes completed so far: {total_episodes}")
        #     logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        #     episode_idx += 1

        

    # # Log final results
    # logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
    # logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    # logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    # logging.info(f"Total episodes: {total_episodes}")


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


def _space_mouse_client_handler(conn, addr, q, lock, stop_evt):
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

def _space_mouse_server_thread(host, port, q, lock, stop_evt):
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
            _space_mouse_client_handler(conn, addr, q, lock, stop_evt)
            cprint(f"Manual-control client from {addr} disconnected", "yellow")
            if stop_evt.is_set():
                break
    finally:
        try:
            srv.close()
        except Exception:
            pass
    cprint("Manual-control server stopped", "yellow")



    


class LiberoRunner:

    def __init__(self, args):
        self.args = args

        # Set random seed
        np.random.seed(self.args.seed)

        # Initialize LIBERO task suite
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[self.args.task_suite_name]()
        num_tasks_in_suite = task_suite.n_tasks
        logging.info(f"Task suite: {self.args.task_suite_name}")

        pathlib.Path(self.args.video_out_path).mkdir(parents=True, exist_ok=True)
        if self.args.task_suite_name == "libero_spatial":
            self.max_steps = 220  # longest training demo has 193 steps
        elif self.args.task_suite_name == "libero_object":
            self.max_steps = 280  # longest training demo has 254 steps
        elif self.args.task_suite_name == "libero_goal":
            self.max_steps = 300  # longest training demo has 270 steps
        elif self.args.task_suite_name == "libero_10":
            self.max_steps = 520  # longest training demo has 505 steps
        elif self.args.task_suite_name == "libero_90":
            self.max_steps = 400  # longest training demo has 373 steps
        else:
            raise ValueError(f"Unknown task suite: {self.args.task_suite_name}")

        self.client = _websocket_client_policy.WebsocketClientPolicy(self.args.host, self.args.port)

        # Start evaluation, evaluate for a single task for now
        task_id = 3
        task = task_suite.get_task(task_id)

        # IMG visualization windows
        self.policy_rollout_window_name = "policy rollout view"
        self.policy_rollout_wrist_window_name = "policy rollout wrist view"
        # cv2.namedWindow(self.policy_rollout_window_name, cv2.WINDOW_NORMAL)
        # cv2.namedWindow(self.policy_rollout_wrist_window_name, cv2.WINDOW_NORMAL)

        # Get default LIBERO initial states, Initialize environment and task description
        self.initial_states = task_suite.get_task_init_states(task_id) # initial_states: np.ndarray], shape: (num_envs, state_dim)
        self.env, self.task_description = _get_parallel_libero_env(task, LIBERO_ENV_RESOLUTION, self.args.seed, self.args.num_envs)
        # HIL_env to avoid env step conflict. 
        self.HIL_env, _ = _get_parallel_libero_env(task, LIBERO_ENV_RESOLUTION, self.args.seed, 1)
        logging.info(f"\nTask: {self.task_description}")

        # Initializations
        self.env.reset()
        self.HIL_env.reset()
        self.action_plan = collections.deque()
        # The initial states cycle through all available states
        self.init_state_indices = (np.arange(self.args.num_envs)) % self.initial_states.shape[0]
        obs = self.env.set_init_state(self.initial_states[self.init_state_indices])  # use env._get_observations() to get obs after setting initial state, initial_states.shape: (num_envs, num_obs)
        self.dones = np.array([False] * self.args.num_envs)
        state_init = np.stack([np.concatenate(( x["robot0_eef_pos"],_quat2axisangle(x["robot0_eef_quat"]), x["robot0_gripper_qpos"],)) for x in obs], axis=0)

        # list of indices to indicate the envs that need hil, also some indicators initialization
        self.hil_indicator = np.array([False] * self.args.num_envs)
        self.hil_envs_indices_set = set()
        self.running_env_indices = np.where(self.hil_indicator == False)[0]

        # success and timestep trackers initialization
        self.success_episodes = np.array([0] * self.args.num_envs, dtype=int)
        self.failure_episodes = np.array([0] * self.args.num_envs, dtype=int)
        self.env_timesteps = np.array([0] * self.args.num_envs, dtype=int)
        self.env_running_timesteps = self.env_timesteps[self.running_env_indices]

        # Failure detection initialization manually set a "OOD" state for debugging, will be replaced by ARMADA later
        self.OOD_state_rate = np.array([0.005]*self.args.num_envs)  # 1% OOD states
        # self.OOD_state_rate = np.array([0.0]*self.args.num_envs)  # 0% OOD states
        
        # action, state, img buffers for all envs
        self.action_buffer = np.zeros((self.args.max_buffer_size, self.args.num_envs, len(LIBERO_DUMMY_ACTION)), dtype=float)
        self.input_states_buffer = np.zeros((self.args.max_buffer_size, self.args.num_envs, state_init.shape[1]), dtype=float)
        self.img_buffer = np.zeros((self.args.max_buffer_size, self.args.num_envs, self.args.resize_size, self.args.resize_size, 3), dtype=np.uint8)
        self.wrist_img_buffer = np.zeros((self.args.max_buffer_size, self.args.num_envs, self.args.resize_size, self.args.resize_size, 3), dtype=np.uint8)

        # add a global obs_manager to manage the obs for all envs in HIL and  NON-HIL control
        self.obs_manager = copy.deepcopy(obs)
        # for state traceback, save all states for now, may change to a windowed buffer later
        self.state_record_buffer = np.zeros((self.args.max_buffer_size, self.args.num_envs, self.initial_states.shape[1]), dtype=float)
        
        ###### Variables for spacemouse server manager ######
        self.listening_stop_evt = threading.Event()
        self.HIL_stop_evt = threading.Event()
        self._listening_thread = threading.Thread(target=self._listening_server_loop, daemon=True)
        self._HIL_thread = threading.Thread(target=self._HIL_loop, daemon=True)
        self.lock = threading.Lock()
        self.env_lock = threading.Lock()
        self.action_queue = queue.Queue(maxsize=1)
        self.HIL_active = False
        self.hil_indices = None
        self.HIL_id = None
        self.HIL_window_name = f"HIL Control Env"
        self.HIL_wrist_window_name = f"HIL Control Wrist Env"
        self.HIL_window_active = False

        self.HIL_counterfeit_id = np.array([0], dtype=int)  # workaround. 
        self.counterfeit_env_lock = threading.Lock()

    def eval_libero_parallel(self) -> None:

        # start the HIL server thread
        self.start_listening_thread()
        self.start_HIL_thread()

        while True:
            try:
                
                if len(self.running_env_indices) == 0:
                    # cprint("All envs are under HIL control, skipping policy rollout.", "yellow")
                    time.sleep(1.0)
                    continue

                ########## Recording data ##########
                # Get preprocessed image and Visualize img[0], IMPORTANT: rotate 180 degrees to match train preprocessing
                img = np.stack([np.ascontiguousarray(x["agentview_image"][::-1, ::-1]) for x in self.obs_manager], axis=0)
                wrist_img = np.stack([np.ascontiguousarray(x["robot0_eye_in_hand_image"][::-1, ::-1]) for x in self.obs_manager], axis=0)
                img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, self.args.resize_size, self.args.resize_size))
                wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, self.args.resize_size, self.args.resize_size))
                # cv2.imshow(self.policy_rollout_window_name, cv2.cvtColor(img[0], cv2.COLOR_RGB2BGR))
                # cv2.imshow(self.policy_rollout_wrist_window_name, cv2.cvtColor(wrist_img[0], cv2.COLOR_RGB2BGR))
                # # cv2.imshow(self.policy_rollout_window_name_2, cv2.cvtColor(img[2], cv2.COLOR_RGB2BGR))
                # # cv2.imshow(self.policy_rollout_wrist_window_name_2, cv2.cvtColor(wrist_img[2], cv2.COLOR_RGB2BGR))
                # cv2.waitKey(1)
                # get the current state and record, env.get_sim_state() returns all states in a np.ndarray, we use the state for the model input, compact the state manually
                sim_state_arr = np.stack([np.concatenate(( x["robot0_eef_pos"],_quat2axisangle(x["robot0_eef_quat"]), x["robot0_gripper_qpos"],)) for x in self.obs_manager], axis=0) # shape: (num_envs, 3+4+1)
                with self.env_lock:
                    record_state_arr = np.array(self.env.get_sim_state())

                cprint(f"self.env_running_timesteps: {self.env_running_timesteps}; self.running_env_indices: {self.running_env_indices}", "blue")
                with self.lock:
                    self.img_buffer[self.env_running_timesteps, self.running_env_indices] = img[self.running_env_indices]
                    self.wrist_img_buffer[self.env_running_timesteps, self.running_env_indices] = wrist_img[self.running_env_indices]
                    self.input_states_buffer[self.env_running_timesteps, self.running_env_indices] = sim_state_arr[self.running_env_indices]
                    self.state_record_buffer[self.env_running_timesteps, self.running_env_indices] = record_state_arr[self.running_env_indices]
                ########## End Recording data ##########
                
                ########## inference and action execution ##########
                if not self.action_plan:
                # Finished executing previous action chunk -- compute new chunk
                # Prepare observations dict
                    element = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": np.stack([np.concatenate(( x["robot0_eef_pos"],_quat2axisangle(x["robot0_eef_quat"]), x["robot0_gripper_qpos"],)) for x in self.obs_manager], axis=0),
                        "prompt": np.array([str(self.task_description) for _ in range(self.args.num_envs)]),
                    }

                    # Query model to get action
                    action_chunk = self.client.infer(element)["actions"]
                    assert (
                        len(action_chunk[0]) >= self.args.replan_steps
                    ), f"We want to replan every {self.args.replan_steps} steps, but policy only predicts {len(action_chunk[0])} steps."
                    self.action_plan.extend(action_chunk[:, :self.args.replan_steps].transpose(1,0,2))

                action = self.action_plan.popleft() # shape: (num_envs, 7) -> (6dof+openness)
                with self.lock:
                    self.action_buffer[self.env_running_timesteps, self.running_env_indices] = action[self.running_env_indices]

                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects and we need to wait for them to fall
                action_editable = action.copy()  # the original action is read-only
                t_waiting = np.where(self.env_timesteps < self.args.num_steps_wait)[0]
                action_editable[t_waiting] = LIBERO_DUMMY_ACTION

                # Execute action in environment, only stepping the non-HIL envs
                with self.env_lock:
                    obs, reward, done, info = self.env.step(action_editable[self.running_env_indices], id=self.running_env_indices.tolist())  # type of obs, reward, done, info: <class 'numpy.ndarray'>, shape: (running_envs,); obs is an array of dicts
                with self.lock:
                    self.env_timesteps[self.running_env_indices] += 1
                    self.running_env_indices = np.where(self.hil_indicator == False)[0]
                    self.env_running_timesteps = self.env_timesteps[self.running_env_indices]
                    # update the obs_manager for next policy rollout
                    self.obs_manager[self.running_env_indices] = obs
                    self.dones[self.running_env_indices] = self.dones[self.running_env_indices] | done
                    
                    
                ########## end inference and action execution ##########

                ########## OOD detection: randomly for now ##########
                random_vals = np.random.rand(self.args.num_envs)
                OOD_env_indices = np.where(random_vals < self.OOD_state_rate)[0]
                # keep env_3 not OOD for debugging
                # OOD_env_indices = OOD_env_indices[OOD_env_indices != 3]
                if len(OOD_env_indices) > 0:
                    cprint(f"Env {OOD_env_indices} with random val {random_vals[OOD_env_indices]} entered OOD state at step {self.env_timesteps[OOD_env_indices]}, switching to HIL control.", "red")
                    self.hil_indicator[OOD_env_indices] = True
                    self.hil_envs_indices_set.update(OOD_env_indices.tolist())
                    self.running_env_indices = np.where(self.hil_indicator == False)[0]
                    self.env_running_timesteps = self.env_timesteps[self.running_env_indices]
                    self.dones[OOD_env_indices] = False  # set done flag fail for OOD envs

                    cprint(f"hil_indicator: {self.hil_indicator}, hil_envs_indices_set: {self.hil_envs_indices_set}, dones: {self.dones}", "yellow")

                ########## end OOD detection ##########

                ########## Timeout detection ##########
                timeout_env_indices = np.where(self.env_timesteps >= self.max_steps)[0]
                if len(timeout_env_indices) > 0:
                    cprint(f"Env {timeout_env_indices} reached max steps {self.max_steps}, resetting env and switching to next initial state.", "red")
                    self.failure_episodes[timeout_env_indices] += 1
                    
                    with self.lock:
                        self.init_state_indices[timeout_env_indices] = (self.init_state_indices[timeout_env_indices] + 1) % self.initial_states.shape[0]
                        timeout_init_states = self.initial_states[self.init_state_indices[timeout_env_indices]]
                    with self.env_lock:
                        self.env.reset(id=timeout_env_indices.tolist())
                        obs_timeout = self.env.set_init_state(timeout_init_states, id=timeout_env_indices.tolist())
                        cprint(f"timeout_init_states.shape: {timeout_init_states.shape}, timeout_env_indices.tolist(): {timeout_env_indices.tolist()}", "magenta")

                    with self.lock:
                        # update the obs_manager
                        self.obs_manager[timeout_env_indices] = obs_timeout
                        self.env_timesteps[timeout_env_indices] = 0
                        self.running_env_indices = np.where(self.hil_indicator == False)[0]
                        self.env_running_timesteps = self.env_timesteps[self.running_env_indices]
                        self.dones[timeout_env_indices] = False  # reset done flag for timed-out envs
                        
                ########## success detection, if success, reset the env and record the corresponding data ##########
                success_env_indices = np.where(self.dones == True)[0]
                #### Since FLOAT is for "FAILURE" detection, we don't consider that a env is both successful and OOD here, we can add a logic to prevent a success env from being marked as OOD later if needed.
                if len(success_env_indices) > 0:
                    cprint(f"Env {success_env_indices} succeeded at step {self.env_timesteps[success_env_indices]}.", "green")
                    
                    cprint(f"success_env_indices: {success_env_indices}", "green")
                    with self.lock:
                        self.init_state_indices[success_env_indices] = (self.init_state_indices[success_env_indices] + 1) % self.initial_states.shape[0]
                        success_init_states = self.initial_states[self.init_state_indices[success_env_indices]]
                    with self.env_lock:
                        self.env.reset(id=success_env_indices.tolist())
                        obs_success = self.env.set_init_state(success_init_states, id=success_env_indices.tolist())
                        cprint(f"success_init_states.shape: {success_init_states.shape}, success_env_indices.tolist(): {success_env_indices.tolist()}", "blue")

                    with self.lock:
                        # update the obs_manager
                        self.obs_manager[success_env_indices] = obs_success
                        self.env_timesteps[success_env_indices] = 0
                        self.running_env_indices = np.where(self.hil_indicator == False)[0]
                        self.env_running_timesteps = self.env_timesteps[self.running_env_indices]
                        self.dones[success_env_indices] = False  # reset done flag for succeeded envs
                        self.success_episodes[success_env_indices] += 1

                ########## end success detection ##########

                if self.dones.all():
                    break

                
            except KeyboardInterrupt:
                cprint("KeyboardInterrupt detected, exiting evaluation loop.", "red")
                break

        # cv2.destroyWindow(self.policy_rollout_window_name)
        # cv2.destroyWindow(self.policy_rollout_wrist_window_name)
        # cv2.destroyWindow(self.policy_rollout_window_name_2)
        # cv2.destroyWindow(self.policy_rollout_wrist_window_name_2)

    def start_listening_thread(self):
        """Start the background server thread."""
        if not self._listening_thread.is_alive():
            cprint(f"starting listening thread...", "green")
            self._listening_thread.start()

    def start_HIL_thread(self):
        """Start the background HIL thread."""
        if not self._HIL_thread.is_alive():
            cprint(f"starting HIL thread...", "green")
            self._HIL_thread.start()

    def stop_threads(self, join_timeout: Optional[float] = 1.0):
        """Request stop and optionally join the thread."""
        self.listening_stop_evt.set()
        self.HIL_stop_evt.set()
        self._listening_thread.join(timeout=join_timeout)
        self._HIL_thread.join(timeout=join_timeout)

    # helpers
    def _parse_line(self, text):
        # Try JSON then literal_eval then raw string
        try:
            return json.loads(text)
        except Exception:
            try:
                return ast.literal_eval(text)
            except Exception:
                return text.strip()
    
    def _conn_listener(self, conn, addr):
        # We need to separate the data_processing process from the HIL loop
        conn.settimeout(1.0)
        buffer = b""
        self.HIL_active = False  # whether we've received START and are actively listening for actions
        while not self.listening_stop_evt.is_set():
            try:
                data = conn.recv(4096)
            except socket.timeout:
                data = b""
            if not data:
                time.sleep(0.01)
                continue
            buffer += data
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                text = line.decode().strip()
                if not text:
                    continue
                parsed = self._parse_line(text)
                # cprint(f"parsed data from {addr}: {parsed}", "magenta")
                # control commands
                if isinstance(parsed, str) and parsed.lower() == "start":
                    cprint(f"Received START from {addr}", "green")
                    if not self.HIL_active:
                        with self.lock:
                            self.hil_indices = np.where(self.hil_indicator == True)[0]
                            self.HIL_id = np.array([self.hil_indices[0]])
                        cprint(f"session active, current HIL env indices: {self.hil_indices}", "green")
                        if self.hil_indices.size == 0:
                            cprint(f"No HIL envs currently, stopping the session.", "yellow")
                            with self.lock:
                                self.HIL_active = False
                            continue
                        else:
                            # set the counterfeit env to the HIL_id's current state
                            with self.env_lock:
                                HIL_state = np.array(self.env.get_sim_state())[self.HIL_id]
                            cprint(f"HIL_state.shape: {HIL_state}", "cyan")
                            with self.counterfeit_env_lock:
                                # set the counterfeit env to the HIL env's last finished state
                                self.HIL_env.reset(id=self.HIL_counterfeit_id.tolist())
                                obs_HIL = self.HIL_env.set_init_state(HIL_state, id=self.HIL_counterfeit_id.tolist())
                            with self.lock:
                                self.obs_manager[self.HIL_id] = obs_HIL
                                                     
                        
                        cprint(f"Current HIL env id: {self.HIL_id}", "magenta")
                        with self.lock:
                            # activate HIL control finally
                            self.HIL_active = True

                if isinstance(parsed, str) and parsed.lower() == "stop":
                    cprint(f"Received STOP from {addr}", "yellow")
                    # TODO, when stop, we should return the HIL envs to normal control, by setting hil_indicator false, and pop the env indices from hil_envs_indices_set
                    if self.HIL_active: 
                        with self.lock:
                            self.init_state_indices[self.HIL_id] = (self.init_state_indices[self.HIL_id] + 1) % self.initial_states.shape[0]
                            stop_init_states = self.initial_states[self.init_state_indices[self.HIL_id]]

                        with self.counterfeit_env_lock:
                            self.HIL_env.reset(id=self.HIL_counterfeit_id.tolist())
                            obs_stop = self.HIL_env.set_init_state(stop_init_states, id=self.HIL_counterfeit_id.tolist())
                            cprint(f"obs_stop.keys: {obs_stop[0].keys()}", "magenta")

                        with self.lock:
                            self.obs_manager[self.HIL_id] = obs_stop
                            self.env_timesteps[self.HIL_id] = 0
                            self.dones[self.HIL_id] = False  # reset done flag for succeeded envs
                            self.hil_indicator[self.HIL_id] = False
                            self.hil_envs_indices_set.discard(self.HIL_id[0])
                            # update running_env_indices
                            self.running_env_indices = np.where(self.hil_indicator == False)[0]
                            self.env_running_timesteps = self.env_timesteps[self.running_env_indices]
                            cprint(f"hil_indicator: {self.hil_indicator}, hil_envs_indices_set: {self.hil_envs_indices_set}", "yellow")
                        
                        cprint(f"Env {self.HIL_id} returned to normal policy control.", "yellow")
                    with self.lock:
                        self.HIL_active = False
                    continue

                if self.HIL_active and isinstance(parsed, list) and len(parsed) == len(LIBERO_DUMMY_ACTION) :
                    # got a space-mouse action; apply to HIL envs
                    try:
                        action_arr = np.array([parsed], dtype=float)
                        with self.lock:
                            if self.action_queue.full():
                                self.action_queue.get()
                            self.action_queue.put(action_arr)
                    except Exception:
                        cprint(f"Failed to convert action to floats: {parsed}", "red")
                        continue
    
    def _listening_server_loop(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            srv.bind((self.args.spm_host, self.args.spm_port))
            srv.listen(1)
            srv.settimeout(1.0)
            cprint(f"Manual-control server listening on {self.args.spm_host}:{self.args.spm_port}", "green")
            while not self.listening_stop_evt.is_set():
                try:
                    conn, addr = srv.accept()
                except socket.timeout:
                    continue
                cprint(f"Manual-control client connected from {addr}", "green")
                self._conn_listener(conn, addr)
                cprint(f"Manual-control client from {addr} disconnected", "yellow")
        finally:
            try:
                srv.close()
            except Exception:
                pass
        cprint("Manual-control server stopped", "yellow")
    
    def _HIL_loop(self):
        
        while not self.HIL_stop_evt.is_set():
            try:
                # action payloads are list/tuple/ndarray of floats
                if not self.HIL_active:
                    if self.HIL_window_active:
                        cv2.destroyWindow(self.HIL_window_name)
                        cv2.destroyWindow(self.HIL_wrist_window_name)
                        self.HIL_window_active = False
                    time.sleep(0.05)
                    continue
                
                if not self.HIL_window_active:
                    win_w, win_h = int(self.args.resize_size * 2), int(self.args.resize_size * 2)
                    cv2.namedWindow(self.HIL_wrist_window_name, cv2.WINDOW_NORMAL)
                    cv2.namedWindow(self.HIL_window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(self.HIL_window_name, win_w, win_h)
                    cv2.resizeWindow(self.HIL_wrist_window_name, win_w, win_h)
                    self.HIL_window_active = True
                
                
                # save state and images (try to match the main-loop recording format)
                sim_state = np.stack([np.concatenate(( x["robot0_eef_pos"],_quat2axisangle(x["robot0_eef_quat"]), x["robot0_gripper_qpos"],)) for x in self.obs_manager[self.HIL_id]], axis=0)
                cprint(f"sim_state.shape: {sim_state.shape}", "magenta")

                t_HID = int(self.env_timesteps[self.HIL_id])
                if t_HID < self.input_states_buffer.shape[0]:
                    with self.lock:
                        self.input_states_buffer[t_HID, self.HIL_id] = sim_state
                img = np.stack([np.ascontiguousarray(x["agentview_image"][::-1, ::-1]) for x in self.obs_manager[self.HIL_id]], axis=0)
                wrist_img = np.stack([np.ascontiguousarray(x["robot0_eye_in_hand_image"][::-1, ::-1]) for x in self.obs_manager[self.HIL_id]], axis=0)
                img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, self.args.resize_size, self.args.resize_size))
                wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, self.args.resize_size, self.args.resize_size))
                cprint(f"img.shape: {img.shape}, wrist_img.shape: {wrist_img.shape}", "magenta")
                cv2.imshow(self.HIL_window_name, cv2.cvtColor(img[0], cv2.COLOR_RGB2BGR))
                cv2.imshow(self.HIL_wrist_window_name, cv2.cvtColor(wrist_img[0], cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

                cprint(f"t_HID: {t_HID}", "magenta")
                if t_HID < self.img_buffer.shape[0]:
                    with self.lock:
                        self.img_buffer[t_HID, self.HIL_id] = img
                if t_HID < self.wrist_img_buffer.shape[0]:
                    with self.lock:
                        self.wrist_img_buffer[t_HID, self.HIL_id] = wrist_img

                cprint(f"HIL active for env id: {self.HIL_id}", "magenta")
                # get action from queue
                try:
                    action_arr = self.action_queue.get(timeout=1.0)
                except queue.Empty:
                    cprint(f"No action received yet waiting for action...", "yellow")
                    continue  
                
                # clamp to buffer sizes
                if t_HID < self.action_buffer.shape[0]:
                    with self.lock:
                        self.action_buffer[t_HID, self.HIL_id] = action_arr       

                # Step env for HIL indices
                cprint(f"action_arr from HIL_queue: {action_arr}", "magenta")
                with self.counterfeit_env_lock:
                    obs_id, reward, done, info = self.HIL_env.step(action_arr, id=self.HIL_counterfeit_id.tolist())

                with self.lock:
                    self.env_timesteps[self.HIL_id] += 1
                    self.obs_manager[self.HIL_id] = obs_id
                    self.dones[self.HIL_id] = self.dones[self.HIL_id] | done

                

                # also try to record sim state from env.get_sim_state() if available
                t_HID = int(self.env_timesteps[self.HIL_id])
                cprint(f"Recording at timestep {t_HID} for env {self.HIL_id}", "magenta")
                with self.counterfeit_env_lock:
                    rec_states = np.array(self.HIL_env.get_sim_state())
                if t_HID < self.state_record_buffer.shape[0]:
                    with self.lock:
                        self.state_record_buffer[t_HID, self.HIL_id] = rec_states
                
                cprint(f"rec_states.shape: {rec_states.shape}", "magenta")
                cprint(f"done: {done}, self.dones: {self.dones}", "magenta")
                
                if self.dones[self.HIL_id]:
                    cprint(f"Env {self.HIL_id} finished (done) under HIL control.", "magenta")
                    # TODO save the buffers
                        
                    with self.lock:
                        self.init_state_indices[self.HIL_id] = (self.init_state_indices[self.HIL_id] + 1) % self.initial_states.shape[0]
                        success_init_states = self.initial_states[self.init_state_indices[self.HIL_id]]
                    with self.counterfeit_env_lock:
                        self.HIL_env.reset(id=self.HIL_counterfeit_id.tolist())
                        obs_success = self.HIL_env.set_init_state(success_init_states, id=self.HIL_counterfeit_id.tolist())

                    with self.lock:
                        self.obs_manager[self.HIL_id] = obs_success
                        self.env_timesteps[self.HIL_id] = 0
                        self.dones[self.HIL_id] = False  # reset done flag for succeeded envs
                        self.success_episodes[self.HIL_id] += 1
                        self.hil_indicator[self.HIL_id] = False
                        self.hil_envs_indices_set.discard(self.HIL_id)
                        cprint(f"hil_indicator: {self.hil_indicator}, hil_envs_indices_set: {self.hil_envs_indices_set}", "yellow")

                    cv2.destroyWindow(self.HIL_window_name)
                    cv2.destroyWindow(self.HIL_wrist_window_name)
                    cprint(f"Env {self.HIL_id} returned to normal policy control.", "yellow")
                    with self.lock:
                        self.HIL_active = False  # end the HIL session after one episode
                else: 
                    cprint(f"Env {self.HIL_id} continuing HIL control at timestep {self.env_timesteps[self.HIL_id]}.", "magenta")

                # sleep to match human-control fps
                time.sleep(1.0 / float(max(1, self.args.hil_fps)))

                cprint(f"sleep for {1.0 / float(max(1, self.args.hil_fps))} seconds to match HIL FPS", "magenta")

            except Exception as e:
                cprint(f"Exception in HIL loop: {e}", "red")
                time.sleep(0.1)
                continue

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
    num_trials_per_task: int = 50  # Number of rollouts per task
    num_envs: int = 10  # Number of environments to run in parallel

#################################################################################################################
    # SpaceMouse parameters
#################################################################################################################
    spm_host = "0.0.0.0"
    spm_port = 18002
    hil_fps: int = 20  # Frequency to poll SpaceMouse for actions during human-in-the-loop control

#################################################################################################################
    # Utils
#################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos
    seed: int = 7  # Random Seed (for reproducibility)
    data_record_path: str = "data/libero/h5py_records"  # Path to save h5py records
    max_buffer_size: int = 1500  # max buffer size for action/state/img recording per env

def get_libero_runner(args: Args) -> LiberoRunner:
    return LiberoRunner(args)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # tyro.cli(eval_libero_parallel)
    runner = tyro.cli(get_libero_runner)
    runner.eval_libero_parallel()