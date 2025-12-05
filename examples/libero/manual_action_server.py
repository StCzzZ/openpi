# CUDA_VISIBLE_DEVICES=6 uv run examples/libero/manual_action_server.py

import time
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
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client import msgpack_numpy
import socket
import threading
import queue
import asyncio
import websockets.asyncio.server as _ws_server
import tqdm
import tyro

# ignore warnings
import warnings
warnings.filterwarnings("ignore")
from termcolor import cprint
import json
import ast

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

SPACEMOUSE_DUMMY_ACTION = [0.5] * 6 + [0.0]

def _client_handler(conn, addr, q, stop_event):
    conn.settimeout(1.0)
    buffer = b""
    try:
        while not stop_event.is_set():
            try:
                data = conn.recv(4096)
                cprint(f"data received: {data}", "blue")
            except socket.timeout:
                cprint(f"socket timeout from {addr}, continuing...", "yellow")
                continue
            if not data:
                cprint(f"no data recieved.", "red")
                break
            buffer += data
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                text = line.decode().strip()
                if not text:
                    continue
                if text.lower() == "stop":
                    cprint(f"Received STOP from {addr}", "yellow")
                    stop_event.set()
                    return
                # try JSON first, fallback to python literal
                parsed = None
                try:
                    parsed = json.loads(text)
                except Exception:
                    try:
                        parsed = ast.literal_eval(text)
                    except Exception:
                        # cprint(e, "red")
                        parsed = None
                cprint(f"parsed: {parsed}", "blue")
                cprint(f"text: {text}", "blue")
                # Ignore common HTTP/WebSocket handshake headers (clients using websockets lib send these)
                if parsed is None:
                    cprint("Ignoring HTTP/WebSocket handshake header", "yellow")
                    continue
                if isinstance(parsed, (list, tuple)) and len(parsed) == len(LIBERO_DUMMY_ACTION):
                    try:
                        arr = [float(x) for x in parsed]
                        q.put(arr)
                    except Exception:
                        cprint(f"Received data but failed to convert to floats: {parsed}", "red")
                else:
                    pass

    finally:
        try:
            conn.close()
        except Exception:
            pass

def _server_thread(host, port, q, stop_event):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    srv.settimeout(1.0)
    cprint(f"Manual-control server listening on {host}:{port}", "green")
    try:
        while not stop_event.is_set():
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue
            cprint(f"Manual-control client connected from {addr}", "green")
            _client_handler(conn, addr, q, stop_event)
            cprint(f"Manual-control client from {addr} disconnected", "yellow")
            if stop_event.is_set():
                break
    finally:
        try:
            srv.close()
        except Exception:
            pass
    cprint("Manual-control server stopped", "yellow")

@dataclasses.dataclass
class Args:
    
    spm_host = "0.0.0.0"
    spm_port = 18002
    stop_str = "STOP"  # String to signal stopping manual control

def manual_action_server(args: Args) -> None:

    action_queue = queue.Queue()
    stop_event = threading.Event()

    # start server thread
    server_thread = threading.Thread(target=_server_thread, args=(args.spm_host, args.spm_port, action_queue, stop_event), daemon=True)
    server_thread.start()

    while not stop_event.is_set():
        # enter the listening loop
        waiting_cnt = 1
        while not stop_event.is_set(): # use stop_event.set() to stop the loop
            try:
                action = action_queue.get(timeout=0.4)
                cprint(f"action: {action}", "blue")
            except queue.Empty:
                # cprint(f"failed to get action from queue, continue waiting...", "yellow")
                waiting_cnt += 1
                if waiting_cnt % 50 == 0:
                    cprint(f"Waiting for action for connection", "yellow")
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
            cprint(f"Received valid action: {action_arr}", "green")

            time.sleep(0.1)

if __name__ == '__main__':
    tyro.cli(manual_action_server)