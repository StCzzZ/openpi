# CUDA_VISIBLE_DEVICES=6 uv run examples/libero/manual_action_client.py

"""Simple manual action client example.

Connects to a websocket server started by run_parallel_with_visualization.py and
sends msgpack-packed actions or the text 'STOP' to terminate control.

Usage: run this script, enter the ws:// host:port when prompted (or paste from server output),
then type comma-separated floats for actions and press Enter. Type STOP to end.
"""
import sys
import socket
import time
from termcolor import cprint
import socket
import json


SPACEMOUSE_DUMMY_ACTION = [0.5] * 6 + [0.0]

def main():
    # url = input("Enter websocket url (e.g. ws://127.0.0.1:8765): ").strip()
    url = "ws://192.168.1.7:8080"

    try:
        # Connect using a plain TCP socket. The server in run_parallel_with_visualization
        # listens with a raw socket (not a WebSocket), so using the `websockets` client
        # leads to an HTTP Upgrade handshake that the server doesn't respond to.
        host = url.split("//")[-1].split(":")[0]
        port = int(url.split(":")[-1])
        sock = socket.create_connection((host, port), timeout=5.0)
        sock.settimeout(1.0)
        cprint(f"Connected to {host}:{port} (raw TCP)", "green")
        print("Type STOP to finish control and return.")

        while True:
            # send JSON text terminated by newline so the server can decode as utf-8
            txt = json.dumps(SPACEMOUSE_DUMMY_ACTION) + "\n"
            try:
                sock.sendall(txt.encode("utf-8"))
            except Exception as e:
                print("Send failed:", e)
                break

            
    except Exception as e:
        print("Connection failed:", e)
    finally:
        try:
            sock.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()
