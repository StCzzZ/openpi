Progress update — what I changed and how to use it

Summary

I added a small websocket-based manual-control flow to run_parallel_with_visualization.py so that when you intervene on a failed env (you answer "y"), after resetting that env the script starts a tiny websocket server (on localhost, ephemeral port) and enters a manual-control loop that:
shows a preview window for the selected env,
accepts msgpack-packed actions sent to the websocket,
performs env.step(action, id=[idx]) in the main thread (avoiding thread-safety issues),
stops when the client sends a STOP sentinel or disconnects, and then continues the normal cleanup and statistics code (the existing logic after the intervention block).
I also added an example client manual_action_client.py that connects to the server and allows interactive sending of actions (comma-separated floats) or sending STOP to end manual control.
I ran a quick Python syntax check (py_compile) on both modified/added files — no syntax errors.
What files I changed / added

Modified: run_parallel_with_visualization.py
Added imports (msgpack_numpy, socket, threading, queue, asyncio, websockets.asyncio.server).
Added a helper to start a websocket server in a background thread which pushes received msgpack actions into a queue.Queue.
Integrated the manual-control loop into the intervention block where you previously stepped dummy actions and showed the window. The main thread consumes the queue and calls env.step(...) for the specific env id.
The server prints the websocket address (host:port) so you can connect with the client.
Added: manual_action_client.py
Connects to a ws:// address, receives the initial metadata frame, then loops accepting user input:
Comma-separated floats -> packs and sends via msgpack
STOP -> sends text STOP and exits
How to use it (quick)

Run run_parallel_with_visualization.py as you already do. When an env fails and you choose to intervene for Env idx and type y, the script will:
reset that env and run the stabilization dummy steps (same as before),
start a manual control websocket server and print something like:
ws://127.0.0.1:XXXXX
In another terminal, run the client:
python3 manual_action_client.py
Paste the printed ws URI and press Enter.
Enter actions as comma-separated floats, e.g.:
0.1,0.0,-0.3,0.2,0.0,0.0, -1.0
After sending actions, the main script will step that single env with the action you sent. The client will optionally print server responses if provided.
Type STOP to end manual control. The server will close, the client exits, and run_parallel_with_visualization.py continues to the rest of the episode cleanup and statistics code (the same place the previous intervention flow went).
Design decisions & assumptions

I preserved thread-safety by having the websocket thread only place received actions into a queue; the main thread does env.step(...) for the given idx.
The server binds to localhost (127.0.0.1) and uses an ephemeral port (chosen automatically) and prints it. This avoids being exposed externally by default.
Messages must be msgpack-packed arrays/lists or dicts understood by msgpack_numpy.unpackb. The client example sends plain lists (packed) or the text STOP.
A single client is expected per manual-control session. If the client disconnects we treat that as stop.
The code uses the repo's openpi_client.msgpack_numpy utility for packing/unpacking to be consistent with existing websocket code in the repo.
I intentionally kept the implementation minimal and low-risk (no auth, no TLS). If you need to expose the server beyond localhost or add auth, we can add that.
Limitations and next steps (optional / recommended)

Add authentication (API key) if you want the server reachable from other machines.
Add a small ack protocol (server sends ack on successful step, or sends back resulting observation/metrics) if you want live telemetry on the client side.
Add explicit lifecycle/timeout handling if the client doesn't send anything for long periods.
Add a nicer "space mouse" client that reads the space-mouse device (depends on your hardware / libraries) and sends actions; I kept the example generic so it works for any input method.
Add logging or more robust closing of the asyncio loop in the server thread (currently we join with a small timeout, which should be fine for interactive use).
If you'd like, I can:

Implement a client that reads an actual SpaceMouse device and maps inputs to LIBERO action vectors (I would need details on the device library you use and the required mapping), or
Add optional TLS and API-key header support to the simple server/client for remote control, or
Make the server bind to a user-specified host/port instead of ephemeral, or
Add server-to-client responses (e.g., server returns the resulting image/observation after each step) so you can build a remote UI.
Would you like me to:

Add a SpaceMouse example client (if you tell me which library you use), or
Add API-key support and allow non-localhost binding?
If you want, I can also run a quick dry-run locally (start a small simulated run) to demonstrate the interaction — tell me which you'd prefer and I’ll proceed.