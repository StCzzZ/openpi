import asyncio
import http
import logging
import time
import traceback
from typing import Optional

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets
import websockets.asyncio.server as _server
import websockets.frames
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv


logger = logging.getLogger(__name__)


class WebsocketSpaceMouseServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        metadata: Optional[dict] = None,
        env: Optional[SubprocVectorEnv] = None,
    ) -> None:
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._env = env
        logging.getLogger("websockets.server").setLevel(logging.INFO)
    
    def serve_forever(self) -> None:
        asyncio.run(self.run())
    
    def start_serving(self) -> None:
        asyncio.run(self.start_serving())
    
    def end_serving(self) -> None:
        asyncio.run(self.end_serving())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()
    
    async def start_serving(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.start_serving()

    async def end_serving(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.wait_closed()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        while True:
            try:
                action, env_idx = msgpack_numpy.unpackb(await websocket.recv())

                step_time = time.monotonic()
                obs_idx, reward, done, info = self._env.step(action, id=[env_idx])
                step_time = time.monotonic() - step_time

                await websocket.send(packer.pack((obs_idx, reward, done, info)))
            except websockets.exceptions.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

def _health_check(connection: _server.ServerConnection, request: _server.Request) -> Optional[_server.Response]:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
    return None
