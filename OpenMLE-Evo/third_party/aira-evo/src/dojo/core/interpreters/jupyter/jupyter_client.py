# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Portions of this file are MIT-licensed
# Copyright (c) 2024 Microsoft Corporation
# See THIRD_PARTY_LICENSES.md for the full licence text.
# https://github.com/microsoft/autogen/blob/main/LICENSE-CODE

from __future__ import annotations

import time
import datetime
import json
import sys
from textwrap import indent
import uuid
from dataclasses import dataclass
from types import TracebackType
from typing import Any, List, cast

# from ...doc_utils import export_module

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self


import requests
from requests.adapters import HTTPAdapter, Retry

from .base import JupyterConnectionInfo

import threading
import queue

# import websocket
# from websocket import WebSocket
import websocket
from websocket import WebSocketApp

import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class JupyterClient:
    def __init__(self, connection_info: JupyterConnectionInfo):
        """(Experimental) A client for communicating with a Jupyter gateway server.

        Args:
            connection_info (JupyterConnectionInfo): Connection information
        """
        self._connection_info = connection_info
        self._session = requests.Session()
        retries = Retry(total=5, backoff_factor=0.1)
        self._session.mount("http://", HTTPAdapter(max_retries=retries))

    def _get_headers(self) -> dict[str, str]:
        if self._connection_info.token is None:
            return {}
        return {"Authorization": f"token {self._connection_info.token}"}

    def _get_api_base_url(self) -> str:
        protocol = "https" if self._connection_info.use_https else "http"
        port = f":{self._connection_info.port}" if self._connection_info.port else ""
        return f"{protocol}://{self._connection_info.host}{port}"

    def _get_ws_base_url(self) -> str:
        port = f":{self._connection_info.port}" if self._connection_info.port else ""
        return f"ws://{self._connection_info.host}{port}"

    def list_kernel_specs(self) -> dict[str, dict[str, str]]:
        response = self._session.get(f"{self._get_api_base_url()}/api/kernelspecs", headers=self._get_headers())
        return cast(dict[str, dict[str, str]], response.json())

    def list_kernels(self) -> list[dict[str, str]]:
        response = self._session.get(f"{self._get_api_base_url()}/api/kernels", headers=self._get_headers())
        return cast(list[dict[str, str]], response.json())

    def start_kernel(self, kernel_spec_name: str) -> str:
        """Start a new kernel.

        Args:
            kernel_spec_name (str): Name of the kernel spec to start

        Returns:
            str: ID of the started kernel
        """
        response = self._session.post(
            f"{self._get_api_base_url()}/api/kernels",
            headers=self._get_headers(),
            json={"name": kernel_spec_name},
        )
        return cast(str, response.json()["id"])

    def delete_kernel(self, kernel_id: str) -> None:
        response = self._session.delete(
            f"{self._get_api_base_url()}/api/kernels/{kernel_id}", headers=self._get_headers()
        )
        response.raise_for_status()

    def restart_kernel(self, kernel_id: str) -> None:
        response = self._session.post(
            f"{self._get_api_base_url()}/api/kernels/{kernel_id}/restart", headers=self._get_headers()
        )
        response.raise_for_status()

    def interrupt_kernel(self, kernel_id: str) -> None:
        log.info(f"Interrupting kernel {kernel_id}...")
        response = self._session.post(
            f"{self._get_api_base_url()}/api/kernels/{kernel_id}/interrupt", headers=self._get_headers()
        )
        log.info(f"Kernel {kernel_id} interrupted sent")
        response.raise_for_status()
        log.info(f"Response: {response}")

    def get_kernel_client(self, kernel_id: str) -> JupyterKernelClient:
        ws_url = f"{self._get_ws_base_url()}/api/kernels/{kernel_id}/channels"
        headers = self._get_headers()
        return JupyterKernelClient(ws_url, headers)


class JupyterKernelClient:
    """(Experimental) A client for communicating with a Jupyter kernel."""

    @dataclass
    class ExecutionResult:
        @dataclass
        class DataItem:
            mime_type: str
            data: str

        is_ok: bool
        output: List[str]
        data_items: list[DataItem]
        timed_out: bool = False

    def __init__(self, url: str, headers: dict[str, str]):
        self._session_id: str = uuid.uuid4().hex

        self._message_queue: queue.Queue[str] = queue.Queue()
        self._connected_event = threading.Event()

        header_list = [f"{k}: {v}" for k, v in headers.items()]

        self._ws_app = WebSocketApp(
            url,
            header=header_list,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )

        self._thread = threading.Thread(target=self._ws_app.run_forever, daemon=True)
        self._thread.start()

        self._time_cycle = 300
        self._connected_event.wait(timeout=self._time_cycle)

    def _on_open(self, ws: WebSocketApp) -> None:
        self._connected_event.set()

    def _on_message(self, ws: WebSocketApp, message: str) -> None:
        self._message_queue.put(message)

    def _on_error(self, ws: WebSocketApp, error: Any) -> None:
        print(f"WebSocket error: {error}")

    def _on_close(self, ws: WebSocketApp, close_status_code: int, close_msg: str) -> None:
        pass

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> None:
        self.stop()

    def stop(self) -> None:
        if self._ws_app:
            _ws_app = self._ws_app
            self._ws_app = None
            _ws_app.close()

    def _send_message(self, *, content: dict[str, Any], channel: str, message_type: str) -> str:
        timestamp = datetime.datetime.now().isoformat()
        message_id = uuid.uuid4().hex
        message = {
            "header": {
                "username": "autogen",
                "version": "5.0",
                "session": self._session_id,
                "msg_id": message_id,
                "msg_type": message_type,
                "date": timestamp,
            },
            "parent_header": {},
            "channel": channel,
            "content": content,
            "metadata": {},
            "buffers": {},
        }
        # self._websocket.send_text(json.dumps(message))
        self._ws_app.send(json.dumps(message))
        return message_id

    def _receive_message(self, timeout_seconds: float | None) -> dict[str, Any] | None:
        try:
            data = self._message_queue.get(timeout=timeout_seconds)
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            return cast(dict[str, Any], json.loads(data))
        except queue.Empty:
            return None

    def wait_for_ready(self, timeout_seconds: float | None = None) -> bool:
        message_id = self._send_message(content={}, channel="shell", message_type="kernel_info_request")
        while True:
            message = self._receive_message(timeout_seconds)
            # This means we timed out with no new messages.
            if message is None:
                return False
            if (
                message.get("parent_header", {}).get("msg_id") == message_id
                and message["msg_type"] == "kernel_info_reply"
            ):
                return True

    def execute(self, code: str, timeout_seconds: float | None = None) -> ExecutionResult:
        if timeout_seconds is None:
            # Default to 7 days
            timeout_seconds = 7 * 24 * 60 * 60
        time_cycle = max(7, min(self._time_cycle, timeout_seconds / 5))

        message_id = self._send_message(
            content={
                "code": code,
                "silent": False,
                "store_history": True,
                "user_expressions": {},
                "allow_stdin": False,
                "stop_on_error": True,
            },
            channel="shell",
            message_type="execute_request",
        )

        start_time = time.monotonic()

        text_output = []
        data_output = []
        while True:
            # check if we timed out
            message = self._receive_message(time_cycle)

            if time.monotonic() - start_time > timeout_seconds:
                log.info(f"\033[90m Timeout waiting for output from code block. \033[0m")
                result = JupyterKernelClient.ExecutionResult(
                    is_ok=False,
                    output=["ERROR: Timeout waiting for output from code block."],
                    data_items=[],
                    timed_out=True,
                )
                return result

            if message is None:
                log.info(f"\033[90m No stream received for {time_cycle} \033[0m")
                continue

            # Ignore messages that are not for this execution.
            if message.get("parent_header", {}).get("msg_id") != message_id:
                log.info(
                    f"\033[90m Skipping; Message not for this execution: {message.get('parent_header', {}).get('msg_id')} != {message_id} \033[0m"
                )
                continue

            msg_type = message.get("msg_type")
            content = message.get("content")

            log.info(f"\033[90m Received message: {msg_type} \033[0m")

            if msg_type == "status":
                if content["execution_state"] == "idle":
                    log.info(f"\033[90m Stopping the loop; Execution_state is idle \033[0m")
                    break
                if content["execution_state"] == "busy":
                    log.info(f"\033[90m Kernel became busy with work\033[0m")
                    continue

            if msg_type == "error":
                # Output is an error.
                return JupyterKernelClient.ExecutionResult(
                    is_ok=False,
                    output=["ERROR:", f"{content['ename']}: {content['evalue']}\n", *content["traceback"]],
                    data_items=[],
                )

            if msg_type in ["execute_result", "display_data"]:
                for data_type, data in content["data"].items():
                    if data_type == "text/plain":
                        text_output.append(data)
                    else:
                        data_output.append(self.ExecutionResult.DataItem(mime_type=data_type, data=data))
                continue

            if msg_type == "stream":
                chunk = content["text"]
                text_output.append(chunk)
                indented_chunk = indent(chunk, "... ", lambda line: True).strip()
                log.info(f"\033[90m Stream output (length={len(chunk)} chars):\n{indented_chunk}\033[0m")
                continue

            if msg_type == "execute_input":
                pass  # skip this one

            log.info(f"\033[90m Unknown message type, will not handle: {msg_type}\033[0m")

        return JupyterKernelClient.ExecutionResult(is_ok=True, output=text_output, data_items=data_output)
