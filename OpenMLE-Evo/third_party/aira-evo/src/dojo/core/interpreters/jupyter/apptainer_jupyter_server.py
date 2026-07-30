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

import re
import os
import atexit
import secrets
import signal
import subprocess
import sys
from pathlib import Path
from types import TracebackType

from typing import Dict, List

from .base import JupyterConnectable, JupyterConnectionInfo
from .jupyter_client import JupyterClient

import logging

log = logging.getLogger(__name__)


class ApptainerJupyterServer(JupyterConnectable):
    def __init__(
        self,
        token: str = ...,
        bind_inputs_dir: Path | str | None = None,
        superimage_directory: str = None,
        superimage_version: str = None,
        read_only_overlays: List[str] = None,
        read_only_binds: Dict[str, str] = None,
        env: Dict[str, str] = None,
    ):
        self.read_only_overlays = read_only_overlays or []
        self.read_only_overlays = [Path(path).resolve() for path in self.read_only_overlays]

        self.read_only_binds = read_only_binds or {}
        self.read_only_binds = {Path(k).resolve(): Path("/root") / Path(v) for k, v in self.read_only_binds.items()}

        self.env = env or {}

        bind_inputs_dir = Path(bind_inputs_dir) if bind_inputs_dir is not None else None

        if token == ...:
            token = secrets.token_hex(32)

        self.token = token
        self.bind_inputs_dir = bind_inputs_dir
        self.path_to_superimage = superimage_directory
        self.superimage_version = superimage_version

        args = [
            str(Path(__file__).parent / "sand"),
            "python",
            "-m",
            "jupyter",
            "kernelgateway",
            "--KernelGatewayApp.auth_token",
            token,
            "--JupyterApp.answer_yes=True",
            "--JupyterWebsocketPersonality.list_kernels=True",
            "--KernelGatewayApp.answer_yes=True",
            "--KernelManager.cache_ports=False",
        ]

        env = os.environ.copy()
        for k, v in self.env.items():
            env[f"RAD_{k}"] = v

        bind_configs = []
        if bind_inputs_dir is not None:
            bind_configs.append(f"{bind_inputs_dir}:/root/data:ro")
        for k, v in self.read_only_binds.items():
            k = os.path.abspath(k)
            bind_configs.append(f"{k}:{v}:ro")
        if bind_configs:
            env["APPTAINER_BIND"] = ",".join(bind_configs)

        if superimage_directory is not None:
            env["SUPERIMAGE_DIR"] = superimage_directory
        if superimage_version is not None:
            env["SUPERIMAGE_VERSION"] = superimage_version

        env["BASE_OVERLAYS"] = " ".join(f"--overlay {overlay}:ro" for overlay in self.read_only_overlays)
        log.warning(f"Starting `Sand` wrapper server with env:")
        log.warning(f"  APPTAINER_BIND: {env['APPTAINER_BIND']}")
        log.warning(f"  SUPERIMAGE_DIR: {env['SUPERIMAGE_DIR']}")
        log.warning(f"  SUPERIMAGE_VERSION: {env['SUPERIMAGE_VERSION']}")
        log.warning(f"  BASE_OVERLAYS: {env['BASE_OVERLAYS']}")
        log.warning(f"with args: {args}")
        self._subprocess = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            env=env,
        )

        # Satisfy mypy, we know this is not None because we passed PIPE
        assert self._subprocess.stderr is not None
        # Read stderr until we see "is available at" or the process has exited with an error
        stderr = ""
        while True:
            result = self._subprocess.poll()
            if result is not None:
                stderr += self._subprocess.stderr.read()
                raise ValueError(f"Jupyter gateway server failed to start with exit code: {result}. stderr:\n{stderr}")
            line = self._subprocess.stderr.readline()
            stderr += line

            log.warning(line.rstrip("\n"))

            if "ERROR:" in line:
                error_info = line.split("ERROR:")[1]
                raise ValueError(f"Jupyter gateway server failed to start. {error_info}")

            match = re.search(r"is available at http://([^:]+):(\d+)", line)
            if match:
                self.ip = match.group(1)
                self.port = int(match.group(2))
                break

        result = self._subprocess.poll()
        if result is not None:
            raise ValueError("Jupyter gateway server failed to start.")

        atexit.register(self.stop)

    def stop(self) -> None:
        if self._subprocess is None:
            return
        log.warning("Stopping Jupyter server...")
        if self._subprocess.poll() is None:
            os.killpg(os.getpgid(self._subprocess.pid), signal.SIGTERM)
            log.warning("Sent TERMINATE signal to Jupyter server. Waiting to terminate...")
            try:
                self._subprocess.wait(timeout=120)  # 2 minutes of grace period
            except subprocess.TimeoutExpired:
                log.error("!important : Jupyter server did not terminate. Forcing with KILL signal...")
                os.killpg(os.getpgid(self._subprocess.pid), signal.SIGKILL)
                try:
                    self._subprocess.wait(timeout=120)
                    log.warning("Jupyter killed successfully.")
                except subprocess.TimeoutExpired:
                    log.error("!important : Jupyter server did not terminate. I will wait for forever here")
                    self._subprocess.wait()
            self._subprocess = None
            log.warning("Jupyter server terminated.")

    @property
    def connection_info(self) -> JupyterConnectionInfo:
        return JupyterConnectionInfo(
            host=self.ip,
            use_https=False,
            port=self.port,
            token=self.token,
        )

    def get_client(self) -> JupyterClient:
        return JupyterClient(self.connection_info)

    def __enter__(self):
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> None:
        self.stop()

    def __del__(self):
        self.stop()
