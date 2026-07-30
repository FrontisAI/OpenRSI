from __future__ import annotations

import os

PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def disable_proxy_env() -> None:
    for name in PROXY_ENV_VARS:
        os.environ.pop(name, None)
