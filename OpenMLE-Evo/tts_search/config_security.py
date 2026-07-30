from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_SENSITIVE_CONFIG_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "auth_token",
        "password",
        "secret",
        "token",
    }
)


def _is_sensitive_key(value: Any) -> bool:
    normalized_key = str(value).strip().lower()
    return (
        normalized_key in _SENSITIVE_CONFIG_KEYS
        or normalized_key.endswith("_api_key")
        or normalized_key.endswith("_password")
        or normalized_key.endswith("_secret")
        or normalized_key.endswith("_token")
    )


def _redact_override_string(value: str) -> str:
    key_path, separator, _ = value.partition("=")
    if not separator:
        return value
    leaf_key = key_path.rsplit(".", 1)[-1].lstrip("+~")
    if not _is_sensitive_key(leaf_key):
        return value
    return f"{key_path}=<redacted>"


def redact_sensitive_config(value: Any) -> Any:
    """Return a copy with credential-bearing config fields replaced by null."""
    if isinstance(value, dict):
        redacted: dict[Any, Any] = {}
        for key, item in value.items():
            if _is_sensitive_key(key):
                redacted[key] = None
            else:
                redacted[key] = redact_sensitive_config(item)
        return redacted
    if isinstance(value, list):
        return [redact_sensitive_config(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_sensitive_config(item) for item in value)
    if isinstance(value, str):
        return _redact_override_string(value)
    return value


def redact_sensitive_yaml_file(path: Path) -> bool:
    """Redact a YAML mapping/list in place; return False for unsupported files."""
    if not path.exists():
        return False
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, (dict, list)):
        return False
    path.write_text(
        yaml.safe_dump(
            redact_sensitive_config(payload),
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return True
