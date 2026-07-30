from __future__ import annotations

import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable


TASK_NAME_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")


def normalize_slug(value: str) -> str:
    """Return a Kaggle competition slug from a slug, URL, or path-like value."""
    slug = value.strip()
    if "/" not in slug:
        return slug

    parts = [part for part in slug.split("/") if part]
    for marker in ("competitions", "c"):
        if marker in parts:
            index = parts.index(marker)
            if index + 1 < len(parts):
                return parts[index + 1]
    return parts[-1] if parts else slug


def normalize_slugs(values: Iterable[str]) -> list[str]:
    clean: list[str] = []
    seen: set[str] = set()
    for value in values:
        raw = str(value).strip()
        if not raw or raw.startswith("#"):
            continue
        slug = normalize_slug(raw)
        if slug and slug not in seen:
            clean.append(slug)
            seen.add(slug)
    return clean


def read_slugs_file(path: str | Path) -> list[str]:
    return normalize_slugs(Path(path).read_text(encoding="utf-8").splitlines())


def read_slug_entries(path: str | Path) -> list[str]:
    """Read slug inputs without deduplicating so callers can report duplicates."""
    entries = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        raw_path = Path(raw)
        if raw_path.is_absolute() or any(part in {".", ".."} for part in raw_path.parts):
            entries.append(raw)
        else:
            entries.append(normalize_slug(raw))
    return entries


def read_task_list(path: str | Path) -> list[str]:
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def validate_task_name(value: str) -> str:
    """Return a safe task identifier that cannot escape a batch directory."""
    name = str(value).strip()
    if not name:
        raise ValueError("Task name is empty")
    if name in {".", ".."} or not TASK_NAME_PATTERN.fullmatch(name):
        raise ValueError(f"Unsafe task name: {value!r}")
    return name


def resolve_task_path(root: str | Path, task_name: str) -> Path:
    """Resolve a task directory and enforce containment within root."""
    root_path = Path(root).resolve()
    name = validate_task_name(task_name)
    task_path = (root_path / name).resolve()
    try:
        task_path.relative_to(root_path)
    except ValueError as exc:
        raise ValueError(f"Task path escapes root: {task_name!r}") from exc
    return task_path


def json_safe(value: Any) -> Any:
    """Recursively convert common scientific-Python values to strict JSON data."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, BaseException):
        return {
            "status": "failed",
            "success": False,
            "error": f"{type(value).__name__}: {value}",
        }
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            converted = item()
        except (TypeError, ValueError):
            pass
        else:
            if converted is not value:
                return json_safe(converted)
    return repr(value)


def atomic_write_text(path: str | Path, content: str, encoding: str = "utf-8") -> Path:
    """Atomically replace a text file without exposing a partial result."""
    target = ensure_parent(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        mode = (target.stat().st_mode & 0o777) if target.exists() else 0o644
        os.chmod(temporary, mode)
        with os.fdopen(descriptor, "w", encoding=encoding) as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def atomic_write_json(path: str | Path, value: Any) -> Path:
    return atomic_write_text(
        path,
        json.dumps(json_safe(value), ensure_ascii=False, indent=2),
    )


def collect_task_dirs(root: str | Path) -> list[Path]:
    root_path = Path(root)
    if (root_path / "data").is_dir():
        return [root_path]
    return sorted(path for path in root_path.iterdir() if path.is_dir() and (path / "data").is_dir())


def ensure_parent(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target
