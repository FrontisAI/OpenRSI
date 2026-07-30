from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


def _load_module(file_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("openmle_isolated_prepare", file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load prepare module: {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def execute_prepare(
    prepare_path: str,
    raw_dir: str,
    public_dir: str,
    private_dir: str,
) -> dict[str, bool]:
    """Execute generated prepare code inside the configured OS sandbox."""
    module = _load_module(Path(prepare_path))
    prepare = getattr(module, "prepare", None)
    if not callable(prepare):
        raise ValueError("No prepare function found, code inexecutable")
    prepare(Path(raw_dir), Path(public_dir), Path(private_dir))
    return {"executed": True}
