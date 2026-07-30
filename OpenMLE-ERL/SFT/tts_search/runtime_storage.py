"""Runtime storage defaults for long-running evaluations and training."""

from __future__ import annotations

import os
from pathlib import Path


def _clean_run_id(value: str) -> str:
    clean = value.strip().replace("/", "_").replace(":", "_").replace(" ", "_")
    return clean or "rollout"


def configure_runtime_storage(default_run_id: str = "rollout") -> dict[str, str]:
    """Configure repository-local temp and cache paths unless overridden."""

    default_root = Path(__file__).resolve().parents[1] / ".runtime"
    root = Path(os.environ.get("OPENMLE_STORAGE_ROOT", str(default_root)))
    run_id = _clean_run_id(
        os.environ.get("OPENMLE_RUN_ID")
        or os.environ.get("JOB_NAME")
        or os.environ.get("TASK_NAME")
        or default_run_id
    )
    os.environ.setdefault("OPENMLE_STORAGE_ROOT", str(root))
    os.environ.setdefault("OPENMLE_RUN_ID", run_id)

    defaults = {
        "TMPDIR": root / "tmp" / run_id,
        "XDG_CACHE_HOME": root / "cache" / "xdg",
        "HF_HOME": root / "cache" / "huggingface",
        "TORCH_HOME": root / "cache" / "torch",
        "PIP_CACHE_DIR": root / "cache" / "pip",
        "WANDB_DIR": root / "wandb" / run_id,
    }
    defaults["HUGGINGFACE_HUB_CACHE"] = defaults["HF_HOME"] / "hub"
    defaults["TRANSFORMERS_CACHE"] = defaults["HF_HOME"] / "transformers"

    for key, path in defaults.items():
        os.environ.setdefault(key, str(path))

    ray_tmpdir = os.environ.get("RAY_TMPDIR", "")
    if os.environ.get("ALLOW_EPHEMERAL_TMP") != "1" and (
        not ray_tmpdir or ray_tmpdir.startswith("/tmp/")
    ):
        os.environ["RAY_TMPDIR"] = str(root / "ray" / run_id)

    paths = {key: os.environ[key] for key in defaults}
    paths["RAY_TMPDIR"] = os.environ["RAY_TMPDIR"]
    for path in paths.values():
        Path(path).mkdir(parents=True, exist_ok=True)

    return paths
