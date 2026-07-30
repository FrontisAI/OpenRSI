#!/usr/bin/env python3
"""
Wrapper script for train_async.py that applies custom patches.
This ensures monkey patches are applied before slime modules are loaded.
"""

import importlib.util
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SLIME_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
PROJECT_ROOT = os.path.dirname(SLIME_ROOT)


def _realpath(path: str) -> str:
    return os.path.normcase(os.path.realpath(path))


def _pin_local_repo_paths() -> None:
    local_prefix = _realpath(PROJECT_ROOT)
    pinned_paths = [_realpath(SCRIPT_DIR), _realpath(SLIME_ROOT)]
    filtered_sys_path = []
    for entry in sys.path:
        candidate = entry or os.getcwd()
        try:
            candidate_real = _realpath(candidate)
        except OSError:
            filtered_sys_path.append(entry)
            continue
        if f"{os.sep}slime{os.sep}" in f"{candidate_real}{os.sep}" and not candidate_real.startswith(local_prefix + os.sep):
            continue
        filtered_sys_path.append(entry)

    sys.path[:] = pinned_paths + [entry for entry in filtered_sys_path if _realpath(entry or os.getcwd()) not in pinned_paths]
    os.chdir(SLIME_ROOT)
    spec = importlib.util.find_spec("slime")
    origin = spec.origin if spec is not None else "missing"
    print(f"[TRAIN_ASYNC_WITH_PATCH] Pinned repo root: {PROJECT_ROOT}")
    print(f"[TRAIN_ASYNC_WITH_PATCH] slime import resolution: {origin}")


_pin_local_repo_paths()

print("=" * 60)
print("[TRAIN_ASYNC_WITH_PATCH] Applying custom eval logger patch...")
print("=" * 60)

try:
    import patch_eval_logger

    print("[TRAIN_ASYNC_WITH_PATCH] Eval logger patch module loaded successfully")
except Exception as e:
    print(f"[TRAIN_ASYNC_WITH_PATCH] Warning: Failed to load eval logger patch: {e}")
    import traceback

    traceback.print_exc()

print("[TRAIN_ASYNC_WITH_PATCH] Applying rollout patch...")
try:
    import patch_rollout

    print("[TRAIN_ASYNC_WITH_PATCH] Rollout patch module loaded successfully")
except Exception as e:
    print(f"[TRAIN_ASYNC_WITH_PATCH] Warning: Failed to load rollout patch: {e}")
    import traceback

    traceback.print_exc()

print("[TRAIN_ASYNC_WITH_PATCH] Starting train_async.py...")
print("=" * 60)

TRAIN_ASYNC_PY = os.path.join(SLIME_ROOT, "train_async.py")

if not os.path.exists(TRAIN_ASYNC_PY):
    fallback_root = "/root/slime"
    fallback_train_async = os.path.join(fallback_root, "train_async.py")
    if os.path.exists(fallback_train_async):
        print(f"[TRAIN_ASYNC_WITH_PATCH] Error: expected local train_async.py at {TRAIN_ASYNC_PY}")
        print(f"[TRAIN_ASYNC_WITH_PATCH] Refusing to fall back to {fallback_train_async}")
        sys.exit(1)
    else:
        print(f"[TRAIN_ASYNC_WITH_PATCH] Error: train_async.py not found at {TRAIN_ASYNC_PY}")
        sys.exit(1)

print(f"[TRAIN_ASYNC_WITH_PATCH] Using slime root: {SLIME_ROOT}")

with open(TRAIN_ASYNC_PY, "r") as f:
    code = compile(f.read(), TRAIN_ASYNC_PY, "exec")
    exec(code, {"__name__": "__main__", "__file__": TRAIN_ASYNC_PY})
