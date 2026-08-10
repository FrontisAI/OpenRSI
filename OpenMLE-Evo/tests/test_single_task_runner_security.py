from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_single_task_runner():
    path = (
        REPO_ROOT
        / "third_party"
        / "aira-evo"
        / "examples"
        / "mle_bench"
        / "single_task_runner.py"
    )
    spec = importlib.util.spec_from_file_location("single_task_runner_security", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_dojo_config_redacts_nested_operator_api_keys():
    runner = _load_single_task_runner()
    payload = {
        "solver": {
            "operators": {
                "draft": {
                    "llm": {
                        "client": {
                            "api_key": "unit-secret",
                            "model_id": "unit-model",
                        }
                    }
                }
            }
        }
    }

    redacted = runner._redact_dojo_config_payload(payload)

    assert redacted["solver"]["operators"]["draft"]["llm"]["client"]["api_key"] is None
    assert (
        redacted["solver"]["operators"]["draft"]["llm"]["client"]["model_id"]
        == "unit-model"
    )
