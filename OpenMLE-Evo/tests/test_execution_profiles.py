from __future__ import annotations

from pathlib import Path

import yaml
from hydra import compose, initialize_config_dir

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "tts_search" / "configs"


def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def test_default_experiment_selects_standard_execution_profile():
    config = _read_yaml(CONFIG_ROOT / "experiment" / "default.yaml")

    assert {"/execution": "standard"} in config["defaults"]


def test_execution_profiles_own_mode_workers_and_endpoints():
    standard = _read_yaml(CONFIG_ROOT / "execution" / "standard.yaml")
    multi_gpu = _read_yaml(CONFIG_ROOT / "execution" / "multi_gpu.yaml")

    standard_solver = standard["search"]["runner"]["solver"]
    multi_solver = multi_gpu["search"]["runner"]["solver"]

    assert standard_solver == {
        "execution_mode": "generation",
        "async_workers": 1,
        "async_sandbox_urls": [],
        "async_sandbox_assignment": "round_robin",
        "async_checkpoint_every_commits": 1,
    }
    assert multi_solver["execution_mode"] == "async_steady_state"
    assert multi_solver["async_workers"] == "${oc.decode:${oc.env:AIRAEVO_WORKERS,8}}"
    assert multi_solver["async_sandbox_urls"] == ["${oc.env:SANDBOX_ROUTER_URL}"]
    assert multi_gpu["sandbox"]["base_url"] == "${oc.env:SANDBOX_ROUTER_URL}"


def test_standard_sandbox_default_is_not_the_multi_gpu_router():
    sandbox = _read_yaml(CONFIG_ROOT / "sandbox" / "default.yaml")

    assert sandbox["base_url"] == "${oc.env:SANDBOX_URL}"


def test_search_configs_do_not_duplicate_execution_settings():
    for name in ("airaevo.yaml",):
        solver = _read_yaml(CONFIG_ROOT / "search" / name)["runner"]["solver"]
        assert not {
            "execution_mode",
            "async_workers",
            "async_worker_mode",
            "async_sandbox_urls",
            "async_sandbox_assignment",
            "async_checkpoint_every_commits",
        }.intersection(solver)


def test_hydra_profile_switches_resolved_execution_config(monkeypatch):
    monkeypatch.setenv("OPENMLE_EVAL_DATA", "/datasets/eval.parquet")
    monkeypatch.setenv("OPENMLE_LEADERBOARD_DIR", "/datasets/leaderboards")
    monkeypatch.setenv("OPENMLE_SUBMIT_DATA_DIR_ROOT", "/datasets/prepared")
    monkeypatch.setenv("SANDBOX_URL", "http://sandbox.test")
    monkeypatch.setenv("SANDBOX_ROUTER_URL", "http://router.test")
    monkeypatch.setenv("AIRAEVO_WORKERS", "4")

    with initialize_config_dir(
        config_dir=str(CONFIG_ROOT),
        version_base=None,
    ):
        standard = compose(
            config_name="experiment/openmle_evo",
        )
        multi_gpu = compose(
            config_name="experiment/openmle_evo",
            overrides=["execution=multi_gpu"],
        )

    assert standard.search.runner.solver.execution_mode == "generation"
    assert standard.search.runner.solver.async_workers == 1
    assert standard.sandbox.base_url == "http://sandbox.test"

    assert multi_gpu.search.runner.solver.execution_mode == "async_steady_state"
    assert multi_gpu.search.runner.solver.async_workers == 4
    assert list(multi_gpu.search.runner.solver.async_sandbox_urls) == [
        "http://router.test"
    ]
    assert multi_gpu.sandbox.base_url == "http://router.test"
