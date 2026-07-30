from __future__ import annotations

from pathlib import Path

import yaml

RELEASE_ROOT = Path(__file__).resolve().parents[1]


def test_public_runtime_has_no_private_deployment_defaults():
    forbidden = (
        "/" + "data1/",
        "/" + "nfs2/",
        "/mnt/" + "pubdatasets",
        "101.6." + "65.",
        "183." + "222.",
        "61." + "243.",
        "thu_" + "node",
        "demo-" + "key",
    )
    roots = (
        RELEASE_ROOT / "scripts",
        RELEASE_ROOT / "tts_search",
        RELEASE_ROOT / "third_party" / "aira-evo" / "src",
        RELEASE_ROOT / "third_party" / "aira-evo" / "examples" / "mle_bench",
        RELEASE_ROOT / "third_party" / "aira-evo" / "examples" / "nature_bench",
    )
    checked = []
    for root in roots:
        for path in root.rglob("*"):
            if path.suffix not in {".py", ".yaml", ".yml", ".json"}:
                continue
            text = path.read_text(encoding="utf-8")
            assert not any(value in text for value in forbidden), path
            checked.append(path)
    assert checked


def test_public_airaevo_config_uses_vendored_runtime_and_environment():
    config_path = RELEASE_ROOT / "tts_search" / "configs" / "search" / "airaevo.yaml"
    runner = yaml.safe_load(config_path.read_text(encoding="utf-8"))["runner"]

    assert runner["package_root"] == "third_party/aira-evo"
    assert runner["task_root"] == "third_party/aira-evo/examples/mle_bench"
    assert runner["submit_data_dir_root"] == "${oc.env:OPENMLE_SUBMIT_DATA_DIR_ROOT}"


def test_launchers_pin_current_release_on_pythonpath():
    for name in ("run_standard.sh", "run_multi_gpu.sh", "run_naturebench.sh"):
        text = (RELEASE_ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "third_party/aira-evo/src" in text
        assert "PYTHONPATH" in text


def test_naturebench_lite_v2_is_parallel_and_has_ten_tasks():
    benchmark_root = RELEASE_ROOT / "benchmarks"
    assert (benchmark_root / "mle_bench" / "README.md").is_file()
    nature_root = benchmark_root / "naturebench_lite_v2"
    assert (nature_root / "README.md").is_file()
    assert (nature_root / "RUNNING.md").is_file()

    tasks = [
        line.strip()
        for line in (nature_root / "tasks.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(tasks) == 10
    assert len(set(tasks)) == 10

    experiment = yaml.safe_load(
        (
            RELEASE_ROOT
            / "tts_search"
            / "configs"
            / "experiment"
            / "naturebench_scm_lite_v2.yaml"
        ).read_text(encoding="utf-8")
    )
    assert experiment["data"]["task_list"] == tasks


def test_naturebench_public_config_is_environment_driven():
    config_path = (
        RELEASE_ROOT
        / "tts_search"
        / "configs"
        / "data"
        / "naturebench_scm_all.yaml"
    )
    text = config_path.read_text(encoding="utf-8")

    for name in (
        "NATUREBENCH_ROOT",
        "NATUREBENCH_SCM_HOST",
        "NATUREBENCH_SCM_WORKSPACE_ROOT",
        "NATUREBENCH_SCM_TASK_ROOT",
        "NATUREBENCH_SCM_EVAL_SERVICE_URL",
    ):
        assert f"${{oc.env:{name}" in text

    profile = yaml.safe_load(
        (
            RELEASE_ROOT
            / "tts_search"
            / "configs"
            / "execution"
            / "naturebench_multi_gpu.yaml"
        ).read_text(encoding="utf-8")
    )
    solver = profile["search"]["runner"]["solver"]
    assert solver["execution_mode"] == "async_steady_state"
    assert solver["async_sandbox_urls"] == ["naturebench://task-adapter"]
