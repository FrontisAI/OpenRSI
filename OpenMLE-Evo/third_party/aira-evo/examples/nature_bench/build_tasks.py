#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _sanitize_task_name(task_name: str) -> str:
    return task_name.replace("/", "_").replace("\\", "_")


def _read_optional(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _visible_data_analysis_for_task(data_cfg: Any, task_name: str) -> str:
    root_value = data_cfg.get("visible_data_analysis_root")
    if not root_value:
        return ""

    root = Path(str(root_value)).expanduser()
    if not root.is_absolute():
        root = REPO_ROOT / root
    document_path = root / f"{task_name}.md"
    if not document_path.is_file():
        raise FileNotFoundError(
            f"Missing visible-data analysis for NatureBench task {task_name}: "
            f"{document_path}"
        )

    marker = "## 5. Empirical EDA Addendum"
    document = document_path.read_text(encoding="utf-8")
    _, marker_found, addendum = document.partition(marker)
    if not marker_found or not addendum.strip():
        raise ValueError(
            f"Visible-data analysis for NatureBench task {task_name} must contain "
            f"a non-empty {marker!r} section: {document_path}"
        )
    return addendum.strip()


def _plain(value: Any) -> Any:
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


OFFICIAL_RESOURCE_LINE_DEFAULTS: dict[str, dict[str, Any]] = {
    "cpu_thu": {
        "task_set_path": "gemini_3.5_flash_cpu_thu.txt",
        "resource": "cpu",
        "gpu_mode": "none",
    },
    "gpu_thu_normal": {
        "task_set_path": "gemini_3.5_flash_gpu_thu_normal.txt",
        "resource": "gpu_thu_normal",
        "gpu_mode": "exclusive",
    },
    "gpu_thu_shared": {
        "task_set_path": "gemini_3.5_flash_gpu_thu_shared.txt",
        "resource": "gpu_thu_shared",
        "gpu_mode": "shared",
        "shared_gpu_slots": 4,
    },
    "gpu_amd_normal": {
        "task_set_path": "gemini_3.5_flash_gpu_amd_normal.txt",
        "resource": "gpu_amd_normal",
        "gpu_mode": "exclusive",
    },
    "gpu_amd_shared": {
        "task_set_path": "gemini_3.5_flash_gpu_amd_shared.txt",
        "resource": "gpu_amd_shared",
        "gpu_mode": "shared",
        "shared_gpu_slots": 3,
    },
}


def _task_names_from_cfg(data_cfg: Any, tasks_root: Path) -> list[str]:
    task_list = list(_plain(data_cfg.get("task_list")) or [])
    if task_list:
        return [str(item) for item in task_list]

    task_set_path = data_cfg.get("task_set_path")
    if task_set_path:
        path = Path(str(task_set_path))
        if not path.is_absolute():
            path = Path(str(data_cfg.naturebench_root)) / path
        return [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

    return sorted(path.name for path in tasks_root.iterdir() if path.is_dir())


def _task_set_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def _task_set_ids_from_host(host: str, path: str) -> set[str]:
    process = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            str(host),
            f"cat {shlex.quote(path)}",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if process.returncode != 0:
        detail = (process.stderr or process.stdout or "").strip()
        raise RuntimeError(
            f"Failed to read NatureBench resource task-set from {host}:{path}: "
            f"{detail[-1000:]}"
        )
    return {
        line.strip()
        for line in process.stdout.splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def _merge_resource_line_specs(data_cfg: Any) -> dict[str, dict[str, Any]]:
    overrides = dict(_plain(data_cfg.get("scm_resource_lines")) or {})
    specs: dict[str, dict[str, Any]] = {}
    for line_name, default in OFFICIAL_RESOURCE_LINE_DEFAULTS.items():
        merged = dict(default)
        merged.update(dict(overrides.get(line_name) or {}))
        specs[line_name] = merged
    for line_name, override in overrides.items():
        if line_name not in specs:
            specs[line_name] = dict(override or {})
    return specs


def _resolve_resource_line_task_set_path(
    spec: dict[str, Any],
    *,
    naturebench_root: Path,
    data_cfg: Any,
) -> Path | str:
    task_set_path = str(spec.get("task_set_path") or spec.get("task_set") or "")
    if not task_set_path:
        return ""
    path = Path(task_set_path)
    if path.is_absolute():
        return str(path)

    task_set_root = spec.get("task_set_root", data_cfg.get("scm_task_set_root"))
    if task_set_root:
        return str(Path(str(task_set_root)) / task_set_path)
    return naturebench_root / "task-set" / task_set_path


def _resource_line_map(
    data_cfg: Any,
    naturebench_root: Path,
) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    default_task_set_host = data_cfg.get("scm_task_set_host")
    for line_name, spec in _merge_resource_line_specs(data_cfg).items():
        path = _resolve_resource_line_task_set_path(
            spec,
            naturebench_root=naturebench_root,
            data_cfg=data_cfg,
        )
        if not path:
            continue
        task_set_host = spec.get("task_set_host", default_task_set_host)
        task_ids = (
            _task_set_ids_from_host(str(task_set_host), str(path))
            if task_set_host
            else _task_set_ids(Path(path))
        )
        if not task_ids:
            continue
        line_payload = dict(spec)
        line_payload["resource_line"] = line_name
        line_payload["resource"] = str(line_payload.get("resource") or line_name)
        for task_name in task_ids:
            mapping[task_name] = line_payload
    return mapping


def _task_set_resource_map(naturebench_root: Path) -> dict[str, str]:
    task_set_root = naturebench_root / "task-set"
    mapping: dict[str, str] = {}
    for file_name, resource in (
        ("cpu.txt", "cpu"),
        ("gpu_low.txt", "gpu_low"),
        ("gpu_high.txt", "gpu_high"),
    ):
        for task_name in _task_set_ids(task_set_root / file_name):
            mapping[task_name] = resource
    return mapping


def _legacy_resource_line(resource: str) -> dict[str, Any]:
    if resource == "cpu":
        return {"resource": "cpu", "gpu_mode": "none"}
    return {"resource": resource}


def _line_value(
    line_spec: dict[str, Any],
    data_cfg: Any,
    key: str,
    default: Any = None,
) -> Any:
    if key in line_spec:
        return _plain(line_spec.get(key))
    return _plain(data_cfg.get(key, default))


def _resource_from_metadata(metadata: dict[str, Any]) -> str:
    for key in ("resource", "compute_resource", "cpu_gpu"):
        value = metadata.get(key)
        if value:
            return str(value).lower()
    requirements = metadata.get("compute_resource_requirements") or {}
    gpu_requirement = requirements.get("gpu_compute") or {}
    severity = str(gpu_requirement.get("severity") or "").strip().lower()
    if severity and severity not in {"none", "no", "false", "0"}:
        return "gpu"
    return "cpu"


def _normalize_scm_roots(value: Any) -> list[str]:
    plain = _plain(value)
    if plain is None:
        return []
    if isinstance(plain, str):
        return [part for part in plain.split(":") if part]
    if isinstance(plain, list):
        return [str(part) for part in plain if str(part)]
    return [str(plain)]


def _remote_task_records(data_cfg: Any, task_names: list[str]) -> dict[str, dict[str, Any]]:
    scm_host = data_cfg.get("scm_host")
    if not scm_host:
        return {}
    scm_roots = _normalize_scm_roots(data_cfg.get("scm_task_roots"))
    script = r"""
import json
from pathlib import Path
import sys

payload = json.loads(sys.stdin.read())
records = {}
for task_name in payload["task_names"]:
    for root in payload["roots"]:
        task_dir = Path(root) / task_name
        metadata_path = task_dir / "metadata.json"
        if not metadata_path.exists():
            continue

        def read_text(relative_path):
            path = task_dir / relative_path
            return path.read_text(encoding="utf-8") if path.exists() else ""

        records[task_name] = {
            "task_dir": str(task_dir),
            "problem_dir": str(task_dir / "problem"),
            "data_dir": str(task_dir / "problem" / "data"),
            "metadata": json.loads(metadata_path.read_text(encoding="utf-8")),
            "task_description": read_text("problem/README.md"),
            "data_description": read_text("problem/data_description.md"),
        }
        break
print(json.dumps(records, ensure_ascii=False))
"""
    process = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            str(scm_host),
            f"python3 -c {shlex.quote(script)}",
        ],
        input=json.dumps({"task_names": task_names, "roots": scm_roots}),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"Failed to read NatureBench task metadata from SCM {scm_host}: "
            f"{process.stderr[-1000:] or process.stdout[-1000:]}"
        )
    return dict(json.loads(process.stdout or "{}"))


def _public_user_prompt() -> str:
    return (
        "You are solving a NatureBench scientific coding task. Generate one "
        "self-contained Python solution. Read visible data from DATA_DIR and write "
        "all required outputs under OUTPUT_DIR. The evaluator will call the "
        "NatureBench eval service and report aggregate_improvement; higher is "
        "better, 0 matches the paper/SOTA baseline, and values above 0.1 satisfy "
        "the official Surpass-SOTA threshold. Do not access "
        "hidden evaluation files, ground truth, external web resources, or external "
        "LLM services."
    )


def _task_family_guidance(metadata: dict[str, Any]) -> str:
    domain = str(metadata.get("domain") or "").strip().lower()
    guidance = [
        "Use only visible task data. Infer train/test roles, row alignment, and required output schema from the README, data description, and files rather than filename assumptions.",
        "Start with a compact, reproducible baseline and verify output rows, columns, ordering, and missing-value handling before spending compute on a larger model.",
        "Do not install packages or download models. If a dependency is unavailable, switch to a listed package or a standard-library fallback.",
        "Estimate memory before materializing dense joins, all-pairs features, or large cross-validation grids; prefer bounded folds, chunking, sparse arrays, and a small number of models.",
    ]
    domain_guidance = {
        "cellular omics": "Treat assays, tissues, and dataset instances independently unless visible metadata establishes a shared target; validate matrix orientation and sparse layout for each input.",
        "protein biology": "Respect assay and variant grouping, use only task-provided sequence or structure inputs, and avoid assuming external pretrained weights are present.",
        "physical modeling": "Use the task-provided observations rather than synthetic simulator data, and keep feature construction and validation bounded for large temporal or structural inputs.",
        "molecular design": "Keep classification and regression datasets separate, validate molecule identifiers or structures from visible fields, and choose metrics and splits per dataset rather than forcing one global model.",
        "biomedical modeling": "Inspect coordinate, image, point-cloud, and metadata conventions before modeling; preserve required identifier and coordinate alignment in the output.",
        "relational reasoning": "Preserve treatment, outcome, and group semantics from the visible schema, avoid target leakage, and train only on the labels explicitly provided for training.",
    }
    if domain in domain_guidance:
        guidance.append(domain_guidance[domain])
    return "\n- ".join([guidance[0], *guidance[1:]])


def build_tasks(config_path: Path, output_root: Path | None = None) -> None:
    cfg = OmegaConf.load(config_path)
    data_cfg = cfg.data
    naturebench_root = Path(str(data_cfg.naturebench_root)).expanduser().resolve()
    tasks_root = Path(str(data_cfg.get("tasks_root") or naturebench_root / "tasks"))
    if not tasks_root.is_absolute():
        tasks_root = naturebench_root / tasks_root

    base_dir = output_root.resolve() if output_root is not None else Path(__file__).resolve().parent
    base_dir.mkdir(parents=True, exist_ok=True)

    task_names = _task_names_from_cfg(data_cfg, tasks_root)
    resource_line_map = _resource_line_map(data_cfg, naturebench_root)
    resource_map = _task_set_resource_map(naturebench_root)
    missing_local = [
        task_name
        for task_name in task_names
        if not (tasks_root / task_name / "metadata.json").exists()
    ]
    remote_records = _remote_task_records(data_cfg, missing_local) if missing_local else {}

    for task_name in task_names:
        task_package_dir = tasks_root / task_name
        problem_dir = task_package_dir / "problem"
        data_dir = problem_dir / "data"
        metadata_path = task_package_dir / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            task_description = _read_optional(problem_dir / "README.md")
            data_description = _read_optional(problem_dir / "data_description.md")
        elif task_name in remote_records:
            remote_record = remote_records[task_name]
            task_package_dir = Path(str(remote_record["task_dir"]))
            problem_dir = Path(str(remote_record["problem_dir"]))
            data_dir = Path(str(remote_record["data_dir"]))
            metadata = dict(remote_record["metadata"])
            task_description = str(remote_record.get("task_description") or "")
            data_description = str(remote_record.get("data_description") or "")
        else:
            raise FileNotFoundError(
                f"Could not find NatureBench task package for {task_name} under "
                f"{tasks_root} or configured SCM task roots."
            )

        task_output_dir = base_dir / _sanitize_task_name(task_name)
        task_output_dir.mkdir(parents=True, exist_ok=True)
        interpreter_data_dir = data_dir
        if not data_dir.exists():
            interpreter_data_dir = task_output_dir / "data_preview_empty"
            interpreter_data_dir.mkdir(parents=True, exist_ok=True)
        line_spec = resource_line_map.get(
            task_name,
            _legacy_resource_line(
                resource_map.get(task_name, _resource_from_metadata(metadata))
            ),
        )
        visible_data_analysis = _visible_data_analysis_for_task(data_cfg, task_name)
        task_payload = {
            "benchmark": "naturebench",
            "uuid": task_name,
            "task_name": task_name,
            "source": "NatureBench",
            "domain": metadata.get("domain"),
            "resource": str(line_spec.get("resource") or "cpu"),
            "task_dir": str(task_package_dir),
            "problem_dir": str(problem_dir),
            "data_dir": str(data_dir),
            "interpreter_data_dir": str(interpreter_data_dir),
            "higher_is_better": True,
            "operator_family": "naturebench",
            "solver_defaults": "naturebench/evo.yaml",
            "execution_mode": str(data_cfg.get("execution_mode", "docker")),
            "eval_service_url": str(data_cfg.get("eval_service_url", "http://127.0.0.1:8321")),
            "batch_name": str(data_cfg.get("batch_name", "airaevo-naturebench")),
            "workspace_root": data_cfg.get("workspace_root"),
            "docker_image": data_cfg.get("docker_image"),
            "scm_host": _line_value(line_spec, data_cfg, "scm_host"),
            "scm_workspace_root": _line_value(line_spec, data_cfg, "scm_workspace_root"),
            "scm_task_roots": _plain(data_cfg.get("scm_task_roots")),
            "scm_eval_service_url": _line_value(line_spec, data_cfg, "scm_eval_service_url"),
            "scm_container_eval_service_url": _line_value(
                line_spec,
                data_cfg,
                "scm_container_eval_service_url",
            ),
            "task_description": task_description,
            "data_description": data_description,
            "visible_data_analysis": visible_data_analysis,
            "task_family_guidance": _task_family_guidance(metadata),
            "public_system_prompt": str(data_cfg.get("public_system_prompt", "")),
            "public_user_prompt": str(data_cfg.get("public_user_prompt", _public_user_prompt())),
            "metadata": metadata,
            "submit_repeats": int(
                1
                if data_cfg.get("submit_repeats", None) is None
                else data_cfg.get("submit_repeats")
            ),
            "eval_timeout": int(data_cfg.get("eval_timeout", 3600) or 3600),
            "execution_timeout": int(data_cfg.get("execution_timeout", 600) or 600),
            "candidate_preflight": bool(data_cfg.get("candidate_preflight", True)),
            "candidate_preflight_imports": bool(
                data_cfg.get("candidate_preflight_imports", True)
            ),
            "candidate_preflight_timeout": int(
                data_cfg.get("candidate_preflight_timeout", 45) or 45
            ),
            "scm_gpu_wait_timeout": _plain(data_cfg.get("scm_gpu_wait_timeout")),
        }
        if line_spec.get("resource_line"):
            task_payload["resource_line"] = str(line_spec["resource_line"])
        gpu_mode = line_spec.get("gpu_mode")
        if gpu_mode is not None:
            task_payload["gpu_mode"] = str(gpu_mode)
        if str(gpu_mode or "").lower() == "exclusive":
            for key in ("gpu_devices", "gpu_pool_file"):
                value = _line_value(line_spec, data_cfg, key)
                if value is not None:
                    task_payload[key] = value
        elif str(gpu_mode or "").lower() == "shared":
            for key in (
                "shared_gpu_device",
                "shared_gpu_slots",
                "shared_gpu_pool_file",
            ):
                value = _line_value(line_spec, data_cfg, key)
                if value is not None:
                    task_payload[key] = value
        elif not gpu_mode:
            gpu_devices = _plain(data_cfg.get("gpu_devices"))
            if gpu_devices is not None:
                task_payload["gpu_devices"] = gpu_devices

        (task_output_dir / "config.yaml").write_text(
            yaml.safe_dump(task_payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        (task_output_dir / "task_metadata.json").write_text(
            json.dumps(task_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AIRA-Evo NatureBench tasks")
    parser.add_argument("--config", required=True, help="Path to build config yaml")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Directory where task config directories should be written.",
    )
    args = parser.parse_args()
    output_root = Path(args.output_root) if args.output_root else None
    build_tasks(Path(args.config), output_root=output_root)


if __name__ == "__main__":
    main()
