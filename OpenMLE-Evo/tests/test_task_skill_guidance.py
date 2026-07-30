from __future__ import annotations

import pandas as pd
import pytest
from omegaconf import OmegaConf
from scripts.evaluate_airaevo import _maybe_patch_eval_data_with_task_skill_guidance
from tts_search.task_skill_guidance import (
    TaskSkillGuidanceInjector,
    load_task_skill_map,
)


def test_load_task_skill_map_uses_direct_files_before_recursive_files(tmp_path):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "task-a.md").write_text("direct skill", encoding="utf-8")
    nested_dir = skills_dir / "unlite"
    nested_dir.mkdir()
    (nested_dir / "task-a.md").write_text("nested duplicate", encoding="utf-8")
    (nested_dir / "task-b.md").write_text("nested skill", encoding="utf-8")

    skill_map = load_task_skill_map(skills_dir)

    assert skill_map == {
        "task-a": "direct skill",
        "task-b": "nested skill",
    }


def test_task_skill_guidance_injects_and_skips_existing_heading():
    injector = TaskSkillGuidanceInjector(
        {"task-a": "# Skill: task-a\n\nUse folds."},
        heading="Task-Specific Skill Reference:",
        intro="Optional skill guidance.",
    )

    patched, result = injector.inject("Output Instructions:\nWrite code.", "task-a")

    assert result.changed is True
    assert result.missing is False
    assert "Task-Specific Skill Reference:" in patched
    assert "Optional skill guidance." in patched
    assert "# Skill: task-a" in patched

    second, second_result = injector.inject(patched, "task-a")

    assert second == patched
    assert second_result.changed is False
    assert second_result.skipped_existing is True


def test_task_skill_guidance_missing_task_non_strict_and_strict():
    non_strict = TaskSkillGuidanceInjector({}, strict=False)

    prompt, result = non_strict.inject("base prompt", "missing-task")

    assert prompt == "base prompt"
    assert result.missing is True

    strict = TaskSkillGuidanceInjector({}, strict=True)
    with pytest.raises(ValueError, match="Missing task skill guidance"):
        strict.inject("base prompt", "missing-task")


def test_airaevo_task_skill_guidance_patches_eval_parquet(tmp_path):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "task-a.md").write_text("Use group-aware folds.", encoding="utf-8")
    source = tmp_path / "eval.parquet"
    hydra_dir = tmp_path / ".hydra"
    hydra_dir.mkdir()
    pd.DataFrame(
        [
            {
                "prompt": "Task body\n\n**FINAL OUTPUT**: write submission",
                "metadata": {"task_name": "task-a"},
            },
            {
                "prompt": "Task body",
                "metadata": {"task_name": "missing-task"},
            },
        ]
    ).to_parquet(source, index=False)
    cfg = OmegaConf.create(
        {
            "data": {
                "input_key": "prompt",
                "metadata_key": "metadata",
                "task_skill_guidance": {
                    "enabled": True,
                    "skills_dir": str(skills_dir),
                    "recursive": True,
                    "heading": "Task-Specific Skill Reference:",
                    "intro": "Optional skill guidance.",
                    "strict": False,
                },
            }
        }
    )

    output = _maybe_patch_eval_data_with_task_skill_guidance(
        cfg,
        eval_data_path=source,
        hydra_dir=hydra_dir,
    )
    patched = pd.read_parquet(output)

    assert output == hydra_dir / "eval_task_skill_guidance.parquet"
    assert patched.loc[0, "prompt"].count("Task-Specific Skill Reference:") == 1
    assert "Use group-aware folds." in patched.loc[0, "prompt"]
    assert patched.loc[1, "prompt"] == "Task body"
