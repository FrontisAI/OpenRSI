from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_TASK_SKILL_HEADING = "Task-Specific Skill Reference:"
DEFAULT_TASK_SKILL_INTRO = (
    "The following task-specific skill may be useful. Use it as optional guidance "
    "while still following all benchmark instructions, reading data from DATA_DIR, "
    "and writing ./submission.csv."
)


@dataclass(frozen=True)
class TaskSkillGuidanceResult:
    task_name: str
    changed: bool
    skipped_existing: bool
    missing: bool


def load_task_skill_map(
    skills_dir: Path,
    *,
    recursive: bool = True,
) -> dict[str, str]:
    """Load task-specific skill Markdown files keyed by task name."""

    if not skills_dir.exists():
        raise FileNotFoundError(f"Task skill directory does not exist: {skills_dir}")
    if not skills_dir.is_dir():
        raise NotADirectoryError(f"Task skill path is not a directory: {skills_dir}")

    skill_map: dict[str, str] = {}
    for path in sorted(skills_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            skill_map[path.stem] = text

    if recursive:
        for path in sorted(skills_dir.rglob("*.md")):
            if path.parent == skills_dir or path.stem in skill_map:
                continue
            text = path.read_text(encoding="utf-8").strip()
            if text:
                skill_map[path.stem] = text

    return skill_map


class TaskSkillGuidanceInjector:
    def __init__(
        self,
        skill_map: dict[str, str],
        *,
        heading: str = DEFAULT_TASK_SKILL_HEADING,
        intro: str = DEFAULT_TASK_SKILL_INTRO,
        strict: bool = False,
    ) -> None:
        self.skill_map = dict(skill_map)
        self.heading = heading.strip() or DEFAULT_TASK_SKILL_HEADING
        self.intro = intro.strip()
        self.strict = bool(strict)

    def inject(
        self,
        public_user_prompt: str,
        task_name: str,
    ) -> tuple[str, TaskSkillGuidanceResult]:
        task_name = str(task_name).strip()
        skill_text = self.skill_map.get(task_name)
        if not skill_text:
            if self.strict:
                raise ValueError(f"Missing task skill guidance for task: {task_name}")
            return public_user_prompt, TaskSkillGuidanceResult(
                task_name=task_name,
                changed=False,
                skipped_existing=False,
                missing=True,
            )

        if self.heading in public_user_prompt:
            return public_user_prompt, TaskSkillGuidanceResult(
                task_name=task_name,
                changed=False,
                skipped_existing=True,
                missing=False,
            )

        parts = [self.heading]
        if self.intro:
            parts.append(self.intro)
        parts.append(skill_text.strip())
        block = "\n\n".join(parts)
        patched = f"{public_user_prompt.rstrip()}\n\n{block}\n"
        return patched, TaskSkillGuidanceResult(
            task_name=task_name,
            changed=True,
            skipped_existing=False,
            missing=False,
        )
