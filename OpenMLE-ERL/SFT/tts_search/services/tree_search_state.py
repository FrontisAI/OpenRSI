"""Persistence helpers for tree-search distillation state."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
NODE_GENERATED = "node_generated"
NODE_EVALUATED = "node_evaluated"


def utc_now_iso() -> str:
    """Return a stable UTC timestamp for persisted search records."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class SearchEvent:
    """Append-only event for one generated or evaluated tree-search node."""

    event_type: str
    task_id: str
    search_algorithm: str
    search_step: int
    program_id: str
    task_name: str | None = None
    parent_ids: list[str] = field(default_factory=list)
    generation_mode: str | None = None
    code_path: str | None = None
    code_sha256: str | None = None
    score: float | None = None
    reward: float | None = None
    fitness: float | None = None
    feedback_path: str | None = None
    raw_log_path: str | None = None
    accepted_by_rejection_policy: bool | None = None
    rejection_reason: str | None = None
    generated_count: int | None = None
    completed_count: int | None = None
    accepted_count: int | None = None
    stop_requested: bool = False
    timestamp: str = field(default_factory=utc_now_iso)
    schema_version: int = SCHEMA_VERSION
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.event_type not in {NODE_GENERATED, NODE_EVALUATED}:
            raise ValueError(f"unknown search event type: {self.event_type}")
        if self.search_step < 0:
            raise ValueError("search_step must be >= 0")
        if not self.program_id:
            raise ValueError("program_id is required")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        extra = payload.pop("extra", {}) or {}
        payload.update(extra)
        return {key: value for key, value in payload.items() if value is not None}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SearchEvent":
        known = set(cls.__dataclass_fields__) - {"extra"}
        values = {key: data[key] for key in known if key in data}
        values["extra"] = {key: value for key, value in data.items() if key not in known}
        return cls(**values)


@dataclass
class SearchState:
    """Current resumable state for one task's tree-search run."""

    task_id: str
    search_algorithm: str
    next_search_step: int
    task_name: str | None = None
    generated_count: int = 0
    completed_count: int = 0
    accepted_count: int = 0
    accepted_target: int | None = None
    max_generated: int | None = None
    stop_requested: bool = False
    program_ids: list[str] = field(default_factory=list)
    program_scores: dict[str, float | None] = field(default_factory=dict)
    program_fitness: dict[str, float | None] = field(default_factory=dict)
    program_code_paths: dict[str, str] = field(default_factory=dict)
    parent_map: dict[str, list[str]] = field(default_factory=dict)
    generation_modes: dict[str, str] = field(default_factory=dict)
    island_populations: list[list[str]] | None = None
    generation_buffer: dict[str, Any] | None = None
    journal_path: str | None = None
    event_log_path: str = "search_events.jsonl"
    updated_at: str = field(default_factory=utc_now_iso)
    schema_version: int = SCHEMA_VERSION
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        extra = payload.pop("extra", {}) or {}
        payload.update(extra)
        return {key: value for key, value in payload.items() if value is not None}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SearchState":
        known = set(cls.__dataclass_fields__) - {"extra"}
        values = {key: data[key] for key in known if key in data}
        values["extra"] = {key: value for key, value in data.items() if key not in known}
        return cls(**values)


@dataclass
class ReplaySummary:
    """Counters and indexes reconstructed from search_events.jsonl."""

    generated_program_ids: set[str] = field(default_factory=set)
    evaluated_program_ids: set[str] = field(default_factory=set)
    accepted_program_ids: set[str] = field(default_factory=set)
    generated_but_not_evaluated: list[str] = field(default_factory=list)
    program_ids: list[str] = field(default_factory=list)
    program_scores: dict[str, float | None] = field(default_factory=dict)
    program_fitness: dict[str, float | None] = field(default_factory=dict)
    program_code_paths: dict[str, str] = field(default_factory=dict)
    parent_map: dict[str, list[str]] = field(default_factory=dict)
    generation_modes: dict[str, str] = field(default_factory=dict)
    next_search_step: int = 0
    stop_requested: bool = False

    @property
    def generated_count(self) -> int:
        return len(self.generated_program_ids)

    @property
    def completed_count(self) -> int:
        return len(self.evaluated_program_ids)

    @property
    def accepted_count(self) -> int:
        return len(self.accepted_program_ids)


def append_search_event(event_log_path: Path, event: SearchEvent | dict[str, Any]) -> None:
    """Append one search event to a JSONL event log."""

    event_log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = event.to_dict() if isinstance(event, SearchEvent) else dict(event)
    with open(event_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_search_events(event_log_path: Path) -> list[SearchEvent]:
    if not event_log_path.exists():
        return []
    events: list[SearchEvent] = []
    with open(event_log_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(SearchEvent.from_dict(json.loads(line)))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {event_log_path}:{line_no}") from exc
    return events


def atomic_write_search_state(state_path: Path, state: SearchState | dict[str, Any]) -> None:
    """Atomically write the latest resumable search state."""

    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = state.to_dict() if isinstance(state, SearchState) else dict(state)
    payload["updated_at"] = payload.get("updated_at") or utc_now_iso()
    tmp_path = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    os.replace(tmp_path, state_path)


def load_search_state(state_path: Path) -> SearchState | None:
    if not state_path.exists():
        return None
    return SearchState.from_dict(json.loads(state_path.read_text(encoding="utf-8")))


def replay_search_events(events: Iterable[SearchEvent]) -> ReplaySummary:
    """Replay events with idempotent counters keyed by program_id."""

    summary = ReplaySummary()
    seen_program_order: set[str] = set()
    for event in events:
        program_id = event.program_id
        if program_id not in seen_program_order:
            seen_program_order.add(program_id)
            summary.program_ids.append(program_id)

        summary.next_search_step = max(summary.next_search_step, event.search_step + 1)
        summary.stop_requested = summary.stop_requested or bool(event.stop_requested)
        if event.parent_ids:
            summary.parent_map[program_id] = list(event.parent_ids)
        if event.generation_mode:
            summary.generation_modes[program_id] = str(event.generation_mode)
        if event.code_path:
            summary.program_code_paths[program_id] = str(event.code_path)

        if event.event_type == NODE_GENERATED:
            summary.generated_program_ids.add(program_id)
        elif event.event_type == NODE_EVALUATED:
            summary.generated_program_ids.add(program_id)
            summary.evaluated_program_ids.add(program_id)
            summary.program_scores[program_id] = event.score
            summary.program_fitness[program_id] = event.fitness
            if event.accepted_by_rejection_policy:
                summary.accepted_program_ids.add(program_id)

    summary.generated_but_not_evaluated = [
        program_id
        for program_id in summary.program_ids
        if program_id in summary.generated_program_ids
        and program_id not in summary.evaluated_program_ids
    ]
    return summary


def build_state_from_replay(
    *,
    task_id: str,
    search_algorithm: str,
    summary: ReplaySummary,
    task_name: str | None = None,
    accepted_target: int | None = None,
    max_generated: int | None = None,
    event_log_path: str = "search_events.jsonl",
    stop_requested: bool | None = None,
    island_populations: list[list[str]] | None = None,
    generation_buffer: dict[str, Any] | None = None,
    journal_path: str | None = None,
) -> SearchState:
    return SearchState(
        task_id=task_id,
        task_name=task_name,
        search_algorithm=search_algorithm,
        next_search_step=summary.next_search_step,
        generated_count=summary.generated_count,
        completed_count=summary.completed_count,
        accepted_count=summary.accepted_count,
        accepted_target=accepted_target,
        max_generated=max_generated,
        stop_requested=summary.stop_requested if stop_requested is None else stop_requested,
        program_ids=list(summary.program_ids),
        program_scores=dict(summary.program_scores),
        program_fitness=dict(summary.program_fitness),
        program_code_paths=dict(summary.program_code_paths),
        parent_map=dict(summary.parent_map),
        generation_modes=dict(summary.generation_modes),
        island_populations=island_populations,
        generation_buffer=generation_buffer,
        journal_path=journal_path,
        event_log_path=event_log_path,
    )


def validate_state_consistency(state: SearchState, summary: ReplaySummary) -> None:
    """Fail fast when persisted state and replayed events disagree on counters."""

    mismatches: list[str] = []
    if state.generated_count != summary.generated_count:
        mismatches.append(
            f"generated_count state={state.generated_count} replay={summary.generated_count}"
        )
    if state.completed_count != summary.completed_count:
        mismatches.append(
            f"completed_count state={state.completed_count} replay={summary.completed_count}"
        )
    if state.accepted_count != summary.accepted_count:
        mismatches.append(
            f"accepted_count state={state.accepted_count} replay={summary.accepted_count}"
        )
    if state.next_search_step < summary.next_search_step:
        mismatches.append(
            f"next_search_step state={state.next_search_step} replay={summary.next_search_step}"
        )
    if mismatches:
        raise ValueError("search state is inconsistent with events: " + "; ".join(mismatches))
