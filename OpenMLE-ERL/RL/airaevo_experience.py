"""AIRA-Evo experience utilities adapted for RL ProgramDatabase.

The original AIRA-Evo implementation stores state in Dojo Node/Journal objects.
The RL loop stores evaluated programs in SQLite, so this module keeps the same
selection and prompt-memory signals while operating on Program-like objects.
"""

from __future__ import annotations

import ast
import json
import math
import re
from typing import Any, Iterable

FAILED_STATUSES = {
    "buggy",
    "error",
    "failed",
    "failure",
    "invalid",
    "sandbox_error",
    "scoring_failed",
    "submission_missing",
    "timeout",
    "unknown",
}
SUCCESS_STATUSES = {"ok", "passed", "success", "successful", "valid"}
DEFAULT_PARENT_UTILITY_WEIGHTS = {"score": 1.0, "delta": 0.4, "novelty": 0.25}


def finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def compact_text(value: Any, max_chars: int = 240) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _metadata(program: Any) -> dict[str, Any]:
    metadata = getattr(program, "metadata", {}) or {}
    return dict(metadata) if isinstance(metadata, dict) else {}


def program_selection_score(program: Any) -> float | None:
    """Return the test-reward fitness signal used by RL AIRA-Evo selection."""
    metadata = _metadata(program)
    for key in (
        "metric_static_base_reward",
        "static_base_reward",
        "base_reward",
        "dynamic_base_reward",
        "reward",
    ):
        value = finite_float(metadata.get(key))
        if value is not None:
            return value
    value = finite_float(getattr(program, "base_reward", None))
    if value is not None:
        return value
    return finite_float(getattr(program, "reward", None))


def _status_text(program: Any) -> str:
    metadata = _metadata(program)
    for key in ("code_category", "hack_category", "status", "sandbox_status"):
        value = str(metadata.get(key) or "").strip()
        if value:
            return value
    payload = getattr(program, "payload", {}) or {}
    if isinstance(payload, dict):
        value = str(payload.get("status") or "").strip()
        if value:
            return value
    status_code = getattr(program, "status_code", None)
    return "valid" if status_code == 200 else "unknown"


def status_is_failure(status: str) -> bool:
    normalized = str(status or "").strip().lower()
    if normalized in FAILED_STATUSES:
        return True
    if normalized in {"hack", "hack_verify", "empty", "no_verify", "generation_abort", "generation_abort_group", "no_code"}:
        return True
    return False


def is_success_program(program: Any) -> bool:
    metadata = _metadata(program)
    if int(getattr(program, "hack", 0) or 0) != 0:
        return False
    if int(getattr(program, "status_code", 0) or 0) != 200:
        return False
    if status_is_failure(_status_text(program)):
        return False
    if metadata.get("generation_aborted"):
        return False
    return program_selection_score(program) is not None


def is_debug_candidate(program: Any) -> bool:
    score = program_selection_score(program)
    return (not is_success_program(program)) or score is None or score <= 0.0


def extract_imports(code: str) -> list[str]:
    imports: set[str] = set()
    try:
        tree = ast.parse(code or "")
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split(".", 1)[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".", 1)[0])
    except SyntaxError:
        for match in re.finditer(r"^\s*(?:import|from)\s+([A-Za-z_][\w]*)", code or "", flags=re.MULTILINE):
            imports.add(match.group(1))
    return sorted(imports)


def detect_method_family(code: str, imports: Iterable[str] | None = None) -> str:
    imports_set = {item.lower() for item in (imports or [])}
    text = (code or "").lower()
    families: list[str] = []

    def add(name: str) -> None:
        if name not in families:
            families.append(name)

    if {"lightgbm", "lgbm"} & imports_set or "lgbm" in text or "lightgbm" in text:
        add("lightgbm")
    if "xgboost" in imports_set or "xgb" in text or "xgboost" in text:
        add("xgboost")
    if "catboost" in imports_set or "catboost" in text:
        add("catboost")
    if "torch" in imports_set or "pytorch" in text or "nn.module" in text:
        add("pytorch")
    if {"tensorflow", "keras"} & imports_set or "tensorflow" in text or "keras" in text:
        add("tensorflow")
    if "sklearn" in imports_set or "scikit" in text:
        add("sklearn")
    if "transformers" in imports_set or "bert" in text or "tokenizer" in text:
        add("nlp_transformer")
    if {"cv2", "pil", "torchvision"} & imports_set or "image" in text:
        add("vision")
    if "ensemble" in text or "stacking" in text or "voting" in text:
        add("ensemble")
    if "cross_val" in text or "kfold" in text or "stratifiedkfold" in text:
        add("cv")
    if not families:
        return "general_ml"
    return "+".join(families[:3])


def parent_ids(program: Any) -> list[int]:
    ids: list[int] = []
    parent_id = getattr(program, "parent_id", None)
    if parent_id is not None:
        try:
            ids.append(int(parent_id))
        except (TypeError, ValueError):
            pass
    metadata = _metadata(program)
    for value in metadata.get("crossover_parent_ids") or []:
        try:
            pid = int(value)
        except (TypeError, ValueError):
            continue
        if pid not in ids:
            ids.append(pid)
    return ids


def _program_id(program: Any) -> int | None:
    value = getattr(program, "id", None)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _index(programs: Iterable[Any]) -> dict[int, Any]:
    result: dict[int, Any] = {}
    for program in programs:
        pid = _program_id(program)
        if pid is not None:
            result[pid] = program
    return result


def _positive_delta(child_score: float | None, parent_score: float | None, lower_is_better: bool) -> float | None:
    if child_score is None or parent_score is None:
        return None
    return float(parent_score - child_score if lower_is_better else child_score - parent_score)


def error_signature(program: Any) -> str | None:
    metadata = _metadata(program)
    for key in ("error_signature", "hack_reason", "generation_abort_reason"):
        value = compact_text(metadata.get(key), 160)
        if value:
            return value
    status = _status_text(program)
    status_code = getattr(program, "status_code", None)
    if status_is_failure(status):
        return f"{status}:{status_code}"
    payload = getattr(program, "payload", {}) or {}
    if isinstance(payload, dict):
        detail = payload.get("detail") or payload.get("error")
        if detail:
            return compact_text(detail, 160)
    return None


def _family_count_before(method_family: str, previous_cards: list[dict[str, Any]], node_id: str | None = None) -> int:
    count = 0
    for card in previous_cards:
        if node_id and str(card.get("node_id")) == str(node_id):
            continue
        if str(card.get("method_family_auto") or "unknown") == method_family:
            count += 1
    return count


def build_experience_card(
    program: Any,
    *,
    all_programs: Iterable[Any] | None = None,
    lower_is_better: bool = False,
) -> dict[str, Any]:
    programs = list(all_programs or [])
    by_id = _index(programs)
    previous_cards = []
    for candidate in programs:
        metadata = _metadata(candidate)
        card = metadata.get("airaevo_experience_card")
        if isinstance(card, dict):
            previous_cards.append(card)

    code = str(getattr(program, "code", "") or "")
    imports = extract_imports(code)
    method_family = detect_method_family(code, imports)
    node_id = str(getattr(program, "id", "") or "")
    family_count_before = _family_count_before(method_family, previous_cards, node_id=node_id)
    score = program_selection_score(program)
    parent_scores = []
    for pid in parent_ids(program):
        parent = by_id.get(pid)
        if parent is not None:
            parent_score = program_selection_score(parent)
            if parent_score is not None:
                parent_scores.append(parent_score)
    best_parent_score = None
    if parent_scores:
        best_parent_score = min(parent_scores) if lower_is_better else max(parent_scores)

    metadata = _metadata(program)
    rich_summary = metadata.get("airaevo_rich_summary")
    status = _status_text(program)
    card = {
        "schema_version": 1,
        "node_id": node_id,
        "step_id": int(getattr(program, "id", 0) or 0),
        "operator": str(getattr(program, "generation_mode", "") or metadata.get("generation_mode") or ""),
        "parents": parent_ids(program),
        "parent_node_ids": [str(pid) for pid in parent_ids(program)],
        "generation_id": int(metadata.get("airaevo_generation_id") or 0),
        "score": getattr(program, "score", None),
        "fitness": score,
        "reward": getattr(program, "reward", None),
        "base_reward": getattr(program, "base_reward", None),
        "status": status,
        "status_code": getattr(program, "status_code", None),
        "is_buggy": not is_success_program(program),
        "sandbox_time_used": getattr(program, "running_time", None),
        "imports": imports,
        "method_family_auto": method_family,
        "family_count_before": family_count_before,
        "is_new_direction": family_count_before == 0,
        "novelty_score": 1.0 / math.sqrt(1.0 + family_count_before),
        "delta_vs_parent": _positive_delta(score, best_parent_score, lower_is_better),
        "error_signature": error_signature(program),
        "rank": None,
        "current_best": None,
        "selection_utility": None,
        "plan": compact_text(getattr(program, "raw_text", ""), 500),
        "analysis": compact_text(metadata.get("clear_run_log") or metadata.get("result_info"), 500),
    }
    if isinstance(rich_summary, dict):
        card["rich_summary"] = rich_summary
    return card


def _card(program: Any, all_programs: list[Any]) -> dict[str, Any]:
    metadata = _metadata(program)
    card = metadata.get("airaevo_experience_card")
    if isinstance(card, dict):
        return card
    return build_experience_card(program, all_programs=all_programs)


def _normalize_values(values: list[float | None], lower_is_better: bool) -> list[float]:
    finite_values = [value for value in values if value is not None and math.isfinite(value)]
    if not finite_values:
        return [0.0 for _ in values]
    min_value = min(finite_values)
    max_value = max(finite_values)
    if min_value == max_value:
        return [0.5 if value is not None else 0.0 for value in values]
    normalized = []
    for value in values:
        if value is None or not math.isfinite(value):
            normalized.append(0.0)
        elif lower_is_better:
            normalized.append((max_value - value) / (max_value - min_value))
        else:
            normalized.append((value - min_value) / (max_value - min_value))
    return [float(max(0.0, min(1.0, value))) for value in normalized]


def _normalize_positive(values: list[float | None]) -> list[float]:
    positives = [max(0.0, value or 0.0) for value in values]
    max_value = max(positives or [0.0])
    if max_value <= 0.0:
        return [0.0 for _ in positives]
    return [float(value / max_value) for value in positives]


def softmax(values: list[float], temperature: float = 1.0) -> list[float]:
    if not values:
        return []
    temperature = max(float(temperature or 1.0), 1e-8)
    finite_values = [value if math.isfinite(value) else 0.0 for value in values]
    max_value = max(finite_values)
    exp_values = [math.exp((value - max_value) / temperature) for value in finite_values]
    total = sum(exp_values)
    if total <= 0.0 or not math.isfinite(total):
        return [1.0 / len(values) for _ in values]
    return [float(value / total) for value in exp_values]


def compute_parent_utilities(
    candidates: list[Any],
    *,
    all_programs: Iterable[Any],
    lower_is_better: bool = False,
    temperature: float = 1.0,
    weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    weights = {**DEFAULT_PARENT_UTILITY_WEIGHTS, **dict(weights or {})}
    programs = list(all_programs or [])
    by_id = _index(programs)
    scores = [program_selection_score(program) for program in candidates]
    score_components = _normalize_values(scores, lower_is_better=lower_is_better)
    deltas: list[float | None] = []
    families: list[str] = []
    previous_cards = [_card(program, programs) for program in programs]
    for idx, program in enumerate(candidates):
        card = _card(program, programs)
        families.append(str(card.get("method_family_auto") or "unknown"))
        parent_scores = []
        for pid in parent_ids(program):
            parent = by_id.get(pid)
            if parent is not None:
                parent_score = program_selection_score(parent)
                if parent_score is not None:
                    parent_scores.append(parent_score)
        if parent_scores:
            best_parent = min(parent_scores) if lower_is_better else max(parent_scores)
            deltas.append(_positive_delta(scores[idx], best_parent, lower_is_better))
        else:
            deltas.append(0.0)
    delta_components = _normalize_positive(deltas)
    utilities: list[dict[str, Any]] = []
    for idx, program in enumerate(candidates):
        pid = _program_id(program)
        family = families[idx]
        family_count = _family_count_before(family, previous_cards, node_id=str(pid or ""))
        novelty = 1.0 / math.sqrt(1.0 + family_count)
        utility = (
            float(weights["score"]) * score_components[idx]
            + float(weights["delta"]) * delta_components[idx]
            + float(weights["novelty"]) * novelty
        )
        utilities.append(
            {
                "node_id": pid,
                "method_family_auto": family,
                "fitness": scores[idx],
                "score_source": "rl_test_reward",
                "score_component": score_components[idx],
                "delta_component": delta_components[idx],
                "novelty_component": novelty,
                "utility": utility,
                "probability": 0.0,
            }
        )
    probabilities = softmax([item["utility"] for item in utilities], temperature=temperature)
    for item, probability in zip(utilities, probabilities):
        item["probability"] = probability
    return utilities


def _sort_programs(programs: Iterable[Any], lower_is_better: bool = False) -> list[Any]:
    def key(program: Any) -> tuple[float, int]:
        score = program_selection_score(program)
        score_value = score if score is not None else (float("inf") if lower_is_better else float("-inf"))
        pid = _program_id(program) or 0
        return (score_value, pid)

    return sorted(programs, key=key, reverse=not lower_is_better)


def _recent_ancestors(program: Any, all_programs: list[Any], max_items: int) -> list[Any]:
    by_id = _index(all_programs)
    result: list[Any] = []
    current = program
    seen: set[int] = set()
    while current is not None and len(result) < max_items:
        pids = parent_ids(current)
        if not pids:
            break
        parent = by_id.get(pids[0])
        if parent is None:
            break
        parent_pid = _program_id(parent)
        if parent_pid is None or parent_pid in seen:
            break
        seen.add(parent_pid)
        result.append(parent)
        current = parent
    return result


def _children_of(parent: Any, all_programs: list[Any]) -> list[Any]:
    pid = _program_id(parent)
    if pid is None:
        return []
    return [program for program in all_programs if pid in parent_ids(program) and _program_id(program) != pid]


def build_strategy_board(programs: list[Any], lower_is_better: bool = False) -> dict[str, Any]:
    cards = [_card(program, programs) for program in programs]
    valid_cards = [card for card in cards if finite_float(card.get("fitness")) is not None and not card.get("is_buggy")]
    valid_cards = sorted(valid_cards, key=lambda card: finite_float(card.get("fitness")) or 0.0, reverse=not lower_is_better)
    best_card = valid_cards[0] if valid_cards else None
    family_counts: dict[str, int] = {}
    repeated_errors: dict[str, int] = {}
    for card in cards:
        family = str(card.get("method_family_auto") or "unknown")
        family_counts[family] = family_counts.get(family, 0) + 1
        signature = str(card.get("error_signature") or "")
        if signature:
            repeated_errors[signature] = repeated_errors.get(signature, 0) + 1
    underexplored = [family for family, count in sorted(family_counts.items(), key=lambda item: item[1])[:3]]
    return {
        "best_node": best_card.get("node_id") if best_card else None,
        "best_score": best_card.get("fitness") if best_card else None,
        "current_best_family": best_card.get("method_family_auto") if best_card else None,
        "family_counts": family_counts,
        "underexplored_families": underexplored,
        "repeated_errors": repeated_errors,
    }


def _format_value(value: Any) -> str:
    numeric = finite_float(value)
    if numeric is not None:
        return f"{numeric:.6g}"
    if value is None:
        return "n/a"
    return str(value)


def _rich_summary_lines(prefix: str, card: dict[str, Any]) -> list[str]:
    rich_summary = card.get("rich_summary")
    if isinstance(rich_summary, dict):
        lines = []
        method = compact_text(rich_summary.get("method_overview"), 420)
        parent_exp = compact_text(rich_summary.get("parent_comparison_experience"), 420)
        if method:
            lines.append(f"- {prefix}_method_overview: {method}")
        if parent_exp:
            lines.append(f"- {prefix}_parent_comparison_experience: {parent_exp}")
        if lines:
            return lines
    analysis = compact_text(card.get("analysis"), 360)
    return [f"- {prefix}_legacy_analysis: {analysis}"] if analysis else []


def _memory_node_lines(prefix: str, program: Any, all_programs: list[Any]) -> list[str]:
    card = _card(program, all_programs)
    lines = [
        (
            f"- {prefix}_node_id: {card.get('node_id')} "
            f"(family={card.get('method_family_auto') or 'unknown'}, "
            f"score={_format_value(card.get('fitness'))}, "
            f"delta_vs_parent={_format_value(card.get('delta_vs_parent'))}, "
            f"runtime_seconds={_format_value(card.get('sandbox_time_used'))})"
        ),
        f"- {prefix}_current_best: {bool(card.get('current_best', False))}",
        f"- {prefix}_is_new_direction: {bool(card.get('is_new_direction', False))}",
    ]
    lines.extend(_rich_summary_lines(prefix, card))
    return lines


def _append_section(lines: list[str], title: str, prefix: str, programs: list[Any], all_programs: list[Any]) -> None:
    if not programs:
        return
    lines.extend(["", title + ":"])
    for index, program in enumerate(programs, start=1):
        item_prefix = prefix if len(programs) == 1 else f"{prefix}_{index}"
        lines.extend(_memory_node_lines(item_prefix, program, all_programs))


def build_operator_experience_memory(
    operator: str,
    parent_programs: list[Any],
    *,
    all_programs: Iterable[Any],
    current_program: Any | None = None,
    lower_is_better: bool = False,
    max_related_cards: int = 3,
    ancestor_k: int | None = None,
    sibling_k: int | None = None,
) -> str:
    programs = list(all_programs or [])
    operator = str(operator or "").lower()
    board = build_strategy_board(programs, lower_is_better=lower_is_better)
    cap = max(1, int(max_related_cards or 3))

    if operator == "draft":
        if not programs:
            return ""
        recent = list(reversed(programs))[:cap]
        top = _sort_programs([program for program in programs if is_success_program(program)], lower_is_better)[:cap]
        lines = [
            "Targeted Memory Context for DRAFT",
            "",
            "Use this memory as evidence about previously explored ideas. Propose a meaningfully different solution direction.",
            "",
            "Related board stats:",
            f"- current_best_node: {board.get('best_node') or 'n/a'}",
            f"- current_best_score: {_format_value(board.get('best_score'))}",
            f"- current_best_family: {board.get('current_best_family') or 'unknown'}",
            f"- underexplored_families: {', '.join(board.get('underexplored_families') or []) or 'none'}",
        ]
        _append_section(lines, "Best previous memories", "best", top, programs)
        _append_section(lines, "Recent previous memories", "recent", recent, programs)
        return "\n".join(lines)

    if operator == "improve" and parent_programs:
        parent = parent_programs[0]
        ancestors = _recent_ancestors(parent, programs, 3 if ancestor_k is None else max(0, int(ancestor_k)))
        siblings = _sort_programs(_children_of(parent, programs), lower_is_better)[: 3 if sibling_k is None else max(0, int(sibling_k))]
        parent_card = _card(parent, programs)
        parent_family = str(parent_card.get("method_family_auto") or "unknown")
        lines = [
            "Targeted Memory Context for IMPROVE",
            "",
            "Use this memory as evidence about the selected parent, its recent evolution path, and nearby sibling attempts. Do not treat it as a complete history.",
        ]
        _append_section(lines, "Selected parent memory", "parent", [parent], programs)
        _append_section(lines, "Vertical ancestor memory (recent evolution path)", "ancestor", ancestors, programs)
        _append_section(lines, "Horizontal sibling memory (nearby alternative attempts)", "sibling", siblings, programs)
        lines.extend([
            "",
            "Related board stats:",
            f"- current_best_node: {board.get('best_node') or 'n/a'}",
            f"- current_best_score: {_format_value(board.get('best_score'))}",
            f"- current_best_family: {board.get('current_best_family') or 'unknown'}",
            f"- parent_family: {parent_family}",
            f"- underexplored_families: {', '.join(board.get('underexplored_families') or []) or 'none'}",
        ])
        return "\n".join(lines)

    if operator == "crossover" and len(parent_programs) >= 2:
        p1, p2 = parent_programs[:2]
        c1 = _card(p1, programs)
        c2 = _card(p2, programs)
        family_a = str(c1.get("method_family_auto") or "unknown")
        family_b = str(c2.get("method_family_auto") or "unknown")
        lines = [
            "Targeted Memory Context for CROSSOVER",
            "",
            "Use this memory to identify compatible strengths and conflicts between the two selected parent branches. Do not mechanically merge all code.",
            "",
            "Selected parent memories:",
        ]
        lines.extend(_memory_node_lines("parent_1", p1, programs))
        lines.extend(_memory_node_lines("parent_2", p2, programs))
        _append_section(lines, "Parent 1 vertical ancestor memory", "parent_1_ancestor", _recent_ancestors(p1, programs, 2), programs)
        _append_section(lines, "Parent 2 vertical ancestor memory", "parent_2_ancestor", _recent_ancestors(p2, programs, 2), programs)
        lines.extend([
            "",
            "Complementarity:",
            f"- family_complementarity: {'different_method_families' if family_a != family_b else 'same_method_family'}",
            f"- crossover_hint: combine {family_a} with {family_b}",
        ])
        return "\n".join(lines)

    if operator == "debug":
        focus = current_program or (parent_programs[0] if parent_programs else None)
        signature = error_signature(focus) if focus is not None else None
        related = []
        recent = []
        for candidate in reversed(programs):
            if focus is not None and _program_id(candidate) == _program_id(focus):
                continue
            if signature and error_signature(candidate) == signature:
                related.append(candidate)
            else:
                recent.append(candidate)
        lines = [
            "Targeted Memory Context for DEBUG",
            "",
            "Current error:",
            f"- current_error_signature: {signature or 'unknown'}",
        ]
        if focus is not None:
            _append_section(lines, "Current buggy node memory", "current_buggy", [focus], programs)
        _append_section(lines, "Related previous debug/error memories", "related_error", (related + recent)[:cap], programs)
        return "\n".join(lines)

    return ""


def parse_json_object(text: str) -> dict[str, Any] | None:
    text = str(text or "").strip()
    if not text:
        return None
    candidates = [text]
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if match:
        candidates.insert(0, match.group(1))
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None
