from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

DEFAULT_PARENT_UTILITY_WEIGHTS = {
    "score": 1.0,
    "delta": 0.4,
    "novelty": 0.25,
    "official_score_missing_penalty": 2.0,
}
DEFAULT_PARENT_COMPONENT_NORMALIZATION = {
    "score": "minmax",
    "delta": "minmax",
    "hybrid_minmax_weight": 0.5,
}

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
}
SUCCESS_STATUSES = {"ok", "passed", "success", "successful", "valid"}


def _finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _metric_value(node: Any) -> float | None:
    metric = getattr(node, "metric", None)
    return _finite_float(getattr(metric, "value", None))


def _metric_info(node: Any) -> dict[str, Any]:
    info = getattr(getattr(node, "metric", None), "info", {}) or {}
    if isinstance(info, dict):
        return dict(info)
    return {}


def _official_status(node: Any) -> str:
    return _normalized_status(_metric_info(node).get("status"))


def _official_score_value(node: Any) -> float | None:
    info = _metric_info(node)
    score = _finite_float(info.get("score"))
    if score is not None:
        return score

    if "score" not in info and _status_indicates_success(info.get("status"), info.get("status_code")):
        return _metric_value(node)
    return None


def _has_official_score(node: Any) -> bool:
    if bool(getattr(node, "is_buggy", False)):
        return False
    info = _metric_info(node)
    if not _status_indicates_success(info.get("status"), info.get("status_code")):
        return False
    return _official_score_value(node) is not None


def _normalize_minmax_values(values: list[float | None], lower_is_better: bool) -> list[float]:
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


def _normalize_rank_values(values: list[float | None], lower_is_better: bool) -> list[float]:
    finite_items = [
        (idx, float(value))
        for idx, value in enumerate(values)
        if value is not None and math.isfinite(value)
    ]
    if not finite_items:
        return [0.0 for _ in values]
    if len(finite_items) == 1:
        normalized = [0.0 for _ in values]
        normalized[finite_items[0][0]] = 0.5
        return normalized

    finite_items.sort(key=lambda item: item[1], reverse=not lower_is_better)
    normalized = [0.0 for _ in values]
    index = 0
    while index < len(finite_items):
        end = index + 1
        while end < len(finite_items) and finite_items[end][1] == finite_items[index][1]:
            end += 1
        average_position = (index + end - 1) / 2.0
        component = 1.0 - average_position / (len(finite_items) - 1)
        for item_index in range(index, end):
            normalized[finite_items[item_index][0]] = float(max(0.0, min(1.0, component)))
        index = end
    return normalized


def _normalization_mode(value: Any, default: str = "minmax") -> str:
    mode = str(value or default).strip().lower()
    if mode in {"minmax", "linear"}:
        return "minmax"
    if mode in {"rank", "percentile"}:
        return "rank"
    if mode == "hybrid":
        return "hybrid"
    return default


def _hybrid_minmax_weight(value: Any, default: float = 0.5) -> float:
    try:
        weight = float(value)
    except (TypeError, ValueError):
        weight = default
    return float(max(0.0, min(1.0, weight)))


def _component_normalization_config(config: dict[str, Any] | None) -> dict[str, Any]:
    config = {**DEFAULT_PARENT_COMPONENT_NORMALIZATION, **dict(config or {})}
    return {
        "score": _normalization_mode(config.get("score"), "minmax"),
        "delta": _normalization_mode(config.get("delta"), "minmax"),
        "hybrid_minmax_weight": _hybrid_minmax_weight(config.get("hybrid_minmax_weight"), 0.5),
    }


def _normalize_values(
    values: list[float | None],
    lower_is_better: bool,
    *,
    mode: str = "minmax",
    hybrid_minmax_weight: float = 0.5,
) -> list[float]:
    mode = _normalization_mode(mode)
    minmax_values = _normalize_minmax_values(values, lower_is_better)
    if mode == "minmax":
        return minmax_values

    rank_values = _normalize_rank_values(values, lower_is_better)
    if mode == "rank":
        return rank_values

    weight = _hybrid_minmax_weight(hybrid_minmax_weight)
    return [
        float(weight * minmax_value + (1.0 - weight) * rank_value)
        for minmax_value, rank_value in zip(minmax_values, rank_values)
    ]


def _normalize_positive_deltas(
    positive_deltas: list[float],
    *,
    mode: str = "minmax",
    hybrid_minmax_weight: float = 0.5,
) -> list[float]:
    max_positive_delta = max([delta for delta in positive_deltas if delta > 0.0] or [0.0])
    minmax_values = [
        float(delta / max_positive_delta) if max_positive_delta > 0.0 and delta > 0.0 else 0.0
        for delta in positive_deltas
    ]
    mode = _normalization_mode(mode)
    if mode == "minmax":
        return minmax_values

    rank_values = _normalize_rank_values(positive_deltas, lower_is_better=False)
    rank_values = [
        float(rank_value) if delta > 0.0 else 0.0
        for rank_value, delta in zip(rank_values, positive_deltas)
    ]
    if mode == "rank":
        return rank_values

    weight = _hybrid_minmax_weight(hybrid_minmax_weight)
    return [
        float(weight * minmax_value + (1.0 - weight) * rank_value)
        for minmax_value, rank_value in zip(minmax_values, rank_values)
    ]


def _positive_delta(child_score: float | None, parent_score: float | None, lower_is_better: bool) -> float | None:
    if child_score is None or parent_score is None:
        return None
    delta = parent_score - child_score if lower_is_better else child_score - parent_score
    return float(delta)


def _parent_steps(node: Any) -> list[int | str]:
    parent_ids = []
    for parent in list(getattr(node, "parents", None) or []):
        parent_step = getattr(parent, "step", None)
        if parent_step is not None:
            step = int(parent_step)
            if step > 0:
                parent_ids.append(step - 1)
        else:
            parent_ids.append(str(getattr(parent, "id", "")))
    return parent_ids


def _status_is_failure(status: str) -> bool:
    return str(status or "").strip().lower() in FAILED_STATUSES


def _compact_text(value: Any, max_chars: int = 240) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _format_value(value: Any) -> str:
    numeric = _finite_float(value)
    if numeric is not None:
        return f"{numeric:.6g}"
    if value is None:
        return "n/a"
    return str(value)


def _format_list(values: Iterable[Any], *, empty: str = "none") -> str:
    items = [str(value) for value in values if str(value)]
    return ", ".join(items) if items else empty


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

    if any(token in text for token in ["votingclassifier", "votingregressor", "stacking", "blend", "ensemble"]):
        add("ensemble")
    if "catboost" in imports_set or "catboost" in text:
        add("catboost")
    if "lightgbm" in imports_set or re.search(r"\blgbm?\b", text) or "lightgbm" in text:
        add("lightgbm")
    if "xgboost" in imports_set or re.search(r"\bxgb\b", text) or "xgboost" in text:
        add("xgboost")
    if any(name in imports_set for name in ["transformers", "datasets", "tokenizers", "sentence_transformers"]):
        add("transformers")
    if any(name in imports_set for name in ["torch", "tensorflow", "keras"]) or any(
        token in text for token in ["neuralnetwork", "nn.module", "sequential("]
    ):
        add("neural_net")
    if any(name in imports_set for name in ["cv2", "torchvision", "timm", "albumentations", "PIL".lower()]):
        add("cv")
    if any(name in imports_set for name in ["nltk", "spacy", "sentencepiece", "textblob"]):
        add("nlp")
    if "sklearn" in imports_set and not families:
        add("sklearn")

    return "+".join(families) if families else "unknown"


def detect_error_signature(
    status: str | None,
    status_code: int | None,
    raw_run_log: str,
    clear_run_log: str,
) -> str | None:
    if _status_indicates_success(status, status_code):
        return None

    combined = " ".join([str(status or ""), str(raw_run_log or ""), str(clear_run_log or "")]).lower()
    if "timeout" in combined or "timed out" in combined:
        return "timeout"
    if "submission.csv" in combined and any(token in combined for token in ["missing", "not found", "no such file"]):
        return "submission_missing"
    if any(token in combined for token in ["modulenotfounderror", "importerror", "no module named"]):
        return "import_error"
    if any(token in combined for token in ["column", "columns"]) and any(
        token in combined for token in ["mismatch", "missing", "extra", "wrong"]
    ):
        return "column_mismatch"
    if any(token in combined for token in ["final validation score", "score parser", "metric parse", "scoring failed"]):
        return "metric_parse_failed"
    if status_code is not None and status_code >= 500:
        return "sandbox_error"
    if str(status or "").lower() in {"submission_missing", "scoring_failed", "timeout"}:
        return str(status).lower()
    return None


def _coerce_status_code(status_code: Any) -> int | None:
    if status_code is None:
        return None
    try:
        return int(status_code)
    except (TypeError, ValueError):
        return None


def _normalized_status(status: str | None) -> str:
    return str(status or "").strip().lower()


def _status_indicates_success(status: str | None, status_code: Any) -> bool:
    normalized_status = _normalized_status(status)
    if _status_is_failure(normalized_status):
        return False
    if normalized_status in SUCCESS_STATUSES:
        return True

    code = _coerce_status_code(status_code)
    return code is not None and 200 <= code < 400


def _card_indicates_failure(card: dict[str, Any]) -> bool:
    if bool(card.get("is_buggy", False)):
        return True

    if _status_is_failure(card.get("status")):
        return True

    code = _coerce_status_code(card.get("status_code"))
    return code is not None and code >= 400


def _family_count_before(method_family: str, previous_cards: list[dict[str, Any]], node_id: str | None = None) -> int:
    count = 0
    for card in previous_cards:
        if node_id and str(card.get("node_id")) == str(node_id):
            continue
        if str(card.get("method_family_auto") or "unknown") == method_family:
            count += 1
    return count


def build_experience_card(
    *,
    node: Any,
    step_index: int,
    generation_id: int,
    lower_is_better: bool,
    previous_cards: list[dict[str, Any]],
    step_stat: dict[str, Any],
    usage: dict[str, Any] | None = None,
    clear_run_log: str = "",
    raw_run_log: str = "",
) -> dict[str, Any]:
    usage = dict(usage or {})
    code = str(getattr(node, "code", "") or "")
    imports = extract_imports(code)
    method_family = detect_method_family(code, imports)
    node_id = str(getattr(node, "id", ""))
    family_count_before = _family_count_before(method_family, previous_cards, node_id=node_id)
    novelty_score = 1.0 / math.sqrt(1.0 + family_count_before)

    fitness = _metric_value(node)
    parent_scores = [_metric_value(parent) for parent in list(getattr(node, "parents", None) or [])]
    finite_parent_scores = [score for score in parent_scores if score is not None]
    if finite_parent_scores:
        parent_score = min(finite_parent_scores) if lower_is_better else max(finite_parent_scores)
    else:
        parent_score = None
    delta_vs_parent = _positive_delta(fitness, parent_score, lower_is_better)

    status = str(step_stat.get("status") or "")
    status_code = step_stat.get("status_code")
    card = {
        "schema_version": 1,
        "node_id": node_id,
        "step_id": int(step_index),
        "operator": str(step_stat.get("operator") or step_stat.get("mode") or ""),
        "parents": _parent_steps(node),
        "parent_node_ids": [str(getattr(parent, "id", "")) for parent in list(getattr(node, "parents", None) or [])],
        "generation_id": int(generation_id),
        "score": step_stat.get("score"),
        "fitness": fitness,
        "reward": step_stat.get("reward"),
        "status": status,
        "status_code": status_code,
        "is_buggy": bool(getattr(node, "is_buggy", False)),
        "sandbox_time_used": step_stat.get("sandbox_time_used"),
        "model_time_used": step_stat.get("model_time_used"),
        "model_plus_sandbox_time_used": step_stat.get("model_plus_sandbox_time_used"),
        "cost": float(usage.get("cost") or step_stat.get("cost") or 0.0),
        "prompt_tokens": int(usage.get("prompt_tokens") or step_stat.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or step_stat.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or step_stat.get("total_tokens") or 0),
        "imports": imports,
        "method_family_auto": method_family,
        "family_count_before": family_count_before,
        "is_new_direction": family_count_before == 0,
        "novelty_score": novelty_score,
        "delta_vs_parent": delta_vs_parent,
        "error_signature": detect_error_signature(status, status_code, raw_run_log, clear_run_log),
        "rank": None,
        "current_best": None,
        "selection_utility": None,
        "plan": str(getattr(node, "plan", "") or ""),
        "analysis": str(getattr(node, "analysis", "") or ""),
    }
    rich_summary = getattr(node, "rich_summary", None)
    if isinstance(rich_summary, dict):
        card["rich_summary"] = rich_summary
    return card


def build_strategy_board(cards: list[dict[str, Any]], lower_is_better: bool) -> dict[str, Any]:
    cards = list(cards)
    valid_cards = [
        card
        for card in cards
        if _finite_float(card.get("fitness")) is not None and not bool(card.get("is_buggy", False))
    ]
    sorted_valid_cards = sorted(
        valid_cards,
        key=lambda card: _finite_float(card.get("fitness")),
        reverse=not lower_is_better,
    )
    best_card = sorted_valid_cards[0] if sorted_valid_cards else None

    method_family_stats: dict[str, dict[str, Any]] = {}
    status_count: dict[str, int] = {}
    operator_counts: dict[str, int] = {}
    repeated_errors: dict[str, int] = {}
    score_history: list[dict[str, Any]] = []
    parent_graph: dict[str, list[Any]] = {}
    deltas: list[float] = []
    family_seen: dict[str, int] = {}
    novelty_by_node: dict[str, float] = {}

    for card in cards:
        family = str(card.get("method_family_auto") or "unknown")
        node_id = str(card.get("node_id") or "")
        novelty_by_node[node_id] = 1.0 / math.sqrt(1.0 + family_seen.get(family, 0))
        family_seen[family] = family_seen.get(family, 0) + 1

        status = str(card.get("status") or "unknown")
        operator = str(card.get("operator") or "unknown")
        status_count[status] = status_count.get(status, 0) + 1
        operator_counts[operator] = operator_counts.get(operator, 0) + 1
        parent_graph[node_id] = list(card.get("parents") or [])

        error_signature = card.get("error_signature")
        if error_signature and _card_indicates_failure(card):
            repeated_errors[str(error_signature)] = repeated_errors.get(str(error_signature), 0) + 1

        score_history.append(
            {
                "step_id": card.get("step_id"),
                "node_id": node_id,
                "score": card.get("score"),
                "fitness": card.get("fitness"),
            }
        )

        delta = _finite_float(card.get("delta_vs_parent"))
        if delta is not None:
            deltas.append(delta)

        stats = method_family_stats.setdefault(
            family,
            {
                "count": 0,
                "valid_count": 0,
                "fail_count": 0,
                "best": None,
                "best_node": None,
                "fail_rate": 0.0,
            },
        )
        stats["count"] += 1
        fitness = _finite_float(card.get("fitness"))
        failed = bool(card.get("is_buggy")) or _status_is_failure(status) or fitness is None
        if failed:
            stats["fail_count"] += 1
        else:
            stats["valid_count"] += 1
            previous_best = _finite_float(stats.get("best"))
            is_better = previous_best is None or (fitness < previous_best if lower_is_better else fitness > previous_best)
            if is_better:
                stats["best"] = fitness
                stats["best_node"] = node_id

    for stats in method_family_stats.values():
        count = int(stats["count"])
        stats["fail_rate"] = float(stats["fail_count"]) / count if count else 0.0

    rank_by_node = {
        str(card.get("node_id")): rank for rank, card in enumerate(sorted_valid_cards, start=1)
    }
    family_best_nodes = {
        family: stats.get("best_node") for family, stats in method_family_stats.items() if stats.get("best_node")
    }
    underexplored_families = sorted(
        [
            family
            for family, stats in method_family_stats.items()
            if family != "unknown" and int(stats["count"]) <= 2 and float(stats["fail_rate"]) < 1.0
        ]
    )
    sandbox_times = [_finite_float(card.get("sandbox_time_used")) for card in cards]
    model_times = [_finite_float(card.get("model_time_used")) for card in cards]
    sandbox_times = [value for value in sandbox_times if value is not None]
    model_times = [value for value in model_times if value is not None]

    return {
        "schema_version": 1,
        "num_nodes": len(cards),
        "best_node": best_card.get("node_id") if best_card else None,
        "best_score": best_card.get("fitness") if best_card else None,
        "current_best_family": best_card.get("method_family_auto") if best_card else None,
        "method_family_stats": method_family_stats,
        "underexplored_families": underexplored_families,
        "repeated_errors": repeated_errors,
        "novelty_by_node": novelty_by_node,
        "rank_by_node": rank_by_node,
        "current_best_by_node": {
            str(card.get("node_id")): bool(best_card and card.get("node_id") == best_card.get("node_id"))
            for card in cards
        },
        "family_best_nodes": family_best_nodes,
        "recent_delta_trend": sum(deltas[-5:]) / len(deltas[-5:]) if deltas else None,
        "status_count": status_count,
        "operator_counts": operator_counts,
        "parent_graph": parent_graph,
        "score_history": score_history,
        "runtime_stats": {
            "sandbox_total": sum(sandbox_times),
            "sandbox_avg": sum(sandbox_times) / len(sandbox_times) if sandbox_times else 0.0,
            "model_total": sum(model_times),
            "model_avg": sum(model_times) / len(model_times) if model_times else 0.0,
        },
        "parent_selection_weights": [
            card.get("parent_selection_trace")
            for card in cards
            if card.get("parent_selection_trace") is not None
        ],
    }


def _node_card(node: Any) -> dict[str, Any] | None:
    card = getattr(node, "experience_card", None)
    if isinstance(card, dict):
        return card
    return None


def _collect_attached_cards(
    journal: Any | None,
    nodes: Iterable[Any] | None = None,
) -> list[dict[str, Any]]:
    cards_by_id: dict[str, dict[str, Any]] = {}
    for node in list(getattr(journal, "nodes", []) or []) + list(nodes or []):
        card = _node_card(node)
        if not card:
            continue
        node_id = str(card.get("node_id") or getattr(node, "id", ""))
        if node_id:
            cards_by_id[node_id] = card
    return list(cards_by_id.values())


def _node_id(node: Any) -> str:
    return str(getattr(node, "id", "") or "")


def _dedupe_nodes(nodes: Iterable[Any]) -> list[Any]:
    seen: set[str] = set()
    deduped = []
    for node in nodes:
        node_id = _node_id(node)
        if not node_id or node_id in seen:
            continue
        seen.add(node_id)
        deduped.append(node)
    return deduped


def _journal_nodes(journal: Any | None, extra_nodes: Iterable[Any] | None = None) -> list[Any]:
    return _dedupe_nodes(list(getattr(journal, "nodes", []) or []) + list(extra_nodes or []))


def _rank_nodes_by_utility(
    nodes: list[Any],
    *,
    lower_is_better: bool,
    cards: list[dict[str, Any]],
    weights: dict[str, float] | None = None,
) -> list[Any]:
    if not nodes:
        return []
    utilities = compute_parent_utilities(
        nodes,
        lower_is_better=lower_is_better,
        previous_cards=cards,
        weights=weights,
    )
    utility_by_id = {str(item.get("node_id")): float(item.get("utility") or 0.0) for item in utilities}
    return sorted(nodes, key=lambda node: utility_by_id.get(_node_id(node), 0.0), reverse=True)


def _recent_ancestors(node: Any, max_depth: int) -> list[Any]:
    if max_depth <= 0:
        return []

    ancestors: list[Any] = []
    frontier = list(getattr(node, "parents", None) or [])
    seen: set[str] = {_node_id(node)}
    while frontier and len(ancestors) < max_depth:
        parent = frontier.pop(0)
        parent_id = _node_id(parent)
        if not parent_id or parent_id in seen:
            continue
        seen.add(parent_id)
        ancestors.append(parent)
        frontier.extend(list(getattr(parent, "parents", None) or []))
    return ancestors


def _horizontal_siblings(
    node: Any,
    journal: Any | None,
    *,
    lower_is_better: bool,
    cards: list[dict[str, Any]],
    max_siblings: int,
    sibling_rank_weights: dict[str, float] | None = None,
) -> list[Any]:
    if max_siblings <= 0:
        return []
    parent_ids = {
        _node_id(parent)
        for parent in list(getattr(node, "parents", None) or [])
        if _node_id(parent)
    }
    if not parent_ids:
        return []

    node_id = _node_id(node)
    siblings = []
    for candidate in _journal_nodes(journal):
        candidate_id = _node_id(candidate)
        if not candidate_id or candidate_id == node_id:
            continue
        candidate_parent_ids = {
            _node_id(parent)
            for parent in list(getattr(candidate, "parents", None) or [])
            if _node_id(parent)
        }
        if parent_ids.intersection(candidate_parent_ids):
            siblings.append(candidate)

    return _rank_nodes_by_utility(
        _dedupe_nodes(siblings),
        lower_is_better=lower_is_better,
        cards=cards,
        weights=sibling_rank_weights,
    )[:max_siblings]


def collect_operator_memory_nodes(
    operator: str,
    parent_nodes: list[Any],
    *,
    journal: Any | None,
    lower_is_better: bool,
    current_node: Any | None = None,
    ancestor_k: int | None = None,
    sibling_k: int | None = None,
    sibling_rank_weights: dict[str, float] | None = None,
    max_related_cards: int = 8,
) -> dict[str, list[Any]]:
    """Select the small targeted memory set used by one operator call.

    IMPROVE uses the selected parent, its recent ancestry, and top horizontal
    siblings. CROSSOVER applies the same logic to both parents with smaller
    per-parent budgets. DEBUG prioritizes previous nodes with the same error
    signature, then recent nodes.
    """
    normalized_operator = str(operator or "").lower()
    parent_nodes = _dedupe_nodes(parent_nodes)
    cards = _collect_attached_cards(journal, parent_nodes + ([current_node] if current_node is not None else []))
    sections: dict[str, list[Any]] = {
        "primary": [],
        "vertical": [],
        "horizontal": [],
        "debug_related": [],
    }

    if normalized_operator == "improve":
        if not parent_nodes:
            return sections
        parent = parent_nodes[0]
        use_ancestor_k = 3 if ancestor_k is None else max(0, int(ancestor_k))
        use_sibling_k = 3 if sibling_k is None else max(0, int(sibling_k))
        sections["primary"] = [parent]
        sections["vertical"] = _recent_ancestors(parent, use_ancestor_k)
        sections["horizontal"] = _horizontal_siblings(
            parent,
            journal,
            lower_is_better=lower_is_better,
            cards=cards,
            max_siblings=use_sibling_k,
            sibling_rank_weights=sibling_rank_weights,
        )
        return sections

    if normalized_operator == "crossover":
        use_ancestor_k = 2 if ancestor_k is None else max(0, int(ancestor_k))
        use_sibling_k = 2 if sibling_k is None else max(0, int(sibling_k))
        sections["primary"] = parent_nodes[:2]
        all_vertical: list[Any] = []
        all_horizontal: list[Any] = []
        for index, parent in enumerate(parent_nodes[:2], start=1):
            vertical = _recent_ancestors(parent, use_ancestor_k)
            horizontal = _horizontal_siblings(
                parent,
                journal,
                lower_is_better=lower_is_better,
                cards=cards,
                max_siblings=use_sibling_k,
                sibling_rank_weights=sibling_rank_weights,
            )
            sections[f"parent_{index}_vertical"] = vertical
            sections[f"parent_{index}_horizontal"] = horizontal
            all_vertical.extend(vertical)
            all_horizontal.extend(horizontal)
        sections["vertical"] = _dedupe_nodes(all_vertical)
        sections["horizontal"] = _dedupe_nodes(all_horizontal)
        return sections

    if normalized_operator == "debug":
        focus_node = current_node or (parent_nodes[0] if parent_nodes else None)
        sections["primary"] = [focus_node] if focus_node is not None else []
        signature = _current_error_signature(focus_node)
        related = []
        recent = []
        for candidate in reversed(_journal_nodes(journal)):
            if focus_node is not None and _node_id(candidate) == _node_id(focus_node):
                continue
            card = _card_for_node(candidate, cards)
            if signature and str(card.get("error_signature") or "") == signature:
                related.append(candidate)
            else:
                recent.append(candidate)
        cap = max(1, int(max_related_cards or 8))
        sections["debug_related"] = _dedupe_nodes(related + recent)[:cap]
        return sections

    return sections


def _card_for_node(node: Any, cards: list[dict[str, Any]]) -> dict[str, Any]:
    card = _node_card(node)
    if card:
        return card

    node_id = str(getattr(node, "id", ""))
    for candidate in cards:
        if str(candidate.get("node_id")) == node_id:
            return candidate

    code = str(getattr(node, "code", "") or "")
    imports = extract_imports(code)
    return {
        "node_id": node_id,
        "method_family_auto": detect_method_family(code, imports),
        "fitness": _metric_value(node),
        "status": str((getattr(getattr(node, "metric", None), "info", {}) or {}).get("status") or ""),
        "is_buggy": bool(getattr(node, "is_buggy", False)),
        "error_signature": None,
        "analysis": str(getattr(node, "analysis", "") or ""),
    }


def _rich_summary_lines(prefix: str, card: dict[str, Any]) -> list[str]:
    rich_summary = card.get("rich_summary")
    if isinstance(rich_summary, dict):
        method_overview = _compact_text(rich_summary.get("method_overview"), max_chars=420)
        parent_experience = _compact_text(
            rich_summary.get("parent_comparison_experience"),
            max_chars=420,
        )
        lines = []
        if method_overview:
            lines.append(f"- {prefix}_method_overview: {method_overview}")
        if parent_experience:
            lines.append(f"- {prefix}_parent_comparison_experience: {parent_experience}")
        if lines:
            return lines

    analysis = _compact_text(card.get("analysis"), max_chars=360)
    if analysis:
        return [f"- {prefix}_legacy_analysis: {analysis}"]
    return []


def _memory_node_lines(prefix: str, card: dict[str, Any]) -> list[str]:
    lines = [
        (
            f"- {prefix}_node_id: {card.get('node_id')} "
            f"(family={card.get('method_family_auto') or 'unknown'}, "
            f"score={_format_value(card.get('fitness'))}, "
            f"delta_vs_parent={_format_value(card.get('delta_vs_parent'))}, "
            f"runtime_seconds={_format_value(card.get('sandbox_time_used'))})"
        ),
        f"- {prefix}_rank: {_format_value(card.get('rank'))}",
        f"- {prefix}_current_best: {bool(card.get('current_best', False))}",
        f"- {prefix}_is_new_direction: {bool(card.get('is_new_direction', False))}",
    ]
    lines.extend(_rich_summary_lines(prefix, card))
    return lines


def _append_memory_node_section(
    lines: list[str],
    title: str,
    prefix: str,
    nodes: list[Any],
    cards: list[dict[str, Any]],
) -> None:
    if not nodes:
        return
    lines.extend(["", title + ":"])
    for index, node in enumerate(nodes, start=1):
        card = _card_for_node(node, cards)
        item_prefix = prefix if len(nodes) == 1 else f"{prefix}_{index}"
        lines.extend(_memory_node_lines(item_prefix, card))


def _family_stats_line(label: str, family: str, board: dict[str, Any]) -> str:
    stats = dict(board.get("method_family_stats", {}).get(family) or {})
    if not stats:
        return f"- {label}: unavailable"
    return (
        f"- {label}: count={int(stats.get('count') or 0)}, "
        f"best={_format_value(stats.get('best'))}, "
        f"fail_rate={_format_value(stats.get('fail_rate'))}, "
        f"best_node={stats.get('best_node') or 'n/a'}"
    )


def _format_repeated_errors(repeated_errors: dict[str, Any]) -> str:
    items = [
        f"{signature}={count}"
        for signature, count in sorted(repeated_errors.items(), key=lambda item: str(item[0]))
    ]
    return ", ".join(items) if items else "none"


def _current_error_signature(current_node: Any | None) -> str | None:
    if current_node is None:
        return None
    info = dict(getattr(getattr(current_node, "metric", None), "info", {}) or {})
    status_code = info.get("status_code")
    try:
        status_code = int(status_code) if status_code is not None else None
    except (TypeError, ValueError):
        status_code = None
    return detect_error_signature(
        str(info.get("status") or ""),
        status_code,
        str(info.get("raw_run_log") or ""),
        str(info.get("clear_run_log") or getattr(current_node, "term_out", "") or ""),
    )


def _build_improve_experience_memory(
    parent_nodes: list[Any],
    cards: list[dict[str, Any]],
    board: dict[str, Any],
    sections: dict[str, list[Any]],
) -> str:
    if not parent_nodes:
        return ""
    parent_card = _card_for_node(parent_nodes[0], cards)
    parent_family = str(parent_card.get("method_family_auto") or "unknown")
    lines = [
        "Targeted Memory Context for IMPROVE",
        "",
        "Use this memory as evidence about the selected parent, its recent evolution path, and nearby sibling attempts. Do not treat it as a complete history.",
    ]
    _append_memory_node_section(lines, "Selected parent memory", "parent", sections.get("primary") or parent_nodes[:1], cards)
    _append_memory_node_section(lines, "Vertical ancestor memory (recent evolution path)", "ancestor", sections.get("vertical") or [], cards)
    _append_memory_node_section(lines, "Horizontal sibling memory (nearby alternative attempts)", "sibling", sections.get("horizontal") or [], cards)
    lines.extend(
        [
            "",
            "Related board stats:",
            f"- current_best_node: {board.get('best_node') or 'n/a'}",
            f"- current_best_score: {_format_value(board.get('best_score'))}",
            f"- current_best_family: {board.get('current_best_family') or 'unknown'}",
            _family_stats_line("parent_family_stats", parent_family, board),
            f"- underexplored_families: {_format_list(board.get('underexplored_families') or [])}",
            f"- repeated_errors: {_format_repeated_errors(dict(board.get('repeated_errors') or {}))}",
            f"- recent_delta_trend: {_format_value(board.get('recent_delta_trend'))}",
        ]
    )
    return "\n".join(lines)


def _build_crossover_experience_memory(
    parent_nodes: list[Any],
    cards: list[dict[str, Any]],
    board: dict[str, Any],
    sections: dict[str, list[Any]],
) -> str:
    if len(parent_nodes) < 2:
        return ""
    card_a = _card_for_node(parent_nodes[0], cards)
    card_b = _card_for_node(parent_nodes[1], cards)
    family_a = str(card_a.get("method_family_auto") or "unknown")
    family_b = str(card_b.get("method_family_auto") or "unknown")
    if family_a != family_b:
        complementarity = "different_method_families"
        crossover_hint = f"combine {family_a} with {family_b}"
    else:
        complementarity = "same_method_family"
        crossover_hint = f"combine complementary implementation details within {family_a}"

    lines = [
        "Targeted Memory Context for CROSSOVER",
        "",
        "Use this memory to identify compatible strengths and conflicts between the two selected parent branches. Do not mechanically merge all code.",
        "",
        "Selected parent memories:",
    ]
    lines.extend(_memory_node_lines("parent_1", card_a))
    lines.extend(_memory_node_lines("parent_2", card_b))
    _append_memory_node_section(lines, "Parent 1 vertical ancestor memory", "parent_1_ancestor", sections.get("parent_1_vertical") or [], cards)
    _append_memory_node_section(lines, "Parent 1 horizontal sibling memory", "parent_1_sibling", sections.get("parent_1_horizontal") or [], cards)
    _append_memory_node_section(lines, "Parent 2 vertical ancestor memory", "parent_2_ancestor", sections.get("parent_2_vertical") or [], cards)
    _append_memory_node_section(lines, "Parent 2 horizontal sibling memory", "parent_2_sibling", sections.get("parent_2_horizontal") or [], cards)
    lines.extend(
        [
            "",
            "Complementarity:",
            f"- family_complementarity: {complementarity}",
            f"- crossover_hint: {crossover_hint}",
            _family_stats_line("parent_1_family_stats", family_a, board),
            _family_stats_line("parent_2_family_stats", family_b, board),
            f"- repeated_errors_to_avoid: {_format_repeated_errors(dict(board.get('repeated_errors') or {}))}",
        ]
    )
    return "\n".join(lines)


def _build_debug_experience_memory(
    parent_nodes: list[Any],
    cards: list[dict[str, Any]],
    board: dict[str, Any],
    current_node: Any | None,
    max_related_cards: int,
    sections: dict[str, list[Any]],
) -> str:
    current_node = current_node or (parent_nodes[0] if parent_nodes else None)
    signature = _current_error_signature(current_node)
    related_nodes = sections.get("debug_related") or []

    lines = [
        "Targeted Memory Context for DEBUG",
        "",
        "Current error:",
        f"- current_error_signature: {signature or 'unknown'}",
        f"- repeated_errors: {_format_repeated_errors(dict(board.get('repeated_errors') or {}))}",
    ]
    _append_memory_node_section(
        lines,
        "Current buggy node memory",
        "current_buggy",
        sections.get("primary") or ([current_node] if current_node is not None else []),
        cards,
    )
    if related_nodes:
        lines.append("")
        lines.append("Related previous debug/error memories:")
        for node in related_nodes[:max_related_cards]:
            card = _card_for_node(node, cards)
            lines.append(
                f"- related_error_node: {card.get('node_id')}, "
                f"operator={card.get('operator') or 'unknown'}, "
                f"family={card.get('method_family_auto') or 'unknown'}, "
                f"status={card.get('status') or 'unknown'}, "
                f"score={_format_value(card.get('fitness'))}, "
                f"delta_vs_parent={_format_value(card.get('delta_vs_parent'))}, "
                f"runtime_seconds={_format_value(card.get('sandbox_time_used'))}"
            )
            lines.extend(_rich_summary_lines("related_error", card))
    return "\n".join(lines)


def build_operator_experience_memory(
    operator: str,
    parent_nodes: list[Any],
    *,
    journal: Any | None,
    lower_is_better: bool,
    current_node: Any | None = None,
    max_related_cards: int = 3,
    ancestor_k: int | None = None,
    sibling_k: int | None = None,
    sibling_rank_weights: dict[str, float] | None = None,
) -> str:
    cards = _collect_attached_cards(journal, parent_nodes)
    if not cards and not current_node:
        return ""

    board = build_strategy_board(cards, lower_is_better=lower_is_better)
    normalized_operator = str(operator or "").lower()
    sections = collect_operator_memory_nodes(
        normalized_operator,
        parent_nodes,
        journal=journal,
        lower_is_better=lower_is_better,
        current_node=current_node,
        ancestor_k=ancestor_k,
        sibling_k=sibling_k,
        sibling_rank_weights=sibling_rank_weights,
        max_related_cards=max_related_cards,
    )
    if normalized_operator == "improve":
        return _build_improve_experience_memory(parent_nodes, cards, board, sections)
    if normalized_operator == "crossover":
        return _build_crossover_experience_memory(parent_nodes, cards, board, sections)
    if normalized_operator == "debug":
        return _build_debug_experience_memory(
            parent_nodes,
            cards,
            board,
            current_node,
            max_related_cards=max_related_cards,
            sections=sections,
        )
    return ""


def append_experience_memory(memory: str | None, experience_memory: str) -> str:
    experience_memory = str(experience_memory or "").strip()
    base_memory = str(memory or "").strip()
    if not experience_memory:
        return base_memory
    if not base_memory or base_memory == "(No memory available.)":
        return experience_memory
    return f"{base_memory}\n\n---\n\n{experience_memory}"


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def prompt_memory_enabled(cfg: Any) -> bool:
    experience_cfg = _config_get(cfg, "experience", {}) or {}
    if not bool(_config_get(experience_cfg, "enabled", False)):
        return False
    prompt_memory_cfg = _config_get(experience_cfg, "prompt_memory", {}) or {}
    return bool(_config_get(prompt_memory_cfg, "enabled", True))


def prompt_memory_max_related_cards(cfg: Any, operator: str | None = None) -> int:
    experience_cfg = _config_get(cfg, "experience", {}) or {}
    prompt_memory_cfg = _config_get(experience_cfg, "prompt_memory", {}) or {}
    if operator:
        operator_cfg = _config_get(prompt_memory_cfg, str(operator or "").lower(), {}) or {}
        value = _config_get(operator_cfg, "max_related_cards", None)
        if value is not None:
            try:
                return max(1, int(value))
            except (TypeError, ValueError):
                pass
    try:
        return max(1, int(_config_get(prompt_memory_cfg, "max_related_cards", 3)))
    except (TypeError, ValueError):
        return 3


def prompt_memory_use_base_memory(cfg: Any) -> bool:
    experience_cfg = _config_get(cfg, "experience", {}) or {}
    prompt_memory_cfg = _config_get(experience_cfg, "prompt_memory", {}) or {}
    return bool(_config_get(prompt_memory_cfg, "use_base_memory", False))


def prompt_memory_operator_k(cfg: Any, operator: str, key: str, default: int) -> int:
    experience_cfg = _config_get(cfg, "experience", {}) or {}
    prompt_memory_cfg = _config_get(experience_cfg, "prompt_memory", {}) or {}
    operator_cfg = _config_get(prompt_memory_cfg, str(operator or "").lower(), {}) or {}
    try:
        return max(0, int(_config_get(operator_cfg, key, default)))
    except (TypeError, ValueError):
        return default


def prompt_memory_sibling_rank_weights(cfg: Any) -> dict[str, float] | None:
    experience_cfg = _config_get(cfg, "experience", {}) or {}
    prompt_memory_cfg = _config_get(experience_cfg, "prompt_memory", {}) or {}
    sibling_ranking_cfg = _config_get(prompt_memory_cfg, "sibling_ranking", {}) or {}
    weights = _config_get(sibling_ranking_cfg, "weights", None)
    if weights is None:
        return None
    try:
        return {str(key): float(value) for key, value in dict(weights).items()}
    except (TypeError, ValueError):
        return None


def compute_parent_utilities(
    nodes: list[Any],
    *,
    lower_is_better: bool,
    previous_cards: list[dict[str, Any]] | None = None,
    weights: dict[str, float] | None = None,
    component_normalization: dict[str, Any] | None = None,
    temperature: float = 1.0,
) -> list[dict[str, Any]]:
    weights = {**DEFAULT_PARENT_UTILITY_WEIGHTS, **dict(weights or {})}
    component_normalization = _component_normalization_config(component_normalization)
    previous_cards = list(previous_cards or [])
    metric_scores = [_metric_value(node) for node in nodes]
    # Parent selection must follow the same self-validation metric used by the
    # journal and final node selection.  The sandbox/raw score is kept in the
    # trace for diagnostics only; using it here would leak test feedback into
    # the search controller.
    official_scores = [
        _official_score_value(node) if _has_official_score(node) else None
        for node in nodes
    ]
    selection_scores = metric_scores
    score_components = _normalize_values(
        selection_scores,
        lower_is_better,
        mode=str(component_normalization["score"]),
        hybrid_minmax_weight=float(component_normalization["hybrid_minmax_weight"]),
    )

    raw_deltas: list[float | None] = []
    families: list[str] = []
    candidate_family_counts: dict[str, int] = {}
    for idx, node in enumerate(nodes):
        imports = extract_imports(str(getattr(node, "code", "") or ""))
        family = detect_method_family(str(getattr(node, "code", "") or ""), imports)
        families.append(family)
        candidate_family_counts[family] = candidate_family_counts.get(family, 0) + 1

        parent_scores = [
            _metric_value(parent)
            for parent in list(getattr(node, "parents", None) or [])
        ]
        finite_parent_scores = [score for score in parent_scores if score is not None]
        if finite_parent_scores:
            parent_score = min(finite_parent_scores) if lower_is_better else max(finite_parent_scores)
            raw_deltas.append(_positive_delta(selection_scores[idx], parent_score, lower_is_better))
        else:
            raw_deltas.append(0.0)

    positive_deltas = [max(0.0, raw_delta or 0.0) for raw_delta in raw_deltas]
    delta_components = _normalize_positive_deltas(
        positive_deltas,
        mode=str(component_normalization["delta"]),
        hybrid_minmax_weight=float(component_normalization["hybrid_minmax_weight"]),
    )
    utilities: list[dict[str, Any]] = []
    for idx, node in enumerate(nodes):
        node_id = str(getattr(node, "id", ""))
        family = families[idx]
        score_component = score_components[idx]
        raw_delta = raw_deltas[idx]
        delta_component = delta_components[idx]
        previous_family_count = _family_count_before(family, previous_cards, node_id=node_id)
        if not previous_cards:
            previous_family_count = max(0, candidate_family_counts.get(family, 1) - 1)
        novelty_component = 1.0 / math.sqrt(1.0 + previous_family_count)
        official_score = official_scores[idx]
        official_score_missing = False
        official_score_missing_penalty = 0.0
        utility = (
            float(weights["score"]) * score_component
            + float(weights["delta"]) * delta_component
            + float(weights["novelty"]) * novelty_component
            - official_score_missing_penalty
        )
        utilities.append(
            {
                "node_id": node_id,
                "method_family_auto": family,
                "fitness": metric_scores[idx],
                "official_score": official_score,
                "score_source": "self_validation",
                "component_normalization": dict(component_normalization),
                "official_score_used": False,
                "official_status": _official_status(node),
                "official_score_missing": official_score_missing,
                "official_score_missing_penalty": official_score_missing_penalty,
                "score_component": score_component,
                "delta_component": delta_component,
                "novelty_component": novelty_component,
                "utility": utility,
                "probability": 0.0,
            }
        )

    probabilities = softmax([item["utility"] for item in utilities], temperature=temperature)
    for item, probability in zip(utilities, probabilities):
        item["probability"] = probability
    return utilities


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


def load_experience_cards(output_dir: Path) -> list[dict[str, Any]]:
    cards_path = output_dir / "experience_cards.jsonl"
    if cards_path.exists():
        cards = []
        for line in cards_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                cards.append(json.loads(line))
        return cards

    cards = []
    for card_path in sorted(output_dir.glob("step_*/experience_card.json")):
        cards.append(json.loads(card_path.read_text(encoding="utf-8")))
    return cards
