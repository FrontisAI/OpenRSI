import math
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

EXPERIENCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "dojo"
    / "solvers"
    / "evo"
    / "experience.py"
)
spec = importlib.util.spec_from_file_location("experience_under_test", EXPERIENCE_PATH)
experience = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(experience)


def make_node(code: str, metric: float | None, *, parents=None, is_buggy=False):
    node = SimpleNamespace(
        id=uuid4().hex,
        code=code,
        plan="plan",
        parents=list(parents or []),
        operators_used=["improve" if parents else "draft"],
        metric=SimpleNamespace(value=metric, info={"status": "success"}),
        is_buggy=is_buggy,
        analysis="existing aira analysis",
        step=1,
        exec_time=12.0,
    )
    return node


def attach_card(node, *, fitness=None, delta=None, family="lightgbm", rich_summary=None, runtime=12.0):
    node.experience_card = {
        "node_id": node.id,
        "method_family_auto": family,
        "fitness": fitness if fitness is not None else node.metric.value,
        "delta_vs_parent": delta,
        "rank": None,
        "current_best": False,
        "is_new_direction": False,
        "status": "success",
        "is_buggy": False,
        "sandbox_time_used": runtime,
        "analysis": "legacy analysis",
    }
    if rich_summary is not None:
        node.experience_card["rich_summary"] = rich_summary
    return node


def test_auto_card_extracts_method_family_delta_and_novelty():
    parent = make_node(
        "import lightgbm as lgb\nprint('parent')",
        0.70,
    )
    child = make_node(
        "import lightgbm as lgb\nfrom catboost import CatBoostClassifier\nprint('child')",
        0.76,
        parents=[parent],
    )

    card = experience.build_experience_card(
        node=child,
        step_index=3,
        generation_id=1,
        lower_is_better=False,
        previous_cards=[],
        step_stat={"status": "success", "status_code": 200, "score": 0.76, "reward": 0.76},
        usage={"cost": 0.1, "prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30, "latency": 4.0},
        clear_run_log="ok",
        raw_run_log="ok",
    )

    assert card["node_id"] == child.id
    assert card["step_id"] == 3
    assert card["parents"] == [0]
    assert card["imports"] == ["catboost", "lightgbm"]
    assert card["method_family_auto"] == "catboost+lightgbm"
    assert card["delta_vs_parent"] == pytest.approx(0.06)
    assert card["is_new_direction"] is True
    assert card["novelty_score"] == pytest.approx(1.0)
    assert card["analysis"] == "existing aira analysis"


def test_board_aggregates_family_stats_and_repeated_errors():
    cards = [
        {
            "node_id": "a",
            "step_id": 0,
            "operator": "draft",
            "parents": [],
            "fitness": 0.70,
            "status": "success",
            "is_buggy": False,
            "method_family_auto": "lightgbm",
            "error_signature": None,
            "delta_vs_parent": None,
            "sandbox_time_used": 10.0,
            "model_time_used": 1.0,
        },
        {
            "node_id": "b",
            "step_id": 1,
            "operator": "improve",
            "parents": [0],
            "fitness": 0.76,
            "status": "valid",
            "is_buggy": False,
            "method_family_auto": "catboost",
            "error_signature": None,
            "delta_vs_parent": 0.06,
            "sandbox_time_used": 20.0,
            "model_time_used": 2.0,
        },
        {
            "node_id": "c",
            "step_id": 2,
            "operator": "debug",
            "parents": [1],
            "fitness": None,
            "status": "timeout",
            "is_buggy": True,
            "method_family_auto": "neural_net",
            "error_signature": "timeout",
            "delta_vs_parent": None,
            "sandbox_time_used": 30.0,
            "model_time_used": 3.0,
        },
    ]

    board = experience.build_strategy_board(cards, lower_is_better=False)

    assert board["best_node"] == "b"
    assert board["best_score"] == pytest.approx(0.76)
    assert board["current_best_family"] == "catboost"
    assert board["method_family_stats"]["catboost"]["best"] == pytest.approx(0.76)
    assert board["method_family_stats"]["catboost"]["fail_rate"] == pytest.approx(0.0)
    assert board["method_family_stats"]["neural_net"]["fail_rate"] == pytest.approx(1.0)
    assert board["repeated_errors"] == {"timeout": 1}
    assert board["operator_counts"] == {"draft": 1, "improve": 1, "debug": 1}
    assert board["recent_delta_trend"] == pytest.approx(0.06)


def test_parent_utilities_combine_score_delta_and_novelty():
    best_parent = make_node("import lightgbm", 0.90)
    improved_parent_parent = make_node("import xgboost", 0.60)
    improved_parent = make_node("import xgboost", 0.82, parents=[improved_parent_parent])
    new_direction = make_node("from catboost import CatBoostRegressor", 0.78)

    previous_cards = [
        {"node_id": best_parent.id, "method_family_auto": "lightgbm"},
        {"node_id": improved_parent_parent.id, "method_family_auto": "xgboost"},
        {"node_id": improved_parent.id, "method_family_auto": "xgboost"},
    ]

    utilities = experience.compute_parent_utilities(
        [best_parent, improved_parent, new_direction],
        lower_is_better=False,
        previous_cards=previous_cards,
        weights={"score": 1.0, "delta": 0.4, "novelty": 0.25},
    )

    by_id = {item["node_id"]: item for item in utilities}
    assert by_id[best_parent.id]["score_component"] == pytest.approx(1.0)
    assert by_id[improved_parent.id]["delta_component"] > by_id[best_parent.id]["delta_component"]
    assert by_id[new_direction.id]["novelty_component"] == pytest.approx(1.0)
    assert by_id[best_parent.id]["utility"] > by_id[new_direction.id]["utility"]
    assert math.isclose(sum(item["probability"] for item in utilities), 1.0)


def test_parent_utilities_use_self_validation_not_official_score():
    high_val_low_official = make_node("import lightgbm", 0.99)
    high_val_low_official.metric.info = {
        "score": 0.10,
        "raw_score": 0.10,
        "validation_score": 0.99,
        "status": "success",
        "status_code": 200,
    }
    low_val_high_official = make_node("import lightgbm", 0.84)
    low_val_high_official.metric.info = {
        "score": 0.98,
        "raw_score": 0.98,
        "validation_score": 0.84,
        "status": "success",
        "status_code": 200,
    }
    missing_official_high_val = make_node("import lightgbm", 0.95)
    missing_official_high_val.metric.info = {
        "score": None,
        "raw_score": None,
        "validation_score": 0.95,
        "status": "success",
        "status_code": 200,
    }

    utilities = experience.compute_parent_utilities(
        [high_val_low_official, low_val_high_official, missing_official_high_val],
        lower_is_better=False,
        previous_cards=[],
        weights={"score": 1.0, "delta": 0.0, "novelty": 0.0},
    )

    by_id = {item["node_id"]: item for item in utilities}
    assert by_id[high_val_low_official.id]["official_score"] == pytest.approx(0.10)
    assert by_id[low_val_high_official.id]["official_score"] == pytest.approx(0.98)
    assert by_id[missing_official_high_val.id]["official_score"] is None
    assert by_id[high_val_low_official.id]["score_source"] == "self_validation"
    assert by_id[high_val_low_official.id]["official_score_used"] is False
    assert by_id[missing_official_high_val.id]["official_score_missing"] is False
    assert by_id[high_val_low_official.id]["score_component"] == pytest.approx(1.0)
    assert by_id[missing_official_high_val.id]["score_component"] > by_id[low_val_high_official.id]["score_component"]
    assert by_id[high_val_low_official.id]["utility"] > by_id[low_val_high_official.id]["utility"]
    assert by_id[missing_official_high_val.id]["utility"] > by_id[low_val_high_official.id]["utility"]


def test_parent_utilities_can_switch_score_component_normalization():
    nodes = [
        make_node("import lightgbm", 0.900),
        make_node("import lightgbm", 0.805),
        make_node("import lightgbm", 0.804),
        make_node("import lightgbm", 0.803),
        make_node("import lightgbm", 0.802),
    ]

    minmax_utilities = experience.compute_parent_utilities(
        nodes,
        lower_is_better=False,
        previous_cards=[],
        weights={"score": 1.0, "delta": 0.0, "novelty": 0.0},
        component_normalization={"score": "minmax", "delta": "minmax"},
    )
    rank_utilities = experience.compute_parent_utilities(
        nodes,
        lower_is_better=False,
        previous_cards=[],
        weights={"score": 1.0, "delta": 0.0, "novelty": 0.0},
        component_normalization={"score": "rank", "delta": "minmax"},
    )
    hybrid_utilities = experience.compute_parent_utilities(
        nodes,
        lower_is_better=False,
        previous_cards=[],
        weights={"score": 1.0, "delta": 0.0, "novelty": 0.0},
        component_normalization={"score": "hybrid", "delta": "minmax", "hybrid_minmax_weight": 0.5},
    )

    minmax_by_id = {item["node_id"]: item for item in minmax_utilities}
    rank_by_id = {item["node_id"]: item for item in rank_utilities}
    hybrid_by_id = {item["node_id"]: item for item in hybrid_utilities}
    second_node = nodes[1]

    assert minmax_by_id[second_node.id]["score_component"] < 0.05
    assert rank_by_id[nodes[0].id]["score_component"] == pytest.approx(1.0)
    assert rank_by_id[second_node.id]["score_component"] == pytest.approx(0.75)
    assert rank_by_id[nodes[-1].id]["score_component"] == pytest.approx(0.0)
    expected_hybrid = 0.5 * minmax_by_id[second_node.id]["score_component"] + 0.5 * 0.75
    assert hybrid_by_id[second_node.id]["score_component"] == pytest.approx(expected_hybrid)


def test_parent_utilities_can_switch_delta_component_normalization():
    parent = make_node("import lightgbm", 0.50)
    large_delta = make_node("import lightgbm", 0.60, parents=[parent])
    mid_delta = make_node("import lightgbm", 0.52, parents=[parent])
    small_delta = make_node("import lightgbm", 0.518, parents=[parent])
    no_delta = make_node("import lightgbm", 0.50, parents=[parent])

    minmax_utilities = experience.compute_parent_utilities(
        [large_delta, mid_delta, small_delta, no_delta],
        lower_is_better=False,
        previous_cards=[],
        weights={"score": 0.0, "delta": 1.0, "novelty": 0.0},
        component_normalization={"score": "minmax", "delta": "minmax"},
    )
    rank_utilities = experience.compute_parent_utilities(
        [large_delta, mid_delta, small_delta, no_delta],
        lower_is_better=False,
        previous_cards=[],
        weights={"score": 0.0, "delta": 1.0, "novelty": 0.0},
        component_normalization={"score": "minmax", "delta": "rank"},
    )

    minmax_by_id = {item["node_id"]: item for item in minmax_utilities}
    rank_by_id = {item["node_id"]: item for item in rank_utilities}

    assert minmax_by_id[mid_delta.id]["delta_component"] == pytest.approx(0.2)
    assert rank_by_id[large_delta.id]["delta_component"] == pytest.approx(1.0)
    assert rank_by_id[mid_delta.id]["delta_component"] == pytest.approx(2 / 3)
    assert rank_by_id[small_delta.id]["delta_component"] == pytest.approx(1 / 3)
    assert rank_by_id[no_delta.id]["delta_component"] == pytest.approx(0.0)


def test_method_and_error_helpers_are_stable():
    code = "import xgboost as xgb\nfrom sklearn.ensemble import VotingClassifier\n"
    assert experience.extract_imports(code) == ["sklearn", "xgboost"]
    assert experience.detect_method_family(code, experience.extract_imports(code)) == "ensemble+xgboost"
    assert experience.detect_error_signature("timeout", None, "", "") == "timeout"
    assert experience.detect_error_signature("failed", 400, "ModuleNotFoundError: nope", "") == "import_error"
    assert experience.detect_error_signature("success", 200, "previous attempt timed out", "") is None


def test_board_repeated_errors_ignores_successful_polluted_cards():
    cards = [
        {
            "node_id": "success",
            "step_id": 0,
            "operator": "debug",
            "parents": [],
            "fitness": 0.90,
            "status": "success",
            "status_code": 200,
            "is_buggy": False,
            "method_family_auto": "lightgbm",
            "error_signature": "timeout",
            "delta_vs_parent": None,
        },
        {
            "node_id": "failed",
            "step_id": 1,
            "operator": "debug",
            "parents": [],
            "fitness": None,
            "status": "timeout",
            "status_code": 504,
            "is_buggy": True,
            "method_family_auto": "neural_net",
            "error_signature": "timeout",
            "delta_vs_parent": None,
        },
    ]

    board = experience.build_strategy_board(cards, lower_is_better=False)

    assert board["repeated_errors"] == {"timeout": 1}


def test_improve_experience_memory_includes_parent_card_and_board_stats():
    parent = make_node("import lightgbm", 0.78)
    attach_card(
        parent,
        fitness=0.78,
        delta=0.04,
        rich_summary={
            "method_overview": "LightGBM with target encoding and 5-fold validation.",
            "parent_comparison_experience": "Improved over the parent by adding safer categorical handling.",
        },
    )
    other = make_node("from catboost import CatBoostClassifier", 0.73)
    attach_card(other, fitness=0.73, family="catboost")
    journal = SimpleNamespace(nodes=[parent, other])

    memory = experience.build_operator_experience_memory(
        "improve",
        [parent],
        journal=journal,
        lower_is_better=False,
    )

    assert "Targeted Memory Context for IMPROVE" in memory
    assert f"parent_node_id: {parent.id}" in memory
    assert "score=0.78" in memory
    assert "delta_vs_parent=0.04" in memory
    assert "runtime_seconds=12" in memory
    assert "LightGBM with target encoding and 5-fold validation." in memory
    assert "parent_family_stats: count=1" in memory
    assert "current_best_family: lightgbm" in memory
    assert "underexplored_families: catboost, lightgbm" in memory


def test_crossover_experience_memory_includes_two_cards_and_complementarity():
    node_a = make_node("import lightgbm", 0.80)
    node_b = make_node("from catboost import CatBoostRegressor", 0.77)
    attach_card(
        node_a,
        fitness=0.80,
        family="lightgbm",
        rich_summary={
            "method_overview": "Strong LightGBM tabular baseline.",
            "parent_comparison_experience": "Stable validation score with fast runtime.",
        },
    )
    attach_card(
        node_b,
        fitness=0.77,
        family="catboost",
        rich_summary={
            "method_overview": "CatBoost handles categorical features directly.",
            "parent_comparison_experience": "Slightly weaker score but complementary method family.",
        },
    )
    journal = SimpleNamespace(nodes=[node_a, node_b])

    memory = experience.build_operator_experience_memory(
        "crossover",
        [node_a, node_b],
        journal=journal,
        lower_is_better=False,
    )

    assert "Targeted Memory Context for CROSSOVER" in memory
    assert f"parent_1_node_id: {node_a.id}" in memory
    assert f"parent_2_node_id: {node_b.id}" in memory
    assert "family_complementarity: different_method_families" in memory
    assert "combine lightgbm with catboost" in memory
    assert "Strong LightGBM tabular baseline." in memory
    assert "CatBoost handles categorical features directly." in memory


def test_debug_experience_memory_includes_current_error_and_repeated_errors():
    buggy = make_node("import missing_package", None, is_buggy=True)
    buggy.metric.info = {
        "status": "failed",
        "status_code": 400,
        "raw_run_log": "ModuleNotFoundError: No module named missing_package",
    }
    buggy.experience_card = {
        "node_id": buggy.id,
        "method_family_auto": "unknown",
        "fitness": None,
        "delta_vs_parent": None,
        "status": "failed",
        "is_buggy": True,
        "error_signature": "import_error",
        "sandbox_time_used": 5.0,
        "rich_summary": {
            "method_overview": "The current node imports a missing package before any training starts.",
            "parent_comparison_experience": "Compared with its parent, this attempt fails earlier due to an import error.",
        },
    }
    failed_before = make_node("import missing_package", None, is_buggy=True)
    failed_before.experience_card = {
        "node_id": failed_before.id,
        "method_family_auto": "unknown",
        "fitness": None,
        "status": "failed",
        "is_buggy": True,
        "error_signature": "import_error",
        "operator": "debug",
    }
    journal = SimpleNamespace(nodes=[failed_before])

    memory = experience.build_operator_experience_memory(
        "debug",
        [buggy],
        journal=journal,
        lower_is_better=False,
        current_node=buggy,
    )

    assert "Targeted Memory Context for DEBUG" in memory
    assert "current_error_signature: import_error" in memory
    assert f"current_buggy_node_id: {buggy.id}" in memory
    assert "The current node imports a missing package" in memory
    assert "repeated_errors: import_error=" in memory
    assert f"related_error_node: {failed_before.id}" in memory


def test_collect_operator_memory_nodes_uses_parent_ancestors_and_top_siblings():
    ancestor_1 = attach_card(make_node("import lightgbm", 0.60), fitness=0.60)
    ancestor_2 = attach_card(make_node("import lightgbm", 0.65, parents=[ancestor_1]), fitness=0.65, delta=0.05)
    ancestor_3 = attach_card(make_node("import lightgbm", 0.70, parents=[ancestor_2]), fitness=0.70, delta=0.05)
    parent = attach_card(make_node("import lightgbm", 0.74, parents=[ancestor_3]), fitness=0.74, delta=0.04)
    sibling_best = attach_card(make_node("import lightgbm", 0.73, parents=[ancestor_3]), fitness=0.73, delta=0.03)
    sibling_next = attach_card(make_node("from catboost import CatBoostClassifier", 0.72, parents=[ancestor_3]), fitness=0.72, delta=0.02, family="catboost")
    sibling_low = attach_card(make_node("import sklearn", 0.55, parents=[ancestor_3]), fitness=0.55, delta=-0.15, family="sklearn")
    cousin = attach_card(make_node("import xgboost", 0.90, parents=[ancestor_2]), fitness=0.90, delta=0.25, family="xgboost")
    journal = SimpleNamespace(
        nodes=[
            ancestor_1,
            ancestor_2,
            ancestor_3,
            parent,
            sibling_best,
            sibling_next,
            sibling_low,
            cousin,
        ]
    )

    sections = experience.collect_operator_memory_nodes(
        "improve",
        [parent],
        journal=journal,
        lower_is_better=False,
        sibling_k=2,
        ancestor_k=3,
    )

    assert [node.id for node in sections["primary"]] == [parent.id]
    assert [node.id for node in sections["vertical"]] == [ancestor_3.id, ancestor_2.id, ancestor_1.id]
    assert [node.id for node in sections["horizontal"]] == [sibling_best.id, sibling_next.id]
    assert cousin.id not in [node.id for node in sections["horizontal"]]


def test_prompt_memory_master_switch_disables_targeted_memory():
    assert experience.prompt_memory_enabled({"experience": {"enabled": False}}) is False
    assert experience.prompt_memory_enabled({"experience": {"enabled": True}}) is True
    assert (
        experience.prompt_memory_enabled(
            {
                "experience": {
                    "enabled": True,
                    "prompt_memory": {"enabled": False},
                }
            }
        )
        is False
    )
