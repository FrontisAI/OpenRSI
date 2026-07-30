import sys
import types
from pathlib import Path

if "black" not in sys.modules:
    black_stub = types.ModuleType("black")
    black_stub.FileMode = object
    black_stub.format_str = lambda code, mode=None: code
    sys.modules["black"] = black_stub

from dojo.core.solvers.utils.response import prompt_score_sanitization_enabled
from dojo.core.solvers.utils.response import sanitize_execution_output_for_prompt

OFFICIAL_SCORE_LOG = """Fold 1: 0.91
Final Validation Score: 0.9123

--- JigsawMetrics Submission Scorer ---
Reading prediction file...
Calculating score...
Final Score: 0.4567
prefix ##SCORE##0.4567 suffix

submission.csv Grader Feedback: ## Execution Result
**Status**: completed
**Score**: 0.4567
**Result**: success
"""


REPO_SRC = Path(__file__).resolve().parents[1] / "src"


def compact_source(text: str) -> str:
    return "".join(text.split())


def assert_official_score_removed(prompt_text: str):
    assert "Final Validation Score: 0.9123" in prompt_text
    assert "Final Score: 0.4567" not in prompt_text
    assert "##SCORE##0.4567" not in prompt_text
    assert "**Score**: 0.4567" not in prompt_text
    assert "Official sandbox score redacted" in prompt_text


def test_sanitizer_preserves_self_validation_and_removes_official_score():
    sanitized = sanitize_execution_output_for_prompt(OFFICIAL_SCORE_LOG)

    assert_official_score_removed(sanitized)


def test_sanitizer_can_be_disabled_for_inference_compatibility():
    assert sanitize_execution_output_for_prompt(OFFICIAL_SCORE_LOG, enabled=False) == OFFICIAL_SCORE_LOG


def test_prompt_score_sanitization_follows_experience_switch():
    assert prompt_score_sanitization_enabled({"experience": {"enabled": False}}) is False
    assert prompt_score_sanitization_enabled({}) is False
    assert prompt_score_sanitization_enabled({"experience": {"enabled": True}}) is True
    assert (
        prompt_score_sanitization_enabled(
            {
                "experience": {
                    "enabled": True,
                    "prompt_score_sanitization": {"enabled": False},
                }
            }
        )
        is False
    )


def test_operator_prompt_sources_sanitize_term_out_before_prompting():
    source_expectations = {
        "dojo/core/solvers/operators/analyze.py": [
            "execution_output=sanitize_execution_output_for_prompt(input_node.term_out,enabled=prompt_score_sanitization_enabled(cfg),)",
        ],
        "dojo/core/solvers/operators/improve.py": [
            'sanitize_execution_output_for_prompt(input_node.term_out,enabled=sanitize_prompt_scores,)',
        ],
        "dojo/core/solvers/operators/crossover.py": [
            'sanitize_execution_output_for_prompt(input_node1.term_out,enabled=sanitize_prompt_scores,)',
            'sanitize_execution_output_for_prompt(input_node2.term_out,enabled=sanitize_prompt_scores,)',
        ],
        "dojo/core/solvers/operators/debug.py": [
            "execution_output=sanitize_execution_output_for_prompt(input_node.term_out,enabled=sanitize_prompt_scores,)",
        ],
        "dojo/core/solvers/operators/rich_memory_summary.py": [
            'sanitize_execution_output_for_prompt(getattr(input_node,"term_out","")or"",enabled=sanitize_prompt_scores,)',
            "sanitize_execution_output_for_prompt(",
        ],
    }

    for rel_path, snippets in source_expectations.items():
        source = compact_source((REPO_SRC / rel_path).read_text())
        assert "sanitize_execution_output_for_prompt" in source
        for snippet in snippets:
            assert compact_source(snippet) in source


def test_rich_memory_prompt_source_uses_validation_metric_not_official_score():
    source = compact_source(
        (REPO_SRC / "dojo/core/solvers/operators/rich_memory_summary.py").read_text()
    )

    assert compact_source('"score": _value(current_card.get("fitness"),') in source
    assert compact_source('"parent_score": _value(') in source
    assert compact_source('current_card.get("score")') not in source
    assert compact_source('parent_card.get("score")') not in source
    assert compact_source('current_info.get("score")') not in source
    assert compact_source('parent_info.get("score")') not in source
