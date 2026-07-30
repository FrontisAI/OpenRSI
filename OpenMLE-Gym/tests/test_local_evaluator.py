from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openmle_gym.local_evaluator import (
    _evaluate_task,
    _evaluate_with_ai,
    _inspect_csv,
    _validate_ai_quality,
    analyze_task_structure,
    evaluation_has_task_failures,
)


METRIC_CODE = """
class FixtureMetrics:
    def evaluate(self, y_true=None, y_pred=None):
        return 0.5
"""


def _make_task(root: Path, name: str = "fixture") -> Path:
    task = root / name
    public = task / "data" / "public"
    private = task / "data" / "private"
    utils = task / "utils"
    raw = task / "raw"
    public.mkdir(parents=True)
    private.mkdir(parents=True)
    utils.mkdir(parents=True)
    raw.mkdir(parents=True)
    (public / "description.txt").write_text("A fixture task.", encoding="utf-8")
    (public / "train.csv").write_text(
        'id,text,target\n1,"first line\\nsecond line",0\n2,plain,1\n',
        encoding="utf-8",
    )
    (public / "test.csv").write_text(
        'id,text\n3,"test line\\ncontinued"\n4,plain\n',
        encoding="utf-8",
    )
    (public / "sample_submission.csv").write_text(
        "id,target\n3,0\n4,1\n",
        encoding="utf-8",
    )
    (private / "test_answer.csv").write_text(
        "id,target\n3,1\n4,0\n",
        encoding="utf-8",
    )
    (raw / "train.csv").write_text(
        "id,text,target\n1,a,0\n2,b,1\n3,c,1\n4,d,0\n",
        encoding="utf-8",
    )
    (raw / "test.csv").write_text(
        "id,text\n5,e\n6,f\n",
        encoding="utf-8",
    )
    (utils / "prepare.py").write_text(
        "def prepare(raw, public, private):\n    pass\n",
        encoding="utf-8",
    )
    (utils / "metric.py").write_text(METRIC_CODE, encoding="utf-8")
    return task


def _judge_result() -> dict:
    return {
        "task_validity": 5,
        "task_validity_reason": "The task is meaningful.",
        "data_sufficiency": 4,
        "data_sufficiency_reason": "The fixture is sufficient for this test.",
        "raw_data_usage": 4,
        "raw_data_usage_reason": "Preparation is defined.",
        "task_complexity": 4,
        "task_complexity_reason": "The task is non-trivial.",
        "data_quality": 4,
        "data_quality_reason": "The package is aligned.",
        "major_issues": [],
    }


class LocalEvaluatorTests(unittest.TestCase):
    def test_csv_inspection_counts_logical_records_and_labels_preview(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "multiline.csv"
            path.write_text(
                'id,text\n1,"line one\\nline two"\n2,plain\n',
                encoding="utf-8",
            )
            evidence = _inspect_csv(path, max_preview_rows=1)
        self.assertEqual(evidence["row_count"], 2)
        self.assertEqual(evidence["preview_row_count"], 1)
        self.assertTrue(evidence["preview_is_truncated"])
        self.assertEqual(evidence["preview_rows"][0][1], "line one\\nline two")

    def test_structure_validation_uses_csv_records_and_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = _make_task(Path(directory))
            result = analyze_task_structure(task)
        self.assertEqual(result["train_rows"], 2)
        self.assertEqual(result["test_rows"], 2)
        self.assertEqual(result["validation"]["status"], "passed")
        preview = result["csv_previews"]["test_answer.csv"]
        self.assertEqual(preview["total_rows"], 2)
        self.assertEqual(preview["preview_row_count"], 2)
        usage = result["raw_usage_evidence"]
        self.assertTrue(usage["row_conservation"])
        self.assertEqual(usage["labeled_row_utilization_ratio"], 1.0)
        self.assertEqual(
            usage["raw_test_role"],
            "unlabeled_original_test_intentionally_excluded",
        )

    def test_structural_mismatch_is_a_deterministic_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = _make_task(Path(directory))
            (task / "data" / "public" / "sample_submission.csv").write_text(
                "id,target\n3,0\n",
                encoding="utf-8",
            )
            result = analyze_task_structure(task)
        self.assertEqual(result["validation"]["status"], "failed")
        finding_ids = {
            finding["id"] for finding in result["validation"]["findings"]
        }
        self.assertIn("row_count_mismatch_sample_submission_csv", finding_ids)

    def test_metric_owns_submission_answer_schema_and_identifier_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = _make_task(Path(directory))
            (task / "data" / "public" / "sample_submission.csv").write_text(
                "id,prob_12h,prob_24h\n3,0.1,0.2\n4,0.3,0.4\n",
                encoding="utf-8",
            )
            (task / "data" / "private" / "test_answer.csv").write_text(
                "event_id,time_to_hit_hours,event\n30,12,1\n40,72,0\n",
                encoding="utf-8",
            )
            result = analyze_task_structure(task)
        self.assertEqual(result["metric_check"]["status"], "ok")
        self.assertEqual(result["validation"]["status"], "passed")
        finding_ids = {
            finding["id"] for finding in result["validation"]["findings"]
        }
        self.assertNotIn("submission_answer_schema_mismatch", finding_ids)
        self.assertNotIn("submission_answer_identifier_mismatch", finding_ids)

    def test_online_hard_gate_does_not_call_quality_judge(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = _make_task(Path(directory))
            (task / "data" / "public" / "sample_submission.csv").write_text(
                "id,target\n3,0\n",
                encoding="utf-8",
            )
            with patch(
                "openmle_gym.local_evaluator._evaluate_with_ai"
            ) as evaluator:
                result = _evaluate_task(
                    root_dir=str(task.parent),
                    task=task.name,
                    overview={},
                    local_only=False,
                    skip_llm=False,
                )
        evaluator.assert_not_called()
        self.assertEqual(result["validation"]["status"], "failed")
        self.assertEqual(result["quality_scores"]["overall_score"], 0)
        self.assertEqual(
            result["quality_scores"]["recommendation"],
            "not_recommended",
        )
        self.assertEqual(
            result["quality_evaluation"]["source"],
            "deterministic_hard_gate",
        )

    def test_local_only_performs_no_quality_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = _make_task(Path(directory))
            result = _evaluate_task(
                root_dir=str(task.parent),
                task=task.name,
                overview={},
                local_only=True,
                skip_llm=False,
            )
        self.assertEqual(result["validation"]["status"], "passed")
        self.assertEqual(result["quality_scores"], {})
        self.assertEqual(result["quality_evaluation"]["status"], "skipped")

    def test_llm_failure_is_not_reported_as_not_recommended(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = _make_task(Path(directory))
            with patch(
                "openmle_gym.local_evaluator._evaluate_with_ai",
                side_effect=TimeoutError("provider timeout"),
            ):
                result = _evaluate_task(
                    root_dir=str(task.parent),
                    task=task.name,
                    overview={},
                    local_only=False,
                    skip_llm=False,
                )
        self.assertEqual(result["validation"]["status"], "passed")
        self.assertEqual(result["quality_scores"], {})
        self.assertEqual(result["quality_evaluation"]["status"], "failed")
        self.assertNotIn("recommendation", result["quality_scores"])

        with tempfile.TemporaryDirectory() as directory:
            combined = Path(directory) / "all_results.json"
            combined.write_text(json.dumps([result]), encoding="utf-8")
            self.assertTrue(
                evaluation_has_task_failures({"combined": combined})
            )

    def test_quality_result_is_validated_and_recomputed(self) -> None:
        raw = _judge_result()
        raw["overall_score"] = 0
        raw["recommendation"] = "not_recommended"
        result = _validate_ai_quality(raw)
        self.assertEqual(result["overall_score"], 4.2)
        self.assertEqual(result["recommendation"], "recommended")

    def test_major_issue_requires_structured_evidence(self) -> None:
        raw = _judge_result()
        raw["major_issues"] = ["unsupported claim"]
        with self.assertRaisesRegex(ValueError, "evidence object"):
            _validate_ai_quality(raw)

    def test_prompt_documents_random_submission_and_holdout_contract(self) -> None:
        task_info = {
            "task_name": "fixture",
            "missing_required": [],
            "metric_check": {"status": "ok", "score": 0.1},
            "csv_evidence": {},
            "train_rows": 8,
            "test_rows": 2,
            "file_types": [],
            "raw_size_mb": 1,
            "data_size_mb": 1,
        }
        with (
            patch.dict(
                "os.environ",
                {
                    "OPENMLE_EVAL_LLM_API_KEY": "key",
                    "OPENMLE_EVAL_LLM_MODEL": "model",
                },
                clear=True,
            ),
            patch(
                "openmle_gym.local_evaluator._evaluate_with_openai",
                return_value=_judge_result(),
            ) as evaluator,
        ):
            _evaluate_with_ai(task_info)
        prompt = evaluator.call_args.args[0]
        self.assertIn("random schema-valid predictions", prompt)
        self.assertIn("holdout from the labeled raw training set is expected", prompt)
        self.assertIn(
            "raw_usage_evidence is the authoritative row-usage accounting",
            prompt,
        )
        self.assertIn(
            "Never call it inconsistent, misleading, or mismatched",
            prompt,
        )
        self.assertIn(
            "Do not use outside knowledge about the original competition",
            prompt,
        )
        self.assertIn(
            "Compare its row counts only with processed_csv_evidence, "
            "never with raw_csv_evidence",
            prompt,
        )
        self.assertIn(
            "Treat metric.validate_submission() and metric.evaluate() as one "
            "scoring pipeline",
            prompt,
        )
        self.assertIn(
            "Extra, missing, or duplicate prediction rows are invalid",
            prompt,
        )
        self.assertIn(
            "Report an identifier issue only when the evidence contains an actual",
            prompt,
        )
        self.assertIn('"metric_code": "No metric.py file"', prompt)


if __name__ == "__main__":
    unittest.main()
