from __future__ import annotations

import unittest
from unittest.mock import patch

from openmle_gym.llm_config import build_llm_config, eval_llm_config
from openmle_gym.local_evaluator import _evaluate_with_ai


class LLMConfigTests(unittest.TestCase):
    def test_stage_settings_are_independent(self) -> None:
        environment = {
            "OPENMLE_BUILD_LLM_API_KEY": "build-key",
            "OPENMLE_BUILD_LLM_BASE_URL": "https://build.example/v1",
            "OPENMLE_BUILD_LLM_MODEL": "build-model",
            "OPENMLE_EVAL_LLM_API_KEY": "eval-key",
            "OPENMLE_EVAL_LLM_BASE_URL": "https://eval.example/v1",
            "OPENMLE_EVAL_LLM_MODEL": "eval-model",
        }
        with patch.dict("os.environ", environment, clear=True):
            build = build_llm_config()
            evaluation = eval_llm_config()
        self.assertEqual(build.api_key, "build-key")
        self.assertEqual(build.model, "build-model")
        self.assertEqual(evaluation.api_key, "eval-key")
        self.assertEqual(evaluation.model, "eval-model")

    def test_legacy_openai_settings_are_fallbacks(self) -> None:
        environment = {
            "OPENAI_API_KEY": "legacy-key",
            "OPENAI_API_BASE": "https://legacy.example/v1",
            "MODEL": "legacy-model",
            "OPENMLE_EVAL_LLM_MODEL": "eval-model",
        }
        with patch.dict("os.environ", environment, clear=True):
            evaluation = eval_llm_config()
        self.assertEqual(evaluation.api_key, "legacy-key")
        self.assertEqual(evaluation.base_url, "https://legacy.example/v1")
        self.assertEqual(evaluation.model, "eval-model")

    def test_explicit_eval_settings_select_openai_quality_path(self) -> None:
        task_info = {
            "task_name": "fixture",
            "missing_required": [],
            "metric_check": {"status": "ok"},
            "csv_previews": {},
            "train_rows": 1,
            "test_rows": 1,
            "file_types": [],
            "raw_size_mb": 0,
            "data_size_mb": 0,
        }
        judge_result = {
            "task_validity": 5,
            "task_validity_reason": "valid",
            "data_sufficiency": 4,
            "data_sufficiency_reason": "sufficient",
            "raw_data_usage": 4,
            "raw_data_usage_reason": "uses raw data",
            "task_complexity": 4,
            "task_complexity_reason": "non-trivial",
            "data_quality": 3,
            "data_quality_reason": "usable",
            "major_issues": [],
            # These values must be ignored and recomputed by the evaluator.
            "overall_score": 0,
            "recommendation": "not_recommended",
        }
        with (
            patch.dict(
                "os.environ",
                {
                    "OPENMLE_EVAL_LLM_API_KEY": "eval-key",
                    "OPENMLE_EVAL_LLM_MODEL": "eval-model",
                },
                clear=True,
            ),
            patch(
                "openmle_gym.local_evaluator._evaluate_with_openai",
                return_value=judge_result,
            ) as openai_evaluator,
            patch("openmle_gym.local_evaluator._get_anthropic_client") as anthropic_client,
        ):
            result = _evaluate_with_ai(task_info)
        self.assertEqual(result["overall_score"], 4.0)
        self.assertEqual(result["recommendation"], "recommended")
        openai_evaluator.assert_called_once()
        anthropic_client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
