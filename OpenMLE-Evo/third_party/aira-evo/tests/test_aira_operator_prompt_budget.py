from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
OPERATORS_ROOT = (
    REPO_ROOT
    / "third_party"
    / "aira-evo"
    / "src"
    / "dojo"
    / "configs"
    / "solver"
    / "operators"
    / "mlebench"
    / "aira_operators"
)
SEARCH_CONFIGS = [
    REPO_ROOT / "tts_search" / "configs" / "search" / "airaevo.yaml",
]


OPERATOR_PROMPTS = [
    "draft.yaml",
    "improve.yaml",
    "crossover.yaml",
    "debug.yaml",
    "improve_experience.yaml",
    "crossover_experience.yaml",
    "debug_experience.yaml",
]
DEBUG_PROMPTS = ["debug.yaml", "debug_experience.yaml"]
NON_DEBUG_PROMPTS = [prompt for prompt in OPERATOR_PROMPTS if prompt not in DEBUG_PROMPTS]
TIMEOUT_RECOVERY_GUIDANCE = "If the previous attempt timed out, reduce compute first"


def test_aira_operator_prompts_include_sandbox_budget_and_submission_guard():
    for prompt_name in OPERATOR_PROMPTS:
        prompt_text = (OPERATORS_ROOT / prompt_name).read_text()

        assert "Be aware of the running time of the code, it should complete within {{execution_timeout}}." in prompt_text
        assert "A single sandbox run must finish within {{execution_timeout}}." in prompt_text
        assert "same row count and required columns/order as sample_submission.csv" in prompt_text
        assert "preserves the sample id/order" in prompt_text
        assert "contains no NaN/inf values" in prompt_text
        assert "- execution_timeout" in prompt_text


def test_timeout_recovery_guidance_is_debug_only():
    for prompt_name in DEBUG_PROMPTS:
        prompt_text = (OPERATORS_ROOT / prompt_name).read_text()
        assert TIMEOUT_RECOVERY_GUIDANCE in prompt_text

    for prompt_name in NON_DEBUG_PROMPTS:
        prompt_text = (OPERATORS_ROOT / prompt_name).read_text()
        assert TIMEOUT_RECOVERY_GUIDANCE not in prompt_text


def test_airaevo_final_configs_default_to_two_hour_execution_timeout():
    for config_path in SEARCH_CONFIGS:
        config_text = config_path.read_text()

        assert "execution_timeout: 7200" in config_text
