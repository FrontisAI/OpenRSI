from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    api_key: str | None
    base_url: str | None
    model: str | None

    def require(self, stage: str) -> "OpenAICompatibleConfig":
        missing = [
            name
            for name, value in (
                ("API key", self.api_key),
                ("model", self.model),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(
                f"{stage} LLM configuration is missing: {', '.join(missing)}"
            )
        return self


def _stage_config(prefix: str) -> OpenAICompatibleConfig:
    return OpenAICompatibleConfig(
        api_key=os.environ.get(f"{prefix}_API_KEY") or os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get(f"{prefix}_BASE_URL") or os.environ.get("OPENAI_API_BASE"),
        model=os.environ.get(f"{prefix}_MODEL") or os.environ.get("MODEL"),
    )


def build_llm_config() -> OpenAICompatibleConfig:
    """Resolve builder LLM settings, with legacy OPENAI_* compatibility."""
    return _stage_config("OPENMLE_BUILD_LLM")


def eval_llm_config() -> OpenAICompatibleConfig:
    """Resolve metadata/evaluator LLM settings, with legacy OPENAI_* compatibility."""
    return _stage_config("OPENMLE_EVAL_LLM")


def has_explicit_eval_llm_config() -> bool:
    return any(
        os.environ.get(name)
        for name in (
            "OPENMLE_EVAL_LLM_API_KEY",
            "OPENMLE_EVAL_LLM_BASE_URL",
            "OPENMLE_EVAL_LLM_MODEL",
        )
    )
