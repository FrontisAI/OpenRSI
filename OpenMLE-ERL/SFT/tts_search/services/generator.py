"""
Generator Service - Handles code generation using LLM.

This service is responsible for generating code solutions for ML tasks.
It operates independently from the evaluation service and communicates
through async queues managed by the scheduler.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from httpx import HTTPStatusError
from litellm import ModelResponse, cost_per_token
from litellm.exceptions import (
    APIConnectionError,
    BadRequestError,
    RateLimitError,
    ServiceUnavailableError,
)
from litellm.router import Router
from omegaconf import DictConfig, OmegaConf

from tts_search.reward_func_utils import extract_code

try:
    from litellm.exceptions import MidStreamFallbackError
except ImportError:

    class MidStreamFallbackError(Exception):
        """Compatibility exception for LiteLLM versions without this type."""

        def __init__(
            self,
            message: str,
            *,
            llm_provider: str | None = None,
            model: str | None = None,
            response: Any = None,
        ) -> None:
            super().__init__(message)
            self.llm_provider = llm_provider
            self.model = model
            self.response = response

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)


@dataclass
class GenerationRequest:
    """Request for code generation."""

    request_id: str
    task_id: str
    task_name: str
    messages: list[dict[str, str]]
    model_name: str
    data_dir: str
    metadata: dict[str, Any]
    step_index: int
    output_dir: Path | None = None
    search_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    """Result from code generation."""

    request_id: str
    task_id: str
    task_name: str
    code: str
    raw_text: str
    reasoning_content: str | None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float
    model_name: str
    messages: list[dict[str, str]]
    metadata: dict[str, Any]
    step_index: int
    data_dir: str
    output_dir: Path | None = None
    search_metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def success(self) -> bool:
        return self.error is None and self.code != ""


class GeneratorService:
    """
    Service for code generation using LLM.

    This service handles:
    - Async code generation with retry logic
    - Token usage tracking
    - Cost calculation
    - Output artifact persistence
    """

    def __init__(
        self,
        router: Router,
        concurrency: int = 64,
        max_retries: int = 10,
        model_list: list[Any] | None = None,
    ):
        """
        Initialize the generator service.

        Args:
            router: LiteLLM router for model requests
            concurrency: Maximum concurrent generation requests
            max_retries: Maximum retries for failed requests
        """
        self._router = router
        self._semaphore = asyncio.Semaphore(concurrency)
        self._max_retries = max_retries
        self._model_list = model_list or []
        self._request_queue: asyncio.Queue[GenerationRequest] = asyncio.Queue()
        self._result_queue: asyncio.Queue[GenerationResult] = asyncio.Queue()
        self._running = False
        self._workers: list[asyncio.Task] = []

        # Statistics
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._total_tokens = 0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_cost = 0.0
        self._total_prompt_cost = 0.0
        self._total_response_cost = 0.0

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """
        Generate code for a single request.

        Args:
            request: Generation request with task details

        Returns:
            GenerationResult with generated code and metadata
        """
        raw_text = None
        reasoning_content: str | None = None
        retry_count = 0
        response: ModelResponse | None = None
        usage: dict[str, Any] = {}
        error_msg: str | None = None
        force_stream, include_usage = self._get_streaming_config(request.model_name)

        while raw_text is None or reasoning_content is None:
            if retry_count >= self._max_retries:
                logger.error(
                    "LLM retries exhausted; returning empty generation. task_id=%s task_name=%s step=%d model=%s",
                    request.task_id,
                    request.task_name,
                    request.step_index,
                    request.model_name,
                )
                raw_text = ""
                reasoning_content = None
                usage = {}
                error_msg = (
                    f"LLM retries exhausted after {self._max_retries} attempts"
                )
                break

            reasoning_content = None
            usage = {}
            response = None
            try:
                async with self._semaphore:
                    if force_stream:
                        logger.info("Using streaming mode")
                        raw_text, reasoning_content, usage = (
                            await self._acompletion_stream_collect(
                                model_name=request.model_name,
                                messages=request.messages,
                                include_usage=include_usage,
                            )
                        )
                    else:
                        logger.info("Using non-streaming mode")
                        response = await self._router.acompletion(
                            model=request.model_name,
                            messages=request.messages,  # type: ignore
                        )  # type: ignore
            except RateLimitError as exc:
                retry_count += 1
                logger.warning(
                    "Rate limit hit; retrying. retry=%d task_id=%s task_name=%s step=%d model=%s error=%s",
                    retry_count,
                    request.task_id,
                    request.task_name,
                    request.step_index,
                    request.model_name,
                    exc,
                )
                await asyncio.sleep(1.0 * retry_count)  # Exponential backoff
                continue
            except (ServiceUnavailableError, APIConnectionError) as exc:
                retry_count += 1
                logger.warning(
                    "Provider connection/service error; retrying. retry=%d task_id=%s task_name=%s step=%d model=%s error=%s",
                    retry_count,
                    request.task_id,
                    request.task_name,
                    request.step_index,
                    request.model_name,
                    exc,
                )
                await asyncio.sleep(1.0 * retry_count)
                continue
            except MidStreamFallbackError as exc:
                retry_count += 1
                logger.warning(
                    "Streaming interrupted (mid-stream). retry=%d task_id=%s task_name=%s step=%d model=%s error=%s",
                    retry_count,
                    request.task_id,
                    request.task_name,
                    request.step_index,
                    request.model_name,
                    exc,
                )
                if request.output_dir:
                    step_dir = request.output_dir / f"step_{request.step_index}"
                    step_dir.mkdir(parents=True, exist_ok=True)
                    user_content = (
                        str(request.messages[-1].get("content", ""))
                        if request.messages
                        else ""
                    )
                    (step_dir / "user_prompt.md").write_text(
                        user_content, encoding="utf-8"
                    )
                    (step_dir / "stream_error.txt").write_text(
                        str(exc), encoding="utf-8"
                    )
                await asyncio.sleep(1.0 * retry_count)
                continue
            except HTTPStatusError as exc:
                retry_count += 1
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status in {502, 503, 504}:
                    logger.warning(
                        "Upstream gateway error; retrying. retry=%d task_id=%s task_name=%s step=%d model=%s status=%s",
                        retry_count,
                        request.task_id,
                        request.task_name,
                        request.step_index,
                        request.model_name,
                        status,
                    )
                    await asyncio.sleep(1.0 * retry_count)
                    continue
                raise
            except BadRequestError as exc:
                msg = str(exc)
                if (not force_stream) and (
                    ("Please switch to streaming mode" in msg)
                    or ("generation timed out" in msg)
                ):
                    retry_count += 1
                    logger.warning(
                        "Non-stream timed out; switching to streaming. retry=%d task_id=%s task_name=%s step=%d model=%s error=%s",
                        retry_count,
                        request.task_id,
                        request.task_name,
                        request.step_index,
                        request.model_name,
                        exc,
                    )
                    async with self._semaphore:
                        raw_text, reasoning_content, usage = (
                            await self._acompletion_stream_collect(
                                model_name=request.model_name,
                                messages=request.messages,
                                include_usage=include_usage,
                            )
                        )
                    response = None
                    force_stream = True
                else:
                    raise
            except Exception as exc:
                retry_count += 1
                logger.warning(
                    "Generation error; retrying. retry=%d task_id=%s error=%s",
                    retry_count,
                    request.task_id,
                    exc,
                )
                await asyncio.sleep(1.0)
                continue

            if not force_stream:
                assert response is not None
                message = response.choices[0].message  # type: ignore
                content = message.content  # type: ignore

                if isinstance(content, list):
                    # Anthropic-style content blocks: split text and thinking.
                    logger.debug("Anthropic-style content blocks")
                    text_parts: list[str] = []
                    thinking_parts: list[str] = []
                    for block in content:
                        if block.type == "thinking":
                            thinking_parts.append(block.thinking)
                        elif block.type == "text":
                            text_parts.append(block.text)
                    raw_text = "".join(text_parts) if text_parts else None
                    reasoning_content = (
                        "".join(thinking_parts) if thinking_parts else None
                    )
                    logger.debug(
                        "Anthropic-style reasoning content: %s",
                        (reasoning_content or "")[:100],
                    )
                else:
                    raw_text = str(content) if content is not None else None

                if reasoning_content is None:
                    # OpenAI/LiteLLM reasoning content may live in a separate field.
                    logger.debug("OpenAI/LiteLLM content blocks")

                    # Resolve reasoning fields in provider-neutral priority order.
                    for attr in ("reasoning_content", "reasoning", "thinking"):
                        if hasattr(message, attr):
                            logger.debug(
                                "Found reasoning-like attribute: %s", attr
                            )
                            reasoning_content = getattr(message, attr)
                            break

                    if reasoning_content is None and hasattr(
                        message, "reasoning_details"
                    ):
                        logger.debug("Found reasoning_details attribute")
                        reasoning_content = message.reasoning_details[0]["text"]  # type: ignore

                    if reasoning_content is None and hasattr(
                        message, "provider_specific_fields"
                    ):
                        logger.debug("Found provider_specific_fields attribute")
                        psf = message.provider_specific_fields
                        if isinstance(psf, dict) and "reasoning_details" in psf:
                            logger.debug(
                                "Found reasoning_details in provider_specific_fields"
                            )
                            reasoning_details = psf["reasoning_details"]
                            if (
                                isinstance(reasoning_details, list)
                                and len(reasoning_details) > 0
                            ):
                                # Extract text from all reasoning blocks and concatenate
                                reasoning_texts = []
                                for detail in reasoning_details:
                                    if isinstance(detail, dict) and "text" in detail:
                                        reasoning_texts.append(detail["text"])
                                if reasoning_texts:
                                    reasoning_content = "".join(reasoning_texts)


                usage = response.get("usage", {}) or {}

            if raw_text is None:
                retry_count += 1
                logger.warning(
                    "Empty LLM response; retrying. retry=%d task_id=%s task_name=%s step=%d model=%s",
                    retry_count,
                    request.task_id,
                    request.task_name,
                    request.step_index,
                    request.model_name,
                )
                await asyncio.sleep(1.0 * retry_count)
                continue
            if reasoning_content is None:
                retry_count += 1
                logger.warning(
                    "Reasoning content is None; retrying. retry=%d task_id=%s task_name=%s step=%d model=%s response=%s",
                    retry_count,
                    request.task_id,
                    request.task_name,
                    request.step_index,
                    request.model_name,
                    str(response),
                )
                await asyncio.sleep(1.0 * retry_count)
                continue
        # if raw_text is None:
        #     error_msg = f"Failed to generate after {retry_count} retries"
        #     logger.error(
        #         "Generation failed: task_id=%s task_name=%s error=%s",
        #         request.task_id,
        #         request.task_name,
        #         error_msg,
        #     )
        #     return GenerationResult(
        #         request_id=request.request_id,
        #         task_id=request.task_id,
        #         task_name=request.task_name,
        #         code="",
        #         raw_text="",
        #         reasoning_content=None,
        #         prompt_tokens=0,
        #         completion_tokens=0,
        #         total_tokens=0,
        #         cost=0.0,
        #         model_name=request.model_name,
        #         messages=request.messages,
        #         metadata=request.metadata,
        #         step_index=request.step_index,
        #         data_dir=request.data_dir,
        #         output_dir=request.output_dir,
        #         error=error_msg,
        #     )

        # Extract code from response
        code = extract_code(raw_text)

        # Calculate token usage and cost
        if response is not None:
            usage = response.get("usage", {}) or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens))

        # Calculate costs
        prompt_cost_usd = 0.0
        response_cost_usd = 0.0

        hidden_params = getattr(response, "_hidden_params", None)

        # Try to get prompt_cost and response_cost from hidden_params
        has_prompt_cost = False
        has_response_cost = False
        if isinstance(hidden_params, dict):
            if "response_cost" in hidden_params:
                logger.debug("hidden params has response cost")
                response_cost_usd = float(hidden_params["response_cost"])
                has_response_cost = True

        # Calculate missing costs using cost_per_token
        if not has_prompt_cost or not has_response_cost:
            try:
                calculated_prompt_cost, calculated_completion_cost = cost_per_token(
                    model=request.model_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
                if not has_prompt_cost:
                    prompt_cost_usd = float(calculated_prompt_cost)
                if not has_response_cost:
                    response_cost_usd = float(calculated_completion_cost)
            except Exception as exc:
                logger.warning(
                    "Could not calculate cost for model=%s: %s. Using 0.0 for missing costs.",
                    request.model_name,
                    exc,
                )
        logger.debug(
            f"prompt_cost_usd: {prompt_cost_usd}, response_cost_usd: {response_cost_usd}"
        )

        # Calculate total cost (prompt + response)
        total_cost_usd = prompt_cost_usd + response_cost_usd

        # Save artifacts if output directory is specified
        if request.output_dir:
            step_dir = request.output_dir / f"step_{request.step_index}"
            step_dir.mkdir(parents=True, exist_ok=True)
            user_content = (
                str(request.messages[-1].get("content", "")) if request.messages else ""
            )
            response_payload: dict[str, Any] = {
                "raw_text": raw_text,
                "reasoning_content": reasoning_content,
                "usage": usage,
            }
            if response is not None:
                if hasattr(response, "model_dump"):
                    response_payload["response"] = response.model_dump()
                elif hasattr(response, "dict"):
                    response_payload["response"] = response.dict()
                else:
                    try:
                        response_payload["response"] = dict(response)
                    except Exception:
                        response_payload["response"] = str(response)
            logger.debug(f"response_payload: {str(response_payload)[:50]}")
            (step_dir / "user_prompt.md").write_text(user_content, encoding="utf-8")
            (step_dir / "response.md").write_text(raw_text, encoding="utf-8")
            try:
                response_json = json.dumps(
                    response_payload,
                    indent=2,
                    default=lambda o: o.isoformat() if hasattr(o, "isoformat") else str(o),
                )
            except Exception as e:
                logger.warning(
                    "Failed to serialize response_payload to JSON: %s. Writing minimal payload.",
                    e,
                )
                response_json = json.dumps(
                    {
                        "raw_text": raw_text,
                        "reasoning_content": reasoning_content,
                        "usage": {k: str(v) for k, v in (usage or {}).items()},
                        "response_serialization_error": str(e),
                    },
                    indent=2,
                )
            (step_dir / "response.json").write_text(response_json, encoding="utf-8")
            (step_dir / "valid_code.py").write_text(code, encoding="utf-8")
            if reasoning_content:
                (step_dir / "reasoning.md").write_text(
                    str(reasoning_content), encoding="utf-8"
                )

        # Update statistics
        self._total_requests += 1
        if error_msg is None and code != "":
            self._successful_requests += 1
        else:
            self._failed_requests += 1
        self._total_tokens += total_tokens
        self._total_prompt_tokens += prompt_tokens
        self._total_completion_tokens += completion_tokens
        self._total_cost += total_cost_usd
        self._total_prompt_cost += prompt_cost_usd
        self._total_response_cost += response_cost_usd

        return GenerationResult(
            request_id=request.request_id,
            task_id=request.task_id,
            task_name=request.task_name,
            code=code,
            raw_text=raw_text,
            reasoning_content=reasoning_content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost=total_cost_usd,
            model_name=request.model_name,
            messages=request.messages,
            metadata=request.metadata,
            step_index=request.step_index,
            data_dir=request.data_dir,
            output_dir=request.output_dir,
            search_metadata=dict(request.search_metadata or {}),
            error=error_msg,
        )

    def _to_plain_dict(self, value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, DictConfig):
            return OmegaConf.to_container(value, resolve=True)  # type: ignore
        if isinstance(value, dict):
            return dict(value)
        return {}

    def _get_model_litellm_params(self, model_name: str) -> dict[str, Any]:
        for model_entry in self._model_list:
            entry = self._to_plain_dict(model_entry)
            entry_name = str(entry.get("model_name", ""))
            if entry_name == str(model_name):
                return self._to_plain_dict(entry.get("litellm_params"))
        return {}

    def _get_streaming_config(self, model_name: str) -> tuple[bool, bool]:
        # logger.info(f"model_list: {self._model_list}")
        params = self._get_model_litellm_params(model_name)
        # logger.info(f"params: {params}")
        use_stream = bool(params.get("stream", False))
        include_usage = bool(params.get("include_usage", True))
        return use_stream, include_usage

    async def _acompletion_stream_collect(
        self,
        *,
        model_name: str,
        messages: list[dict[str, str]],
        include_usage: bool = True,
    ) -> tuple[str | None, str | None, dict[str, Any]]:
        stream_options = {"include_usage": True} if include_usage else None
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        usage: dict[str, Any] = {}

        try:
            stream = await self._router.acompletion(
                model=model_name,
                messages=messages,  # type: ignore
                stream=True,
                stream_options=stream_options,
            )  # type: ignore

            async for chunk in stream:
                if isinstance(chunk, dict) and chunk.get("usage"):
                    usage = chunk.get("usage") or {}

                try:
                    choice0 = chunk["choices"][0]
                except Exception:
                    continue

                delta = choice0.get("delta") or {}
                content = delta.get("content")
                if content:
                    text_parts.append(str(content))

                rc = (
                    delta.get("reasoning_content")
                    or delta.get("reasoning")
                    or delta.get("thinking")
                )
                if rc:
                    reasoning_parts.append(str(rc))

                msg = delta.get("message") if isinstance(delta, dict) else None
                if isinstance(msg, dict):
                    rc2 = msg.get("reasoning_content") or msg.get("reasoning")
                    if rc2:
                        reasoning_parts.append(str(rc2))

        except MidStreamFallbackError as exc:
            partial_text = "".join(text_parts) if text_parts else None
            raise MidStreamFallbackError(
                message=f"{exc}\n[partial_text_len={len(partial_text or '')}]",
                llm_provider=getattr(exc, "llm_provider", None),
                model=getattr(exc, "model", None),
                response=getattr(exc, "response", None),
            ) from exc

        raw_text = "".join(text_parts) if text_parts else None
        reasoning_content = "".join(reasoning_parts) if reasoning_parts else None
        logger.info(
            "stream collected: raw_text_len=%d reasoning_content_len=%d raw_head=%r reasoning_head=%r",
            len(raw_text or ""),
            len(reasoning_content or ""),
            (raw_text or "")[:100],
            (reasoning_content or "")[:100],
        )
        return raw_text, reasoning_content, usage

    async def generate_batch(
        self, requests: list[GenerationRequest]
    ) -> list[GenerationResult]:
        """
        Generate code for multiple requests concurrently.

        Args:
            requests: List of generation requests

        Returns:
            List of generation results
        """
        tasks = [self.generate(req) for req in requests]
        return await asyncio.gather(*tasks)

    def get_stats(self) -> dict[str, Any]:
        """Get service statistics."""
        # Calculate per-token costs
        total_cost_per_token = (
            self._total_cost / self._total_tokens if self._total_tokens > 0 else 0.0
        )
        total_prompt_cost_per_token = (
            self._total_prompt_cost / self._total_prompt_tokens
            if self._total_prompt_tokens > 0
            else 0.0
        )
        total_response_cost_per_token = (
            self._total_response_cost / self._total_completion_tokens
            if self._total_completion_tokens > 0
            else 0.0
        )

        return {
            "total_requests": self._total_requests,
            "successful_requests": self._successful_requests,
            "failed_requests": self._failed_requests,
            "total_tokens": self._total_tokens,
            "total_prompt_tokens": self._total_prompt_tokens,
            "total_completion_tokens": self._total_completion_tokens,
            "total_cost": self._total_cost,
            "total_prompt_cost": self._total_prompt_cost,
            "total_response_cost": self._total_response_cost,
            "total_cost_per_token": total_cost_per_token,
            "total_prompt_cost_per_token": total_prompt_cost_per_token,
            "total_response_cost_per_token": total_response_cost_per_token,
        }
