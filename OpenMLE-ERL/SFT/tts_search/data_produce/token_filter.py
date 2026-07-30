"""Tokenizer-based length filtering for chat SFT rows."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from tts_search.data_produce.common import summary_stats

TOKENIZER: Any = None


def _import_auto_tokenizer() -> Any:
    """Import AutoTokenizer in this local env with a known metadata mismatch."""
    import importlib.metadata as importlib_metadata

    original_version = importlib_metadata.version

    def patched_version(distribution_name: str) -> str:
        normalized = distribution_name.lower().replace("_", "-")
        if normalized == "tokenizers":
            version = original_version(distribution_name)
            if version.startswith("0.22."):
                return "0.21.4"
        if normalized == "huggingface-hub":
            version = original_version(distribution_name)
            if version.startswith("1."):
                return "0.36.0"
        return original_version(distribution_name)

    importlib_metadata.version = patched_version
    try:
        from transformers import AutoTokenizer
    finally:
        importlib_metadata.version = original_version
    return AutoTokenizer


@dataclass(frozen=True)
class TokenFilterConfig:
    """Tokenizer length filtering configuration.

    Args:
        tokenizer_model: Local tokenizer path or model id.
        max_total_tokens: Maximum allowed chat-template token count.
        workers: Number of tokenization worker processes.
        chunksize: Multiprocessing chunk size.
        keep_equal: Keep rows exactly equal to the token limit.
        local_files_only: Load tokenizer without network access.

    Returns:
        Immutable token filtering configuration.
    """

    tokenizer_model: str | Path
    max_total_tokens: int = 32768
    workers: int = 1
    chunksize: int = 32
    keep_equal: bool = False
    local_files_only: bool = True


def _init_worker(tokenizer_model: str, local_files_only: bool) -> None:
    """Initialize a process-local tokenizer.

    Args:
        tokenizer_model: Tokenizer path or model id.
        local_files_only: Whether transformers may access the network.

    Returns:
        None. Stores the tokenizer in the module-level worker cache.
    """
    global TOKENIZER
    AutoTokenizer = _import_auto_tokenizer()

    TOKENIZER = AutoTokenizer.from_pretrained(
        tokenizer_model,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )


def load_tokenizer(tokenizer_model: str | Path, local_files_only: bool = True) -> Any:
    """Load the tokenizer used for SLIME message token accounting."""
    AutoTokenizer = _import_auto_tokenizer()

    return AutoTokenizer.from_pretrained(
        str(tokenizer_model),
        trust_remote_code=True,
        local_files_only=local_files_only,
    )


def count_chat_template_tokens(messages: list[dict[str, str]], tokenizer: Any) -> int:
    """Count final SLIME chat-template tokens for one message list."""
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    return int(len(ids))


def token_count_within_limit(
    token_count: Any,
    *,
    max_total_tokens: int = 32768,
    keep_equal: bool = True,
) -> bool:
    """Return whether a token count passes the configured SLIME limit."""
    try:
        count = int(token_count)
    except (TypeError, ValueError):
        return False
    return count <= max_total_tokens if keep_equal else count < max_total_tokens


def _count_one(item: tuple[int, list[dict[str, str]]]) -> tuple[int, int]:
    """Count tokens for one chat row.

    Args:
        item: Tuple of row index and chat messages.

    Returns:
        Tuple of row index and token count.
    """
    idx, messages = item
    return idx, count_chat_template_tokens(messages, TOKENIZER)


def count_message_tokens(
    rows: list[dict[str, Any]],
    config: TokenFilterConfig,
) -> list[int]:
    """Return exact chat-template token counts for each row.

    Args:
        rows: SFT rows containing ``messages``.
        config: Tokenizer and length-filter configuration.

    Returns:
        Token counts aligned with input row order.
    """

    inputs = [(idx, row["messages"]) for idx, row in enumerate(rows)]
    token_lengths = [0] * len(rows)
    if config.workers <= 1:
        _init_worker(str(config.tokenizer_model), config.local_files_only)
        iterator = map(_count_one, inputs)
        for idx, length in tqdm(
            iterator,
            total=len(inputs),
            desc="Tokenize messages",
            unit="row",
            dynamic_ncols=True,
        ):
            token_lengths[idx] = length
        return token_lengths

    with ProcessPoolExecutor(
        max_workers=config.workers,
        initializer=_init_worker,
        initargs=(str(config.tokenizer_model), config.local_files_only),
    ) as executor:
        iterator = executor.map(_count_one, inputs, chunksize=config.chunksize)
        for idx, length in tqdm(
            iterator,
            total=len(inputs),
            desc="Tokenize messages",
            unit="row",
            dynamic_ncols=True,
        ):
            token_lengths[idx] = length
    return token_lengths


def filter_by_token_length(
    rows: list[dict[str, Any]],
    config: TokenFilterConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Annotate rows with token length and split into kept/dropped rows.

    Args:
        rows: Candidate SFT rows.
        config: Tokenizer and length-filter configuration.

    Returns:
        Kept rows, dropped rows, and summary statistics.
    """

    token_lengths = count_message_tokens(rows, config)
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for row, length in zip(rows, token_lengths, strict=True):
        annotated = dict(row)
        annotated["token_length"] = int(length)
        keep = token_count_within_limit(
            length,
            max_total_tokens=config.max_total_tokens,
            keep_equal=config.keep_equal,
        )
        annotated["kept_token_filter"] = bool(keep)
        if keep:
            kept.append(annotated)
        else:
            dropped.append(annotated)

    stats = {
        "tokenizer_model": str(config.tokenizer_model),
        "max_total_tokens": config.max_total_tokens,
        "keep_equal": config.keep_equal,
        "before": len(rows),
        "after": len(kept),
        "dropped": len(dropped),
        "token_length": summary_stats(token_lengths),
    }
    return kept, dropped, stats
