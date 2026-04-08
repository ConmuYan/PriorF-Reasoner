"""Tokenizer utilities for Qwen3-4B student model.

Provides tokenizer loading with chat-template support, Evidence Card
formatting into Qwen3 chat messages, and chat-template application
that produces proper input_ids / labels tensors for SFT training.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from transformers import AutoTokenizer

from priorf_reasoner_slm.evidence.evidence_schema import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Qwen3 special tokens for non-thinking mode
QWEN3_THINKING_DISABLED = "/no_think"


def load_tokenizer(
    model_name: str = "Qwen/Qwen3-4B",
    max_length: int = 4096,
) -> AutoTokenizer:
    """Load AutoTokenizer with chat-template support for Qwen3-4B.

    Args:
        model_name: HuggingFace model identifier or local path.
        max_length: Maximum sequence length for padding/truncation.

    Returns:
        AutoTokenizer configured for Qwen3 chat format.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.model_max_length = max_length

    logger.info(
        "Loaded tokenizer for %s (vocab=%d, max_length=%d)",
        model_name,
        tokenizer.vocab_size,
        max_length,
    )
    return tokenizer


def format_evidence_card(
    card_json: str | dict[str, Any],
    completion_json: str | None = None,
) -> list[dict[str, str]]:
    """Format an Evidence Card as Qwen3 chat messages.

    Produces the standard 3-turn chat structure:
      1. system  - fraud-reasoning assistant instructions
      2. user    - the Evidence Card JSON
      3. assistant - the prediction JSON (optional, for training)

    When ``completion_json`` is provided, the assistant message is
    included (training mode). When omitted, only system + user are
    returned (inference mode).

    Args:
        card_json: Evidence Card as a JSON string or dict.
        completion_json: Optional M3 prediction JSON string for training.

    Returns:
        List of message dicts with ``role`` and ``content`` keys.
    """
    if isinstance(card_json, dict):
        card_str = json.dumps(card_json, ensure_ascii=False)
    else:
        card_str = card_json

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": card_str},
    ]

    if completion_json is not None:
        messages.append(
            {"role": "assistant", "content": completion_json}
        )

    return messages


def _find_subsequence_start(haystack: list[int], needle: list[int]) -> int:
    """Find the start index of needle within haystack, or len(haystack) if not found.

    Uses a simple linear search (needle is typically short).
    """
    n = len(needle)
    h = len(haystack)
    if n == 0:
        return 0
    if n > h:
        return h
    for i in range(h - n + 1):
        if haystack[i:i + n] == needle:
            return i
    return h  # not found — fallback to end


def apply_chat_template(
    tokenizer: AutoTokenizer,
    messages: list[dict[str, str]],
    max_length: int | None = None,
    add_generation_prompt: bool = False,
    disable_thinking: bool = True,
) -> dict[str, list[int]]:
    """Apply the tokenizer's chat template and prepare SFT inputs.

    For training (assistant message present in ``messages``), this
    produces ``input_ids`` and ``labels`` where prompt tokens have
    label ``-100`` (ignored in loss) and completion tokens retain
    their original token IDs.

    For inference (no assistant message), only ``input_ids`` is
    returned with ``add_generation_prompt=True`` appended.

    Args:
        tokenizer: The loaded AutoTokenizer.
        messages: Chat messages (list of dicts with role/content).
        max_length: Optional override for max sequence length.
        add_generation_prompt: Append generation prompt for inference.
        disable_thinking: Prepend ``/no_think`` to disable Qwen3
            thinking mode.

    Returns:
        Dict with ``input_ids`` (and ``labels`` for training).
    """
    # Determine if this is training (has assistant message) or inference
    has_assistant = any(m["role"] == "assistant" for m in messages)

    if disable_thinking and has_assistant:
        # Insert /no_think into the assistant message content prefix
        for m in messages:
            if m["role"] == "assistant":
                if not m["content"].startswith("/no_think"):
                    m["content"] = QWEN3_THINKING_DISABLED + "\n" + m["content"]
                break
    elif disable_thinking and not has_assistant and add_generation_prompt:
        # For inference with thinking disabled, we add /no_think in generation
        pass

    # Build prompt-only text (without assistant response)
    if has_assistant:
        # Separate prompt messages from completion
        prompt_messages = [m for m in messages if m["role"] != "assistant"]

        # Primary: string-based boundary finding (works consistently for
        # all tokenizers regardless of special-token placement).
        prompt_text_no_gen = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_text_with_gen = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_ids_no_gen = tokenizer.encode(prompt_text_no_gen, add_special_tokens=False)
        prompt_ids_with_gen = tokenizer.encode(prompt_text_with_gen, add_special_tokens=False)
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)

        def _common_prefix_len(a: list[int], b: list[int]) -> int:
            n = min(len(a), len(b))
            i = 0
            while i < n and a[i] == b[i]:
                i += 1
            return i

        assistant_start = max(
            _common_prefix_len(prompt_ids_no_gen, full_ids),
            _common_prefix_len(prompt_ids_with_gen, full_ids),
        )

        # Labels: -100 for prompt tokens, actual ids for completion
        labels = [-100] * assistant_start + full_ids[assistant_start:]

        # Truncate to max_length
        max_len = max_length or tokenizer.model_max_length
        input_ids = full_ids[:max_len]
        labels = labels[:max_len]

        return {
            "input_ids": input_ids,
            "labels": labels,
        }
    else:
        # Inference mode: just tokenize the prompt
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        input_ids = tokenizer.encode(text, add_special_tokens=False)

        max_len = max_length or tokenizer.model_max_length
        input_ids = input_ids[:max_len]

        return {
            "input_ids": input_ids,
        }
