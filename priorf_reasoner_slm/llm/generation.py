"""Generation utilities for the PriorF-Reasoner student model.

Provides single-sample and batch generation of M3 prediction JSON,
along with parsing and validation of the model output.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import torch
from transformers import PreTrainedModel

from priorf_reasoner_slm.evidence.evidence_schema import PredictionOutput
from priorf_reasoner_slm.llm.tokenizer_utils import (
    apply_chat_template,
    format_evidence_card,
    load_tokenizer,
    QWEN3_THINKING_DISABLED,
)

logger = logging.getLogger(__name__)


def _normalize_pattern_hint(value: Any) -> str | None:
    """Map free-form pattern descriptions onto the strict M3 enum."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if "camouflage" in text:
        return "camouflage"
    if "co-attack" in text or "co attack" in text or "coordinated" in text:
        return "co-attack"
    if text == "suspicious":
        return "suspicious"
    # Free-form descriptions that are not one of the named patterns are
    # mapped to the fallback M3 label.
    return "suspicious"


def _normalize_label(value: Any) -> str | None:
    """Map label aliases onto the strict M3 enum."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"fraud", "fraudulent", "anomalous", "anomaly", "positive"}:
        return "fraud"
    if text in {"benign", "normal", "negative", "non-fraud", "nonfraud"}:
        return "benign"
    return None


def normalize_prediction_output(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize common alias schemas into the strict M3 output shape."""
    if not isinstance(payload, dict):
        return payload

    candidate = dict(payload)
    if "label" not in candidate:
        candidate["label"] = _normalize_label(candidate.get("fraud_label"))

    if "score" not in candidate:
        for key in ("score", "confidence", "fraud_prob", "probability"):
            if key in candidate:
                candidate["score"] = candidate[key]
                break

    if "pattern_hint" not in candidate:
        pattern_value = candidate.get("behavioral_pattern")
        if pattern_value is None and "anomaly_types" in candidate:
            anomaly_types = candidate.get("anomaly_types") or []
            if isinstance(anomaly_types, list) and anomaly_types:
                pattern_value = anomaly_types[0]
        candidate["pattern_hint"] = _normalize_pattern_hint(pattern_value)
    else:
        candidate["pattern_hint"] = _normalize_pattern_hint(candidate["pattern_hint"])

    if "evidence" not in candidate:
        evidence_value = candidate.get("key_evidence")
        if isinstance(evidence_value, str):
            candidate["evidence"] = [evidence_value]
        elif evidence_value is not None:
            candidate["evidence"] = evidence_value

    try:
        validated = PredictionOutput.model_validate(candidate)
        return validated.model_dump()
    except Exception:
        return payload


def extract_prediction_score(parsed: dict[str, Any] | None) -> float | None:
    """Extract a normalized fraud score from a parsed prediction object."""
    if parsed is None:
        return None
    normalized = normalize_prediction_output(parsed)
    score = normalized.get("score")
    if score is None:
        return None
    try:
        return float(score)
    except (TypeError, ValueError):
        return None


@torch.no_grad()
def generate_prediction(
    model: PreTrainedModel,
    tokenizer: Any,
    evidence_card_json: str | dict[str, Any],
    max_new_tokens: int = 256,
    temperature: float = 0.1,
    top_p: float = 0.9,
    device: str | torch.device = "cpu",
) -> str:
    """Generate an M3 prediction JSON string from an Evidence Card.

    Formats the Evidence Card as a chat prompt, runs generation, and
    returns the raw text output.

    Args:
        model: The student LLM model.
        tokenizer: The corresponding tokenizer.
        evidence_card_json: Evidence Card as JSON string or dict.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        device: Device for input tensors.

    Returns:
        Generated text (should be valid M3 JSON).
    """
    messages = format_evidence_card(evidence_card_json)
    encoded = apply_chat_template(
        tokenizer,
        messages,
        add_generation_prompt=True,
        disable_thinking=True,
    )

    input_ids = torch.tensor(
        [encoded["input_ids"]], dtype=torch.long, device=device
    )
    attention_mask = torch.ones_like(input_ids)

    # Build generation kwargs
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "attention_mask": attention_mask,
    }

    output_ids = model.generate(input_ids, **gen_kwargs)

    # Decode only the generated tokens (skip the input)
    generated_ids = output_ids[0][input_ids.shape[1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return text.strip()


def parse_prediction_output(text: str) -> dict[str, Any] | None:
    """Parse and validate an M3 prediction JSON output.

    Attempts to extract a valid JSON object from the model output.
    Handles common failure modes: markdown code fences, extra text
    before/after JSON, and partial JSON.

    Args:
        text: Raw model output text.

    Returns:
        Parsed dict if valid JSON found, None otherwise.
    """
    # Strip markdown code fences if present
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first line (```json or ```) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)

    # Try direct JSON parse
    try:
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return normalize_prediction_output(result)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the text
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            result = json.loads(cleaned[start : end + 1])
            if isinstance(result, dict):
                return normalize_prediction_output(result)
        except json.JSONDecodeError:
            pass

    logger.warning("Failed to parse prediction output: %s", text[:200])
    return None


@torch.no_grad()
def batch_generate(
    model: PreTrainedModel,
    tokenizer: Any,
    cards: list[str | dict[str, Any]],
    max_new_tokens: int = 256,
    temperature: float = 0.1,
    top_p: float = 0.9,
    batch_size: int = 4,
    device: str | torch.device = "cpu",
) -> list[str]:
    """Batch generation of M3 predictions for multiple Evidence Cards.

    Processes cards in mini-batches for efficiency.

    Args:
        model: The student LLM model.
        tokenizer: The corresponding tokenizer.
        cards: List of Evidence Card JSON strings or dicts.
        max_new_tokens: Maximum tokens per generation.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        batch_size: Number of cards per forward pass.
        device: Device for input tensors.

    Returns:
        List of generated text strings, one per card.
    """
    results: list[str] = []

    for i in range(0, len(cards), batch_size):
        batch_cards = cards[i : i + batch_size]
        batch_texts: list[str] = []

        for card in batch_cards:
            text = generate_prediction(
                model,
                tokenizer,
                card,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                device=device,
            )
            batch_texts.append(text)

        results.extend(batch_texts)

    return results
