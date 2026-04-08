"""Build HuggingFace-compatible SFT datasets from Evidence Cards.

Produces two dataset formats as specified in read.md:

Format 1 (prompt-completion):
    {"prompt": "<evidence card JSON>", "completion": "<M3 output JSON>"}

Format 2 (conversational / messages):
    {"messages": [
        {"role": "system", "content": "<system prompt>"},
        {"role": "user", "content": "<evidence card JSON>"},
        {"role": "assistant", "content": "<M3 output JSON>"}
    ]}

Both formats are compatible with TRL's SFTTrainer.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from priorf_reasoner_slm.evidence.evidence_schema import (
    EVIDENCE_CARD_TASK,
    SYSTEM_PROMPT,
    EvidenceCard,
    PredictionOutput,
)
from priorf_reasoner_slm.evidence.rationale_templates import generate_prediction
from priorf_reasoner_slm.evidence.serializer import build_evidence_card

logger = logging.getLogger(__name__)


# ---- Single-sample conversion ----


def evidence_card_to_prompt(card: EvidenceCard, indent: int | None = 2) -> str:
    """Serialize an EvidenceCard to the prompt string for the student LLM.

    Args:
        card: Validated EvidenceCard instance.
        indent: JSON indentation. Default 2 for readability.

    Returns:
        JSON string of the Evidence Card.
    """
    return card.model_dump_json(indent=indent)


def prediction_to_completion(prediction: PredictionOutput) -> str:
    """Serialize a PredictionOutput to the completion string.

    Produces compact JSON (no indentation) to match the expected
    training output format.

    Args:
        prediction: Validated PredictionOutput instance.

    Returns:
        Compact JSON string.
    """
    return prediction.model_dump_json()


def build_prompt_completion_sample(
    card: EvidenceCard,
    prediction: PredictionOutput | None = None,
) -> dict[str, str]:
    """Build a single prompt-completion sample for SFTTrainer.

    Args:
        card: EvidenceCard input.
        prediction: Optional pre-computed prediction. If None,
            it is generated from the card using rule-based templates.

    Returns:
        Dict with "prompt" and "completion" keys.
    """
    if prediction is None:
        prediction = generate_prediction(card)

    return {
        "prompt": evidence_card_to_prompt(card),
        "completion": prediction_to_completion(prediction),
    }


def build_conversational_sample(
    card: EvidenceCard,
    prediction: PredictionOutput | None = None,
    system_prompt: str = SYSTEM_PROMPT,
) -> dict[str, list[dict[str, str]]]:
    """Build a single conversational (messages) sample.

    Args:
        card: EvidenceCard input.
        prediction: Optional pre-computed prediction.
        system_prompt: System message content.

    Returns:
        Dict with "messages" key containing role/content pairs.
    """
    if prediction is None:
        prediction = generate_prediction(card)

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": evidence_card_to_prompt(card)},
            {"role": "assistant", "content": prediction_to_completion(prediction)},
        ]
    }


# ---- Batch dataset building ----


def build_prompt_completion_dataset(
    df: pd.DataFrame,
    dataset_name: str | None = None,
    splits: list[str] | None = None,
) -> list[dict[str, str]]:
    """Build a list of prompt-completion samples from a teacher export DataFrame.

    Args:
        df: Teacher-exported DataFrame.
        dataset_name: Optional dataset name override.
        splits: Optional filter for splits to include (e.g. ["train", "val"]).
            If None, all rows are included.

    Returns:
        List of {"prompt": ..., "completion": ...} dicts.
    """
    if splits is not None:
        df = df[df["split"].isin(splits)]

    samples: list[dict[str, str]] = []
    for _, row in df.iterrows():
        card = build_evidence_card(row, dataset_name=dataset_name)
        prediction = generate_prediction(card)
        samples.append(build_prompt_completion_sample(card, prediction))

    logger.info(
        "Built %d prompt-completion samples (splits=%s)",
        len(samples),
        splits,
    )
    return samples


def build_conversational_dataset(
    df: pd.DataFrame,
    dataset_name: str | None = None,
    splits: list[str] | None = None,
    system_prompt: str = SYSTEM_PROMPT,
) -> list[dict[str, list[dict[str, str]]]]:
    """Build a list of conversational samples from a teacher export DataFrame.

    Args:
        df: Teacher-exported DataFrame.
        dataset_name: Optional dataset name override.
        splits: Optional filter for splits to include.
        system_prompt: System message for each sample.

    Returns:
        List of {"messages": [...]} dicts.
    """
    if splits is not None:
        df = df[df["split"].isin(splits)]

    samples: list[dict[str, list[dict[str, str]]]] = []
    for _, row in df.iterrows():
        card = build_evidence_card(row, dataset_name=dataset_name)
        prediction = generate_prediction(card)
        samples.append(build_conversational_sample(card, prediction, system_prompt))

    logger.info(
        "Built %d conversational samples (splits=%s)",
        len(samples),
        splits,
    )
    return samples


# ---- HuggingFace Dataset conversion ----


def to_hf_dataset_prompt_completion(
    samples: list[dict[str, str]],
):
    """Convert prompt-completion samples to a HuggingFace Dataset.

    Args:
        samples: Output of build_prompt_completion_dataset.

    Returns:
        datasets.Dataset with "prompt" and "completion" columns.
    """
    from datasets import Dataset

    return Dataset.from_list(samples)


def to_hf_dataset_conversational(
    samples: list[dict[str, list[dict[str, str]]]],
):
    """Convert conversational samples to a HuggingFace Dataset.

    Args:
        samples: Output of build_conversational_dataset.

    Returns:
        datasets.Dataset with "messages" column.
    """
    from datasets import Dataset

    # HF Dataset needs the messages to be serializable; convert inner dicts
    # to strings for storage, then the Dataset will handle them natively.
    # Actually, HF datasets can handle list-of-dict natively via Sequence/Value features.
    flat_samples = []
    for s in samples:
        flat_samples.append({"messages": json.dumps(s["messages"])})

    ds = Dataset.from_list(flat_samples)
    # Map back to proper structure
    ds = ds.map(lambda x: {"messages": json.loads(x["messages"])})
    return ds


# ---- File I/O ----


def save_jsonl(samples: list[dict], path: str | Path) -> Path:
    """Save samples to a JSONL file.

    Args:
        samples: List of sample dicts (prompt-completion or conversational).
        path: Output file path.

    Returns:
        Path to the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    logger.info("Saved %d samples to %s", len(samples), path)
    return path


def load_jsonl(path: str | Path) -> list[dict]:
    """Load samples from a JSONL file.

    Args:
        path: Input file path.

    Returns:
        List of sample dicts.
    """
    path = Path(path)
    samples: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    logger.info("Loaded %d samples from %s", len(samples), path)
    return samples
