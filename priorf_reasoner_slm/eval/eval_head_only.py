"""Evaluate the classification head alone (binary AUROC/AUPRC).

Isolates the classification head performance by computing predictions
using only the cls_head output (no generation, no teacher signal).
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import torch

from priorf_reasoner_slm.eval.metrics import compute_all_metrics, find_optimal_threshold
from priorf_reasoner_slm.llm.cls_head import BinaryClsHead
from priorf_reasoner_slm.llm.model_wrapper import (
    get_hidden_states,
    get_last_token_hidden,
    load_student_model,
)
from priorf_reasoner_slm.llm.tokenizer_utils import load_tokenizer
from priorf_reasoner_slm.train.train_utils import get_device

logger = logging.getLogger(__name__)

# Fallback hidden dimension for Qwen3-4B-Instruct-2507
HIDDEN_DIM = 2560


def evaluate_head(
    teacher_df: pd.DataFrame,
    model: Any,
    cls_head_state_dict: dict[str, Any] | None = None,
    hidden_dim: int = HIDDEN_DIM,
    model_name: str = "Qwen/Qwen3-4B",
    device: str | None = None,
    sample_fraction: float = 1.0,
) -> dict[str, Any]:
    """Evaluate the classification head in isolation.

    Tokenizes evidence cards, runs them through the LLM to get hidden states,
    then passes last-token hidden to the cls_head for binary prediction.

    Args:
        teacher_df: DataFrame with evidence cards and labels.
        model: The student LLM.
        cls_head_state_dict: State dict for cls_head weights.
            Must be provided to avoid evaluating a randomly initialized head.
        hidden_dim: Hidden dimension of the LLM.
        device: Device for computation. None uses get_device().
        sample_fraction: Fraction of data to evaluate. 1.0 = all.

    Returns:
        Dict with head_auroc, head_auprc, head_f1, head_precision,
        head_recall, head_gmeans, optimal_threshold, num_samples.
    """
    if cls_head_state_dict is None:
        raise ValueError(
            "cls_head_state_dict is required for meaningful head evaluation. "
            "Refusing to evaluate with a randomly initialized cls_head."
        )

    if device is None:
        device = str(get_device())

    teacher_df = teacher_df[teacher_df["label"].isin([0, 1])].reset_index(drop=True)

    if sample_fraction < 1.0:
        n = max(1, int(len(teacher_df) * sample_fraction))
        teacher_df = teacher_df.sample(n=n, random_state=42).reset_index(drop=True)

    logger.info("Evaluating cls head on %d samples", len(teacher_df))

    tokenizer = load_tokenizer(model_name)
    cls_head = BinaryClsHead(hidden_dim=hidden_dim, dropout=0.0).to(device)

    cls_head.load_state_dict(cls_head_state_dict)

    cls_head.eval()

    all_probs: list[float] = []
    all_labels: list[int] = []

    model.eval()
    with torch.no_grad():
        for idx, row in teacher_df.iterrows():
            label = int(row.get("label", 0))
            card_json = row.get("evidence_card_json") or row.get("card_json", "")

            if not card_json:
                logger.warning("Row %d missing card JSON, skipping", idx)
                all_probs.append(0.5)
                all_labels.append(label)
                continue

            # Tokenize the card as a single prompt
            encoded = tokenizer(
                card_json,
                truncation=True,
                max_length=2048,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            # Get hidden states
            hidden = get_hidden_states(model, input_ids, attention_mask)
            last_hidden = get_last_token_hidden(hidden, attention_mask)

            # Cls head prediction
            prob = cls_head.predict_proba(last_hidden).item()
            all_probs.append(prob)
            all_labels.append(label)

    threshold = find_optimal_threshold(all_labels, all_probs)
    metrics = compute_all_metrics(all_labels, all_probs, threshold)

    results = {
        "num_samples": len(teacher_df),
        "head_auroc": metrics["auroc"],
        "head_auprc": metrics["auprc"],
        "head_f1": metrics["f1"],
        "head_precision": metrics["precision"],
        "head_recall": metrics["recall"],
        "head_specificity": metrics["specificity"],
        "head_gmeans": metrics["gmeans"],
        "optimal_threshold": threshold,
    }

    logger.info(
        "Head eval: head_auroc=%.4f, head_auprc=%.4f, head_f1=%.4f",
        results["head_auroc"], results["head_auprc"], results["head_f1"],
    )

    return results


def evaluate_head_from_checkpoint(
    teacher_df: pd.DataFrame,
    adapter_path: str,
    cls_head_path: str,
    model_name: str = "Qwen/Qwen3-4B",
    sample_fraction: float = 1.0,
) -> dict[str, Any]:
    """Evaluate the cls head from a co-training checkpoint.

    Args:
        teacher_df: DataFrame with evidence cards and labels.
        adapter_path: Path to the LoRA adapter.
        cls_head_path: Path to the cls_head.pt weights.
        model_name: HuggingFace model name.
        sample_fraction: Fraction to evaluate.

    Returns:
        Dict of head evaluation metrics.
    """
    from priorf_reasoner_slm.train.train_utils import load_adapter

    device = get_device()

    # Load base model + adapter
    base_model = load_student_model(model_name=model_name, device=device)
    model = load_adapter(base_model, adapter_path, device=device)
    model.eval()

    # Load cls head
    cls_head_state_dict = torch.load(cls_head_path, map_location=device)
    hidden_dim = int(
        cls_head_state_dict.get("linear.weight", torch.empty(1, HIDDEN_DIM)).shape[1]
    )

    return evaluate_head(
        teacher_df=teacher_df,
        model=model,
        cls_head_state_dict=cls_head_state_dict,
        hidden_dim=hidden_dim,
        model_name=model_name,
        device=str(device),
        sample_fraction=sample_fraction,
    )


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Evaluate classification head")
    parser.add_argument("--teacher_csv", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--cls_head_path", type=str, required=True)
    args = parser.parse_args()

    if args.teacher_csv.endswith(".parquet"):
        df = pd.read_parquet(args.teacher_csv)
    else:
        df = pd.read_csv(args.teacher_csv)
    results = evaluate_head_from_checkpoint(
        df, args.adapter_path, args.cls_head_path, sample_fraction=0.1
    )

    print("\n=== Classification Head Evaluation ===")
    for key, value in results.items():
        print(f"  {key}: {value}")
