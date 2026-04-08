"""Faithfulness evaluation: sufficiency and comprehensiveness.

Faithfulness metrics measure whether the model's predictions are
actually driven by the provided evidence and teacher signal, or
whether they are spurious.

- Sufficiency: Does removing the evidence change the prediction?
    If removing evidence significantly changes predictions, the evidence
    was sufficient to drive the original prediction.

- Comprehensiveness: Does removing the teacher signal change the prediction?
    If removing the teacher signal significantly changes predictions,
    the teacher signal was comprehensive (necessary).

- Evidence-ablation impact: Systematic study of how ablating different
    evidence components affects the final prediction.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd
import torch

from priorf_reasoner_slm.eval.metrics import (
    compute_all_metrics,
    faithfulness_impact,
    sufficiency_score,
    comprehensiveness_score,
)
from priorf_reasoner_slm.llm.generation import (
    extract_prediction_score,
    generate_prediction,
    parse_prediction_output,
)
from priorf_reasoner_slm.llm.model_wrapper import (
    get_hidden_states,
    get_last_token_hidden,
    load_student_model,
)
from priorf_reasoner_slm.llm.tokenizer_utils import load_tokenizer
from priorf_reasoner_slm.train.train_utils import get_device

logger = logging.getLogger(__name__)


def sufficiency(
    original_scores: list[float],
    ablated_scores: list[float],
) -> float:
    """Compute sufficiency score: mean |original - ablated|.

    Higher score = evidence was more sufficient (removing it changes prediction more).
    Lower score = evidence was NOT sufficient (removal barely changes prediction).

    Args:
        original_scores: Scores with full evidence.
        ablated_scores: Scores after removing evidence.

    Returns:
        Sufficiency score.
    """
    return sufficiency_score(original_scores, ablated_scores)


def comprehensiveness(
    original_scores: list[float],
    ablated_scores: list[float],
) -> float:
    """Compute comprehensiveness score: mean |original - ablated|.

    Higher score = teacher signal was more comprehensive (needed for prediction).
    Lower score = teacher signal was NOT comprehensive (removal barely changes prediction).

    Args:
        original_scores: Scores with teacher signal.
        ablated_scores: Scores after removing teacher signal.

    Returns:
        Comprehensiveness score.
    """
    return comprehensiveness_score(original_scores, ablated_scores)


def evaluate_sufficiency(
    teacher_df: pd.DataFrame,
    model: Any,
    tokenizer: Any,
    evidence_col: str = "evidence_card_json",
    label_col: str = "label",
    device: str | None = None,
    sample_fraction: float = 0.1,
) -> dict[str, Any]:
    """Evaluate sufficiency by ablating evidence and measuring score change.

    For each sample:
    1. Generate prediction WITH evidence (original)
    2. Generate prediction WITHOUT evidence (ablated)
    3. Measure change in fraud probability

    Args:
        teacher_df: DataFrame with evidence cards and labels.
        model: Student LLM.
        tokenizer: Tokenizer.
        evidence_col: Column name for evidence card JSON.
        label_col: Column name for true label.
        device: Device for generation.
        sample_fraction: Fraction of data to evaluate.

    Returns:
        Dict with sufficiency metrics and per-sample scores.
    """
    if device is None:
        device = str(get_device())

    if sample_fraction < 1.0:
        n = max(1, int(len(teacher_df) * sample_fraction))
        teacher_df = teacher_df.sample(n=n, random_state=42).reset_index(drop=True)

    logger.info("Evaluating sufficiency on %d samples", len(teacher_df))

    original_scores: list[float] = []
    ablated_scores: list[float] = []

    for idx, row in teacher_df.iterrows():
        card_json = row.get(evidence_col, "")
        if not card_json:
            logger.warning("Row %d missing evidence, skipping", idx)
            original_scores.append(0.5)
            ablated_scores.append(0.5)
            continue

        # Original: generate with evidence
        try:
            gen_text = generate_prediction(
                model, tokenizer, card_json, device=device
            )
            parsed = parse_prediction_output(gen_text)
            orig_prob = extract_prediction_score(parsed) or 0.5
        except Exception as exc:
            logger.warning("Original generation failed for row %d: %s", idx, exc)
            orig_prob = 0.5

        # Ablated: generate with empty evidence
        empty_card = "{}"
        try:
            gen_text_ablated = generate_prediction(
                model, tokenizer, empty_card, device=device
            )
            parsed_ablated = parse_prediction_output(gen_text_ablated)
            ablated_prob = extract_prediction_score(parsed_ablated) or 0.5
        except Exception as exc:
            logger.warning("Ablated generation failed for row %d: %s", idx, exc)
            ablated_prob = 0.5

        original_scores.append(orig_prob)
        ablated_scores.append(ablated_prob)

    sufficiency_val = sufficiency(original_scores, ablated_scores)
    impact = faithfulness_impact(original_scores, ablated_scores)

    logger.info(
        "Sufficiency eval: sufficiency=%.4f, flip_rate=%.4f",
        sufficiency_val, impact["prediction_flip_rate"],
    )

    return {
        "sufficiency": sufficiency_val,
        "prediction_flip_rate": impact["prediction_flip_rate"],
        "num_samples": len(teacher_df),
        "original_scores": original_scores,
        "ablated_scores": ablated_scores,
    }


def evaluate_comprehensiveness(
    teacher_df: pd.DataFrame,
    model: Any,
    cls_head_weights: dict[str, Any] | None,
    tokenizer: Any,
    evidence_col: str = "evidence_card_json",
    label_col: str = "label",
    hidden_dim: int | None = None,
    device: str | None = None,
    sample_fraction: float = 0.1,
    cls_head: Any | None = None,
) -> dict[str, Any]:
    """Evaluate comprehensiveness by ablating the teacher signal.

    For each sample:
    1. Get prediction WITH teacher signal (full model)
    2. Get prediction WITHOUT teacher signal (zero teacher_prob)
    3. Measure change in fraud probability

    This is evaluated at the classification-head level by setting
    teacher_prob=0 during the distillation loss computation.

    Args:
        teacher_df: DataFrame with evidence cards, labels, and teacher_probs.
        model: Student LLM with LoRA adapter.
        cls_head_weights: Optional state dict for cls_head.
            Required if ``cls_head`` is not provided.
        tokenizer: Tokenizer.
        evidence_col: Column name for evidence card JSON.
        label_col: Column name for true label.
        hidden_dim: Hidden dimension of the LLM.
        device: Device for computation.
        sample_fraction: Fraction to evaluate.
        cls_head: Optional pre-loaded classification head instance.

    Returns:
        Dict with comprehensiveness metrics and per-sample scores.
    """
    from priorf_reasoner_slm.llm.cls_head import BinaryClsHead

    if device is None:
        device = str(get_device())

    if sample_fraction < 1.0:
        n = max(1, int(len(teacher_df) * sample_fraction))
        teacher_df = teacher_df.sample(n=n, random_state=42).reset_index(drop=True)

    logger.info("Evaluating comprehensiveness on %d samples", len(teacher_df))

    if cls_head is None:
        if cls_head_weights is None:
            raise ValueError(
                "Either cls_head_weights or cls_head must be provided for "
                "comprehensiveness evaluation."
            )
        if hidden_dim is None:
            if "linear.weight" in cls_head_weights:
                hidden_dim = int(cls_head_weights["linear.weight"].shape[1])
            else:
                hidden_dim = int(
                    getattr(getattr(model, "config", None), "hidden_size", 2560)
                )
        cls_head = BinaryClsHead(hidden_dim=hidden_dim, dropout=0.0).to(device)
        cls_head.load_state_dict(cls_head_weights)
    else:
        cls_head = cls_head.to(device)
    cls_head.eval()

    original_scores: list[float] = []
    ablated_scores: list[float] = []

    model.eval()
    with torch.no_grad():
        for idx, row in teacher_df.iterrows():
            card_json = row.get(evidence_col, "")
            teacher_prob = float(row.get("teacher_prob", 0.5))

            if not card_json:
                original_scores.append(0.5)
                ablated_scores.append(0.5)
                continue

            # Tokenize
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

            # Original: include teacher signal
            prob_cls = cls_head.predict_proba(last_hidden).item()

            # Ablated: remove teacher signal from fusion
            # fused_with = alpha * cls_prob + (1-alpha) * teacher_prob
            # fused_without = alpha * cls_prob + (1-alpha) * 0 = alpha * cls_prob
            # Using alpha=0.5 as a balanced default for comprehensiveness
            alpha = 0.5
            fused_with = alpha * prob_cls + (1 - alpha) * teacher_prob
            fused_without = alpha * prob_cls  # teacher signal = 0

            original_scores.append(fused_with)
            ablated_scores.append(fused_without)

    comprehensiveness_val = comprehensiveness(original_scores, ablated_scores)
    impact = faithfulness_impact(original_scores, ablated_scores)

    logger.info(
        "Comprehensiveness eval: comprehensiveness=%.4f, flip_rate=%.4f",
        comprehensiveness_val, impact["prediction_flip_rate"],
    )

    return {
        "comprehensiveness": comprehensiveness_val,
        "prediction_flip_rate": impact["prediction_flip_rate"],
        "num_samples": len(teacher_df),
        "original_scores": original_scores,
        "ablated_scores": ablated_scores,
    }


def evidence_ablation_study(
    teacher_df: pd.DataFrame,
    model: Any,
    tokenizer: Any,
    ablation_fields: list[str] | None = None,
    evidence_col: str = "evidence_card_json",
    device: str | None = None,
    sample_fraction: float = 0.05,
) -> pd.DataFrame:
    """Study the impact of ablating different evidence fields on predictions.

    Args:
        teacher_df: DataFrame with evidence cards.
        model: Student LLM.
        tokenizer: Tokenizer.
        ablation_fields: List of fields to ablate individually. If None,
            uses ["anomaly_types", "score_components", "summary_stats"].
        evidence_col: Column name for evidence card JSON.
        device: Device for generation.
        sample_fraction: Fraction to evaluate.

    Returns:
        DataFrame with per-field ablation impact scores.
    """
    if device is None:
        device = str(get_device())

    if ablation_fields is None:
        ablation_fields = ["anomaly_types", "score_components", "summary_stats"]

    if sample_fraction < 1.0:
        n = max(1, int(len(teacher_df) * sample_fraction))
        teacher_df = teacher_df.sample(n=n, random_state=42).reset_index(drop=True)

    logger.info("Evidence ablation study on %d samples, %d fields",
                len(teacher_df), len(ablation_fields))

    results: list[dict[str, Any]] = []

    for idx, row in teacher_df.iterrows():
        card_json = row.get(evidence_col, "")
        if not card_json:
            continue

        try:
            card = json.loads(card_json)
        except json.JSONDecodeError:
            continue

        # Original score
        try:
            orig_text = generate_prediction(model, tokenizer, card_json, device=device)
            orig_parsed = parse_prediction_output(orig_text)
            orig_prob = extract_prediction_score(orig_parsed) or 0.5
        except Exception:
            orig_prob = 0.5

        # Ablate each field
        for field in ablation_fields:
            if field not in card:
                continue

            ablated_card = card.copy()
            ablated_card[field] = [] if field == "anomaly_types" else {}

            try:
                ablated_text = generate_prediction(
                    model, tokenizer, json.dumps(ablated_card), device=device
                )
                ablated_parsed = parse_prediction_output(ablated_text)
                ablated_prob = extract_prediction_score(ablated_parsed) or 0.5
            except Exception:
                ablated_prob = 0.5

            impact = abs(orig_prob - ablated_prob)
            results.append({
                "sample_idx": idx,
                "field": field,
                "original_prob": orig_prob,
                "ablated_prob": ablated_prob,
                "impact": impact,
            })

    results_df = pd.DataFrame(results)

    # Aggregate
    if len(results_df) > 0:
        agg = results_df.groupby("field")["impact"].agg(["mean", "std", "max"]).reset_index()
        agg.columns = ["field", "mean_impact", "std_impact", "max_impact"]
        logger.info("\n%s", agg.to_string(index=False))

    return results_df


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Evaluate faithfulness metrics")
    parser.add_argument("--teacher_csv", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--cls_head_path", type=str, required=True)
    parser.add_argument("--eval_type", type=str, default="sufficiency",
                        choices=["sufficiency", "comprehensiveness", "ablation"])
    args = parser.parse_args()

    import torch
    from priorf_reasoner_slm.train.train_utils import load_adapter

    device = get_device()
    base_model = load_student_model(device=device)
    model = load_adapter(base_model, args.adapter_path, device=device)
    tokenizer = load_tokenizer()

    df = pd.read_csv(args.teacher_csv)

    if args.eval_type == "sufficiency":
        results = evaluate_sufficiency(df, model, tokenizer, sample_fraction=0.1)
    elif args.eval_type == "comprehensiveness":
        cls_head_weights = torch.load(args.cls_head_path, map_location=device)
        results = evaluate_comprehensiveness(
            df, model, cls_head_weights, tokenizer, sample_fraction=0.1
        )
    else:
        results = evidence_ablation_study(df, model, tokenizer, sample_fraction=0.05)

    print(f"\n=== Faithfulness Evaluation ({args.eval_type}) ===")
    for key, value in results.items():
        if key not in ("original_scores", "ablated_scores"):
            print(f"  {key}: {value}")
