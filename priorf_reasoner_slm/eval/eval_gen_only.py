"""Evaluate generation quality against the M3 prediction schema.

Evaluates how well the student's generated M3 JSON predictions match
the expected schema from ``read.md`` and the ground-truth labels from
the teacher export, focusing on:
- Parse success rate (fraction of outputs that are valid JSON)
- Format correctness under the M3 schema
- Score quality using the generated ``score`` field
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from priorf_reasoner_slm.eval.metrics import compute_all_metrics, find_optimal_threshold
from priorf_reasoner_slm.llm.generation import generate_prediction, parse_prediction_output
from priorf_reasoner_slm.llm.tokenizer_utils import load_tokenizer
from priorf_reasoner_slm.train.train_utils import get_device

logger = logging.getLogger(__name__)


def evaluate_gen(
    teacher_df: pd.DataFrame,
    model: Any,
    tokenizer: Any,
    max_new_tokens: int = 256,
    temperature: float = 0.1,
    device: str | None = None,
    sample_fraction: float = 1.0,
) -> dict[str, Any]:
    """Evaluate generation quality on teacher data.

    Args:
        teacher_df: DataFrame with teacher evidence exports. Must have columns
            ``evidence_card_json`` (or ``card_json``) and ``label``.
        model: The student LLM for generation.
        tokenizer: The model tokenizer.
        max_new_tokens: Max tokens per generation.
        temperature: Sampling temperature.
        device: Device for generation. None uses get_device().
        sample_fraction: Fraction of data to evaluate (for speed). 1.0 = all.

    Returns:
        Dict with metrics:
            - parse_rate: fraction of outputs that parsed as JSON
            - format_correct_rate: fraction with all required fields
            - gen_auroc: AUROC using generated score vs true label
            - gen_auprc: AUPRC using generated score vs true label
            - gen_f1: F1 using generated score vs true label
            - num_samples: total evaluated
            - num_parsed: number successfully parsed
            - num_format_correct: number with correct format
    """
    if device is None:
        device = str(get_device())

    teacher_df = teacher_df[teacher_df["label"].isin([0, 1])].reset_index(drop=True)

    # Sample if requested
    if sample_fraction < 1.0:
        n = max(1, int(len(teacher_df) * sample_fraction))
        teacher_df = teacher_df.sample(n=n, random_state=42).reset_index(drop=True)

    logger.info("Evaluating generation on %d samples", len(teacher_df))

    parse_count = 0
    format_correct_count = 0
    num_missing_card = 0
    num_generation_failed = 0
    num_parse_failed = 0
    num_format_incorrect = 0
    generated_scores: list[float] = []
    true_labels: list[int] = []

    required_fields = {"label", "score", "pattern_hint", "evidence", "rationale"}

    for idx, row in teacher_df.iterrows():
        label = int(row.get("label", 0))

        # Get evidence card JSON
        card_json = row.get("evidence_card_json") or row.get("card_json", "")
        if not card_json:
            logger.warning("Row %d missing evidence_card_json, skipping", idx)
            num_missing_card += 1
            continue

        # Generate prediction
        try:
            gen_text = generate_prediction(
                model=model,
                tokenizer=tokenizer,
                evidence_card_json=card_json,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                device=device,
            )
            parsed = parse_prediction_output(gen_text)
        except Exception as exc:
            logger.warning("Generation failed for row %d: %s", idx, exc)
            num_generation_failed += 1
            parsed = None

        if parsed is None:
            # Failed to parse: do not inject synthetic neutral score.
            num_parse_failed += 1
            continue

        parse_count += 1

        # Check format
        if all(field in parsed for field in required_fields):
            format_correct_count += 1
            score = float(parsed.get("score", 0.5))
            score = max(0.0, min(1.0, score))  # clamp
            generated_scores.append(score)
            true_labels.append(label)
        else:
            # Missing required fields: do not score this sample.
            num_format_incorrect += 1

    n = len(teacher_df)
    num_parsed = parse_count
    num_format_correct = format_correct_count
    parse_rate = parse_count / n if n > 0 else 0.0
    format_correct_rate = format_correct_count / n if n > 0 else 0.0

    # Compute metrics only on successfully scored samples.
    num_scored = len(generated_scores)
    score_coverage = num_scored / n if n > 0 else 0.0
    if num_scored > 0:
        threshold = find_optimal_threshold(true_labels, generated_scores)
        metrics = compute_all_metrics(true_labels, generated_scores, threshold)
    else:
        threshold = 0.5
        metrics = {
            "auroc": 0.5,
            "auprc": 0.5,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "gmeans": 0.0,
        }

    results = {
        "parse_rate": parse_rate,
        "format_correct_rate": format_correct_rate,
        "num_samples": n,
        "num_parsed": num_parsed,
        "num_format_correct": num_format_correct,
        "num_scored": num_scored,
        "score_coverage": score_coverage,
        "num_missing_card": num_missing_card,
        "num_generation_failed": num_generation_failed,
        "num_parse_failed": num_parse_failed,
        "num_format_incorrect": num_format_incorrect,
        "gen_auroc": metrics["auroc"],
        "gen_auprc": metrics["auprc"],
        "gen_f1": metrics["f1"],
        "gen_precision": metrics["precision"],
        "gen_recall": metrics["recall"],
        "gen_gmeans": metrics["gmeans"],
        "optimal_threshold": threshold,
    }

    logger.info(
        "Gen eval: parse_rate=%.3f, format_rate=%.3f, "
        "gen_auroc=%.4f, gen_f1=%.4f",
        parse_rate, format_correct_rate, results["gen_auroc"], results["gen_f1"],
    )

    return results


def evaluate_gen_from_scores_csv(
    csv_path: str,
) -> dict[str, Any]:
    """Evaluate generation from pre-computed scores CSV.

    Expects columns: generated_fraud_prob, label.

    Args:
        csv_path: Path to CSV with pre-computed scores.

    Returns:
        Dict of metrics (same as evaluate_gen, excluding per-sample gen).
    """
    df = pd.read_csv(csv_path)

    required = ["generated_fraud_prob", "label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df = df[df["label"].isin([0, 1])].reset_index(drop=True)
    y_true = df["label"].tolist()
    y_score = df["generated_fraud_prob"].tolist()
    threshold = find_optimal_threshold(y_true, y_score)
    metrics = compute_all_metrics(y_true, y_score, threshold)

    return {
        "num_samples": len(df),
        "gen_auroc": metrics["auroc"],
        "gen_auprc": metrics["auprc"],
        "gen_f1": metrics["f1"],
        "gen_precision": metrics["precision"],
        "gen_recall": metrics["recall"],
        "gen_gmeans": metrics["gmeans"],
        "optimal_threshold": threshold,
    }


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Evaluate generation quality")
    parser.add_argument("--teacher_csv", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, required=True)
    args = parser.parse_args()

    import torch
    from pathlib import Path
    from priorf_reasoner_slm.llm.model_wrapper import load_student_model

    device = get_device()
    model = load_student_model(device=device)
    tokenizer = load_tokenizer()

    df = pd.read_csv(args.teacher_csv)
    results = evaluate_gen(df, model, tokenizer, sample_fraction=0.1)

    print("\n=== Generation Evaluation Results ===")
    for key, value in results.items():
        print(f"  {key}: {value}")
