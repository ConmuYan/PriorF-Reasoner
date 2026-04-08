"""Evaluate fused predictions with alpha blending.

Combines student cls-head probability and teacher probability:

    p_final = alpha * p_cls + (1 - alpha) * p_teacher

Supports:
- Fixed alpha evaluation
- Alpha optimization on validation set
- Full metric reporting
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import torch

from priorf_reasoner_slm.eval.metrics import compute_all_metrics, find_optimal_threshold
from priorf_reasoner_slm.llm.fusion import fuse_predictions, optimize_alpha
from priorf_reasoner_slm.train.train_utils import get_device

logger = logging.getLogger(__name__)


def evaluate_fusion(
    teacher_df: pd.DataFrame,
    cls_probs: list[float] | pd.Series,
    teacher_probs: list[float] | pd.Series | None = None,
    alpha: float = 0.5,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Evaluate fused predictions at a fixed alpha.

    Args:
        teacher_df: DataFrame with true labels (column ``label``).
        cls_probs: Student cls-head fraud probabilities.
        teacher_probs: Teacher fraud probabilities. If None, reads from
            ``teacher_prob`` column of teacher_df.
        alpha: Fusion weight for cls_head. Higher = more student weight.
        threshold: Classification threshold. If None, finds optimal.

    Returns:
        Dict with fusion metrics: fusion_auroc, fusion_auprc, fusion_f1,
        fusion_precision, fusion_recall, fusion_gmeans, alpha,
        optimal_threshold (if threshold was None).
    """
    if teacher_probs is None:
        teacher_probs = teacher_df["teacher_prob"].tolist()

    valid_mask = teacher_df["label"].isin([0, 1]).to_numpy()
    teacher_df = teacher_df.loc[valid_mask].reset_index(drop=True)
    y_true = teacher_df["label"].tolist()
    cls_probs_arr = np.asarray(cls_probs)[valid_mask]
    teacher_probs_arr = np.asarray(teacher_probs)[valid_mask]

    # Fuse
    fused_probs = alpha * cls_probs_arr + (1.0 - alpha) * teacher_probs_arr

    if threshold is None:
        threshold = find_optimal_threshold(y_true, fused_probs.tolist())

    metrics = compute_all_metrics(y_true, fused_probs.tolist(), threshold)

    results = {
        "alpha": alpha,
        "fusion_auroc": metrics["auroc"],
        "fusion_auprc": metrics["auprc"],
        "fusion_f1": metrics["f1"],
        "fusion_precision": metrics["precision"],
        "fusion_recall": metrics["recall"],
        "fusion_specificity": metrics["specificity"],
        "fusion_gmeans": metrics["gmeans"],
        "optimal_threshold": threshold,
    }

    logger.info(
        "Fusion eval (alpha=%.2f): auroc=%.4f, auprc=%.4f, f1=%.4f",
        alpha, results["fusion_auroc"], results["fusion_auprc"], results["fusion_f1"],
    )

    return results


def evaluate_fusion_with_optimization(
    teacher_df: pd.DataFrame,
    cls_probs: list[float] | pd.Series,
    teacher_probs: list[float] | pd.Series | None = None,
    n_steps: int = 100,
    holdout_ratio: float = 0.5,
) -> dict[str, Any]:
    """Evaluate fused predictions with alpha optimization on validation set.

    Uses a holdout split to avoid leakage: alpha is optimized on one subset
    and metrics are reported on a separate subset.

    Args:
        teacher_df: DataFrame with true labels (column ``label``).
        cls_probs: Student cls-head probabilities.
        teacher_probs: Teacher probabilities. If None, reads from
            ``teacher_prob`` column of teacher_df.
        n_steps: Number of alpha values to search.
        holdout_ratio: Fraction of data for alpha optimization (remaining
            half is used for unbiased metric evaluation). Default 0.5.

    Returns:
        Dict with optimal_alpha, best_accuracy, fusion_auroc, fusion_auprc,
        fusion_f1, and all other standard metrics at the optimal alpha.
    """
    if teacher_probs is None:
        teacher_probs = teacher_df["teacher_prob"].tolist()

    valid_mask = teacher_df["label"].isin([0, 1]).to_numpy()
    teacher_df = teacher_df.loc[valid_mask].reset_index(drop=True)
    y_true = np.asarray(teacher_df["label"].tolist())
    cls_probs_arr = np.asarray(cls_probs)[valid_mask]
    teacher_probs_arr = np.asarray(teacher_probs)[valid_mask]

    # Stratified holdout split to avoid leakage
    pos_mask = y_true == 1
    neg_mask = ~pos_mask
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()
    n_holdout_pos = max(1, int(n_pos * holdout_ratio))
    n_holdout_neg = max(1, int(n_neg * holdout_ratio))

    pos_idx = np.where(pos_mask)[0]
    neg_idx = np.where(neg_mask)[0]
    rng = np.random.RandomState(42)
    holdout_pos_idx = set(rng.choice(pos_idx, n_holdout_pos, replace=False))
    holdout_neg_idx = set(rng.choice(neg_idx, n_holdout_neg, replace=False))
    holdout_idx = np.array(list(holdout_pos_idx | holdout_neg_idx))
    all_idx = set(range(len(y_true)))
    eval_idx = np.array(list(all_idx - set(holdout_idx.tolist())))

    # Alpha optimization on holdout subset
    opt_cls = cls_probs_arr[holdout_idx]
    opt_teacher = teacher_probs_arr[holdout_idx]
    opt_labels = y_true[holdout_idx]
    optimal_alpha, best_acc = optimize_alpha(
        opt_cls, opt_teacher, opt_labels, n_steps=n_steps
    )

    logger.info("Optimal alpha: %.4f (holdout accuracy: %.4f)", optimal_alpha, best_acc)

    # Evaluate at optimal alpha on held-out evaluation subset
    eval_cls = cls_probs_arr[eval_idx]
    eval_teacher = teacher_probs_arr[eval_idx]
    eval_labels = y_true[eval_idx]
    fused_probs = optimal_alpha * eval_cls + (1.0 - optimal_alpha) * eval_teacher
    threshold = find_optimal_threshold(eval_labels, fused_probs.tolist())
    metrics = compute_all_metrics(eval_labels, fused_probs.tolist(), threshold)

    results = {
        "optimal_alpha": optimal_alpha,
        "best_holdout_accuracy": best_acc,
        "fusion_auroc": metrics["auroc"],
        "fusion_auprc": metrics["auprc"],
        "fusion_f1": metrics["f1"],
        "fusion_precision": metrics["precision"],
        "fusion_recall": metrics["recall"],
        "fusion_specificity": metrics["specificity"],
        "fusion_gmeans": metrics["gmeans"],
        "optimal_threshold": threshold,
        "holdout_size": len(holdout_idx),
        "eval_size": len(eval_idx),
    }

    return results


def evaluate_alpha_sweep(
    teacher_df: pd.DataFrame,
    cls_probs: list[float] | pd.Series,
    teacher_probs: list[float] | pd.Series | None = None,
    n_steps: int = 20,
) -> list[dict[str, Any]]:
    """Sweep alpha values and return metrics at each step.

    Useful for analyzing the tradeoff between student and teacher.

    Args:
        teacher_df: DataFrame with true labels.
        cls_probs: Student cls-head probabilities.
        teacher_probs: Teacher probabilities. If None, reads from df.
        n_steps: Number of alpha values (0.0, 0.05, ..., 1.0).

    Returns:
        List of metric dicts, one per alpha value.
    """
    if teacher_probs is None:
        teacher_probs = teacher_df["teacher_prob"].tolist()

    valid_mask = teacher_df["label"].isin([0, 1]).to_numpy()
    teacher_df = teacher_df.loc[valid_mask].reset_index(drop=True)
    y_true = teacher_df["label"].tolist()
    cls_probs_arr = np.asarray(cls_probs)[valid_mask]
    teacher_probs_arr = np.asarray(teacher_probs)[valid_mask]

    results: list[dict[str, Any]] = []

    for step in range(n_steps + 1):
        alpha = step / n_steps
        fused = alpha * cls_probs_arr + (1.0 - alpha) * teacher_probs_arr
        threshold = find_optimal_threshold(y_true, fused.tolist())
        metrics = compute_all_metrics(y_true, fused.tolist(), threshold)

        results.append({
            "alpha": alpha,
            "auroc": metrics["auroc"],
            "auprc": metrics["auprc"],
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "gmeans": metrics["gmeans"],
        })

    return results


def evaluate_fusion_from_csv(
    csv_path: str,
    alpha: float | None = None,
) -> dict[str, Any]:
    """Evaluate fusion from a pre-computed CSV.

    Expects columns: label, cls_prob, teacher_prob.

    Args:
        csv_path: Path to CSV file.
        alpha: Fixed alpha. If None, optimizes alpha.

    Returns:
        Fusion evaluation metrics dict.
    """
    df = pd.read_csv(csv_path)

    required = ["label", "cls_prob", "teacher_prob"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df = df[df["label"].isin([0, 1])].reset_index(drop=True)
    if alpha is None:
        return evaluate_fusion_with_optimization(
            df, df["cls_prob"], df["teacher_prob"]
        )
    else:
        return evaluate_fusion(df, df["cls_prob"], df["teacher_prob"], alpha=alpha)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Evaluate fused predictions")
    parser.add_argument("--csv", type=str, required=True, help="CSV with label, cls_prob, teacher_prob")
    parser.add_argument("--alpha", type=float, default=None, help="Fixed alpha (optimizes if omitted)")
    args = parser.parse_args()

    results = evaluate_fusion_from_csv(args.csv, alpha=args.alpha)

    print("\n=== Fusion Evaluation Results ===")
    for key, value in results.items():
        print(f"  {key}: {value}")
