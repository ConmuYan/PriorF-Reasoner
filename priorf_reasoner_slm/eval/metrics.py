"""Evaluation metrics for PriorF-Reasoner fraud detection.

Provides standard classification metrics and faithfulness metrics
(sufficiency and comprehensiveness).

Metrics:
- AUROC: Area Under the Receiver Operating Characteristic curve
- AUPRC: Area Under the Precision-Recall Curve
- F1: F1 score at the optimal threshold
- Precision: Precision at the optimal threshold
- Recall: Recall at the optimal threshold
- G-means: Geometric mean of sensitivity and specificity
- Sufficiency: Does removing evidence change the prediction?
- Comprehensiveness: Does removing the teacher signal change the prediction?
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.metrics import (
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def compute_auroc(y_true: list[int], y_score: list[float]) -> float:
    """Compute AUROC (Area Under the ROC Curve).

    Args:
        y_true: Ground-truth binary labels (0=benign, 1=fraud).
        y_score: Predicted fraud scores/probabilities.

    Returns:
        AUROC value in [0, 1].
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if len(np.unique(y_true)) < 2:
        logger.warning("Only one class present in y_true; returning 0.5")
        return 0.5

    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(auc(fpr, tpr))


def compute_auprc(y_true: list[int], y_score: list[float]) -> float:
    """Compute AUPRC (Area Under the Precision-Recall Curve).

    Args:
        y_true: Ground-truth binary labels.
        y_score: Predicted fraud scores/probabilities.

    Returns:
        AUPRC value in [0, 1].
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if len(np.unique(y_true)) < 2:
        logger.warning("Only one class present in y_true; returning 0.5")
        return 0.5

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return float(auc(recall, precision))


def find_optimal_threshold(y_true: list[int], y_score: list[float]) -> float:
    """Find the optimal classification threshold using F1 score.

    Args:
        y_true: Ground-truth binary labels.
        y_score: Predicted fraud scores/probabilities.

    Returns:
        The threshold that maximizes F1 score.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    thresholds = np.linspace(0, 1, 100)
    best_f1 = 0.0
    best_thresh = 0.5

    for thresh in thresholds:
        preds = (y_score >= thresh).astype(int)
        if preds.sum() == 0:
            continue
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    return float(best_thresh)


def compute_f1(
    y_true: list[int],
    y_score: list[float],
    threshold: float | None = None,
) -> float:
    """Compute F1 score at a given or optimal threshold.

    Args:
        y_true: Ground-truth binary labels.
        y_score: Predicted fraud scores/probabilities.
        threshold: Classification threshold. If None, finds optimal.

    Returns:
        F1 score in [0, 1].
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if threshold is None:
        threshold = find_optimal_threshold(y_true, y_score)

    preds = (y_score >= threshold).astype(int)
    return float(f1_score(y_true, preds, zero_division=0))


def compute_precision(
    y_true: list[int],
    y_score: list[float],
    threshold: float | None = None,
) -> float:
    """Compute precision at a given or optimal threshold.

    Args:
        y_true: Ground-truth binary labels.
        y_score: Predicted fraud scores/probabilities.
        threshold: Classification threshold. If None, finds optimal.

    Returns:
        Precision in [0, 1].
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if threshold is None:
        threshold = find_optimal_threshold(y_true, y_score)

    preds = (y_score >= threshold).astype(int)
    return float(precision_score(y_true, preds, zero_division=0))


def compute_recall(
    y_true: list[int],
    y_score: list[float],
    threshold: float | None = None,
) -> float:
    """Compute recall (sensitivity) at a given or optimal threshold.

    Args:
        y_true: Ground-truth binary labels.
        y_score: Predicted fraud scores/probabilities.
        threshold: Classification threshold. If None, finds optimal.

    Returns:
        Recall in [0, 1].
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if threshold is None:
        threshold = find_optimal_threshold(y_true, y_score)

    preds = (y_score >= threshold).astype(int)
    return float(recall_score(y_true, preds, zero_division=0))


def compute_gmeans(y_true: list[int], y_score: list[float]) -> float:
    """Compute G-means = sqrt(sensitivity * specificity).

    Uses the optimal F1 threshold.

    Args:
        y_true: Ground-truth binary labels.
        y_score: Predicted fraud scores/probabilities.

    Returns:
        G-means in [0, 1].
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    threshold = find_optimal_threshold(y_true, y_score)
    preds = (y_score >= threshold).astype(int)

    tp = ((preds == 1) & (y_true == 1)).sum()
    tn = ((preds == 0) & (y_true == 0)).sum()
    fp = ((preds == 1) & (y_true == 0)).sum()
    fn = ((preds == 0) & (y_true == 1)).sum()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    gmeans = np.sqrt(sensitivity * specificity)
    return float(gmeans)


def compute_specificity(
    y_true: list[int],
    y_score: list[float],
    threshold: float | None = None,
) -> float:
    """Compute specificity at a given or optimal threshold.

    Args:
        y_true: Ground-truth binary labels.
        y_score: Predicted fraud scores/probabilities.
        threshold: Classification threshold. If None, finds optimal.

    Returns:
        Specificity in [0, 1].
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if threshold is None:
        threshold = find_optimal_threshold(y_true, y_score)

    preds = (y_score >= threshold).astype(int)

    tn = ((preds == 0) & (y_true == 0)).sum()
    fp = ((preds == 1) & (y_true == 0)).sum()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return float(specificity)


def compute_all_metrics(
    y_true: list[int],
    y_score: list[float],
    threshold: float | None = None,
) -> dict[str, float]:
    """Compute all standard classification metrics.

    Args:
        y_true: Ground-truth binary labels.
        y_score: Predicted fraud scores/probabilities.
        threshold: Classification threshold. If None, finds optimal.

    Returns:
        Dict with keys: auroc, auprc, f1, precision, recall, specificity,
        gmeans, optimal_threshold.
    """
    if threshold is None:
        threshold = find_optimal_threshold(y_true, y_score)

    return {
        "auroc": compute_auroc(y_true, y_score),
        "auprc": compute_auprc(y_true, y_score),
        "f1": compute_f1(y_true, y_score, threshold),
        "precision": compute_precision(y_true, y_score, threshold),
        "recall": compute_recall(y_true, y_score, threshold),
        "specificity": compute_specificity(y_true, y_score, threshold),
        "gmeans": compute_gmeans(y_true, y_score),
        "optimal_threshold": threshold,
    }


# ---- Faithfulness metrics ----


def sufficiency_score(
    original_scores: list[float],
    ablated_scores: list[float],
) -> float:
    """Compute sufficiency: does the evidence drive the prediction above chance?

    Sufficiency measures whether the evidence drives the prediction above
    the chance level (0.5). Evidence at the chance level provides no
    discriminative signal regardless of the original prediction level.
    Penalizes predictions near 0.5 (chance).

    Score = mean(max(min(original, ablated) - 0.5, 0)) * 2

    Args:
        original_scores: Scores with full evidence.
        ablated_scores: Scores after removing evidence.

    Returns:
        Mean sufficiency above chance (higher = evidence more sufficient).
    """
    original = np.asarray(original_scores)
    ablated = np.asarray(ablated_scores)
    # Penalize near-chance predictions; sufficiency is meaningful only above 0.5
    return float(np.mean(np.maximum(np.minimum(original, ablated) - 0.5, 0)) * 2)


def comprehensiveness_score(
    original_scores: list[float],
    ablated_scores: list[float],
) -> float:
    """Compute comprehensiveness: does removing teacher signal change predictions?

    Comprehensiveness measures how much the prediction changes when
    the teacher signal is removed. A higher difference indicates
    the teacher signal is more comprehensiveness (needed for prediction).

    Score = mean(|original - ablated|)

    Args:
        original_scores: Scores with teacher signal.
        ablated_scores: Scores without teacher signal.

    Returns:
        Mean absolute difference (higher = teacher signal more comprehensive).
    """
    original = np.asarray(original_scores)
    ablated = np.asarray(ablated_scores)
    return float(np.mean(np.abs(original - ablated)))


def faithfulness_impact(
    original_scores: list[float],
    ablated_scores: list[float],
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute faithfulness impact metrics for evidence ablation.

    Args:
        original_scores: Scores with full evidence/teacher signal.
        ablated_scores: Scores after ablation.
        threshold: Classification threshold for prediction changes.

    Returns:
        Dict with sufficiency, comprehensiveness, and prediction_flip_rate.
    """
    original = np.asarray(original_scores)
    ablated = np.asarray(ablated_scores)

    original_preds = (original >= threshold).astype(int)
    ablated_preds = (ablated >= threshold).astype(int)

    flip_rate = float(np.mean(original_preds != ablated_preds))

    return {
        "sufficiency": sufficiency_score(original_scores, ablated_scores),
        "comprehensiveness": comprehensiveness_score(original_scores, ablated_scores),
        "prediction_flip_rate": flip_rate,
    }
