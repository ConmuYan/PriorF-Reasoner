"""Prediction fusion for PriorF-Reasoner inference.

Combines the student's classification-head probability with the
teacher's probability to produce the final fraud prediction::

    p_final = alpha * p_cls + (1 - alpha) * p_teacher
"""

from __future__ import annotations

import numpy as np
import torch


def fuse_predictions(
    cls_prob: torch.Tensor | float,
    teacher_prob: torch.Tensor | float,
    alpha: float = 0.5,
) -> torch.Tensor | float:
    """Fuse student cls-head and teacher probabilities.

    Computes::

        p_final = alpha * p_cls + (1 - alpha) * p_teacher

    Args:
        cls_prob: Student classification-head probability (scalar or tensor).
        teacher_prob: Teacher model probability (scalar or tensor).
        alpha: Fusion weight in [0, 1]. Higher alpha gives more weight
            to the student's classification head.

    Returns:
        Fused probability (same type as inputs).
    """
    return alpha * cls_prob + (1.0 - alpha) * teacher_prob


def optimize_alpha(
    cls_probs: torch.Tensor | np.ndarray | list[float],
    teacher_probs: torch.Tensor | np.ndarray | list[float],
    val_labels: torch.Tensor | np.ndarray | list[int],
    n_steps: int = 100,
) -> tuple[float, float]:
    """Find the optimal fusion weight alpha on a validation set.

    Performs a grid search over alpha values in [0, 1] and returns
    the alpha that maximizes classification accuracy.

    Args:
        cls_probs: Student cls-head probabilities for validation samples.
        teacher_probs: Teacher probabilities for validation samples.
        val_labels: Ground-truth binary labels (0 = benign, 1 = fraud).
        n_steps: Number of alpha values to evaluate (grid resolution).

    Returns:
        Tuple of (optimal_alpha, best_accuracy).
    """
    cls_probs = torch.as_tensor(cls_probs, dtype=torch.float)
    teacher_probs = torch.as_tensor(teacher_probs, dtype=torch.float)
    val_labels = torch.as_tensor(val_labels, dtype=torch.float)

    best_alpha = 0.5
    best_acc = 0.0

    for step in range(n_steps + 1):
        alpha = step / n_steps
        fused = fuse_predictions(cls_probs, teacher_probs, alpha=alpha)
        preds = (fused >= 0.5).float()
        acc = (preds == val_labels).float().mean().item()

        if acc > best_acc:
            best_acc = acc
            best_alpha = alpha

    return best_alpha, best_acc
