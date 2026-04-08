"""Tri-loss computation for PriorF-Reasoner student training.

Implements the three loss components:

    L = L_gen + lambda1 * L_cls + lambda2 * L_distill

- L_gen:     Causal LM cross-entropy on completion tokens
- L_cls:     Binary cross-entropy from the classification head
- L_distill: KL divergence between student and teacher probabilities
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def gen_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute causal LM generation loss on completion tokens.

    The ``labels`` tensor uses ``-100`` for prompt tokens (ignored)
    and actual token IDs for completion tokens. This is the standard
    SFT / causal LM loss.

    Args:
        logits: Model output logits ``(batch, seq_len, vocab_size)``.
        labels: Target token IDs with ``-100`` for ignored positions
            ``(batch, seq_len)``.

    Returns:
        Scalar loss tensor.
    """
    # Shift: logits[:-1] predict labels[1:]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    return loss_fn(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )


def cls_loss(
    cls_logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Compute binary classification loss (BCE with logits).

    Args:
        cls_logits: Classification head logits ``(batch,)``.
        targets: Binary targets ``(batch,)`` where 1 = fraud, 0 = benign.

    Returns:
        Scalar loss tensor.
    """
    loss_fn = nn.BCEWithLogitsLoss()
    return loss_fn(cls_logits, targets.float())


def distill_loss_kl(
    student_logits: torch.Tensor,
    teacher_probs: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Compute KL divergence distillation loss.

    Measures the KL divergence between the student's probability
    distribution and the teacher's probability distribution.

    The student logits are converted to probabilities via softmax
    with temperature scaling.

    Args:
        student_logits: Student classification logits ``(batch,)``.
        teacher_probs: Teacher probabilities ``(batch,)`` in [0, 1].
        temperature: Softmax temperature for smoothing.

    Returns:
        Scalar KL divergence loss tensor.
    """
    # Build 2-class distributions: [p_benign, p_fraud]
    student_probs = torch.sigmoid(student_logits / temperature)
    student_probs = torch.stack([1.0 - student_probs, student_probs], dim=-1)  # (batch, 2)

    teacher_probs_clamped = teacher_probs.clamp(min=1e-8, max=1.0 - 1e-8)
    teacher_dist = torch.stack(
        [1.0 - teacher_probs_clamped, teacher_probs_clamped], dim=-1
    )  # (batch, 2)

    # KL(teacher || student)
    log_student = torch.log(student_probs.clamp(min=1e-8))
    kl = F.kl_div(
        log_student,
        teacher_dist,
        reduction="batchmean",
    )
    return kl * (temperature ** 2)


def distill_loss_mse(
    student_probs: torch.Tensor,
    teacher_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute MSE distillation loss between probability vectors.

    Args:
        student_probs: Student probabilities ``(batch,)``.
        teacher_probs: Teacher probabilities ``(batch,)``.

    Returns:
        Scalar MSE loss tensor.
    """
    return F.mse_loss(student_probs, teacher_probs)


class PriorFLoss(nn.Module):
    """Combined tri-loss for PriorF-Reasoner student training.

    Computes::

        L = L_gen + lambda1 * L_cls + lambda2 * L_distill

    Args:
        lambda_cls: Weight for the classification loss.
        lambda_distill: Weight for the distillation loss.
        distill_temperature: Temperature for KL distillation.
    """

    def __init__(
        self,
        lambda_cls: float = 1.0,
        lambda_distill: float = 1.0,
        distill_temperature: float = 2.0,
    ) -> None:
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_distill = lambda_distill
        self.distill_temperature = distill_temperature

    def forward(
        self,
        lm_logits: torch.Tensor,
        labels: torch.Tensor,
        cls_logits: torch.Tensor,
        cls_targets: torch.Tensor,
        teacher_probs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute all three losses and return the combined loss.

        Args:
            lm_logits: Language model output logits ``(batch, seq_len, vocab)``.
            labels: Token labels with -100 for prompt ``(batch, seq_len)``.
            cls_logits: Classification head logits ``(batch,)``.
            cls_targets: Binary labels ``(batch,)``.
            teacher_probs: Teacher probabilities ``(batch,)``.

        Returns:
            Dict with keys ``loss``, ``gen_loss``, ``cls_loss``,
            ``distill_loss``.
        """
        l_gen = gen_loss(lm_logits, labels)
        l_cls = cls_loss(cls_logits, cls_targets)
        l_distill = distill_loss_kl(
            cls_logits, teacher_probs, self.distill_temperature
        )

        total = l_gen + self.lambda_cls * l_cls + self.lambda_distill * l_distill

        return {
            "loss": total,
            "gen_loss": l_gen,
            "cls_loss": l_cls,
            "distill_loss": l_distill,
        }
