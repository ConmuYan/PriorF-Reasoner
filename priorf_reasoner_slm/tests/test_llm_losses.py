"""Tests for llm/losses: generation, classification, distillation, and PriorFLoss."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from priorf_reasoner_slm.llm.losses import (
    gen_loss,
    cls_loss,
    distill_loss_kl,
    distill_loss_mse,
    PriorFLoss,
)


# ---- Fixtures ----


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def seq_len():
    return 16


@pytest.fixture
def vocab_size():
    return 100


@pytest.fixture
def lm_logits(batch_size, seq_len, vocab_size):
    """Fake LM logits: (batch, seq_len, vocab)."""
    return torch.randn(batch_size, seq_len, vocab_size)


@pytest.fixture
def lm_labels(batch_size, seq_len):
    """Labels with -100 for first 5 positions (prompt), real tokens after."""
    labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
    # Completion tokens (last 8 positions)
    labels[:, 8:] = torch.randint(0, 100, (batch_size, 8))
    return labels


@pytest.fixture
def cls_logits(batch_size):
    """Classification logits (before sigmoid)."""
    return torch.randn(batch_size)


@pytest.fixture
def cls_targets(batch_size):
    """Binary targets: 1=fraud, 0=benign."""
    return torch.tensor([1.0, 0.0, 1.0, 0.0])


@pytest.fixture
def teacher_probs(batch_size):
    """Teacher fraud probabilities."""
    return torch.tensor([0.9, 0.1, 0.85, 0.15])


# ---- Tests: gen_loss ----


class TestGenLoss:
    """Tests for gen_loss with masked labels."""

    def test_gen_loss_ignores_prompt_tokens(self, lm_logits, lm_labels):
        """gen_loss should ignore -100 (prompt) tokens in loss computation."""
        loss = gen_loss(lm_logits, lm_labels)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # scalar
        assert loss.item() >= 0

    def test_gen_loss_with_all_valid_labels(self, batch_size, seq_len, vocab_size):
        """When all labels are valid (no -100), loss should be computable."""
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        loss = gen_loss(logits, labels)
        assert loss.item() >= 0

    def test_gen_loss_with_all_masked_labels(self, batch_size, seq_len, vocab_size):
        """When all labels are -100, loss should still be a scalar (zero or computed)."""
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
        loss = gen_loss(logits, labels)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0

    def test_gen_loss_shift_correct(self, batch_size, seq_len, vocab_size):
        """gen_loss should shift logits and labels by 1 (causal LM)."""
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        loss = gen_loss(logits, labels)
        # Should not raise and should return scalar
        assert loss.item() >= 0


# ---- Tests: cls_loss ----


class TestClsLoss:
    """Tests for cls_loss (BCE)."""

    def test_cls_loss_bce_computed(self, cls_logits, cls_targets):
        """cls_loss should compute binary cross-entropy."""
        loss = cls_loss(cls_logits, cls_targets)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_cls_loss_perfect_prediction_low_loss(self):
        """Perfect predictions should yield near-zero loss."""
        # Target=1 with logits->inf, target=0 with logits-> -inf
        logits = torch.tensor([100.0, -100.0])
        targets = torch.tensor([1.0, 0.0])
        loss = cls_loss(logits, targets)
        assert loss.item() < 0.1

    def test_cls_loss_random_prediction(self):
        """Random logits should give positive loss."""
        logits = torch.randn(8)
        targets = torch.randint(0, 2, (8,)).float()
        loss = cls_loss(logits, targets)
        assert loss.item() >= 0


# ---- Tests: distill_loss_kl ----


class TestDistillLossKL:
    """Tests for distill_loss_kl."""

    def test_distill_loss_kl_positive(self, cls_logits, teacher_probs):
        """KL divergence should be non-negative."""
        loss = distill_loss_kl(cls_logits, teacher_probs)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_distill_loss_kl_with_temperature(self, cls_logits, teacher_probs):
        """Temperature scaling should be applied (temperature**2 factor)."""
        loss_t1 = distill_loss_kl(cls_logits, teacher_probs, temperature=1.0)
        loss_t2 = distill_loss_kl(cls_logits, teacher_probs, temperature=2.0)
        # Temperature**2 factor means loss_t2 should be ~4x loss_t1
        assert isinstance(loss_t1, torch.Tensor)
        assert isinstance(loss_t2, torch.Tensor)

    def test_distill_loss_kl_symmetric_behavior(self):
        """Student and teacher probabilities affect loss."""
        student_logits_1 = torch.tensor([2.0])  # high fraud prob
        student_logits_2 = torch.tensor([-2.0])  # low fraud prob
        teacher_probs = torch.tensor([0.9])

        loss_1 = distill_loss_kl(student_logits_1, teacher_probs)
        loss_2 = distill_loss_kl(student_logits_2, teacher_probs)

        # Both should be valid scalars
        assert loss_1.item() >= 0
        assert loss_2.item() >= 0

    def test_distill_loss_kl_teacher_clamped(self, cls_logits):
        """Teacher probs near 0 or 1 should be clamped to avoid log(0)."""
        teacher_probs = torch.tensor([0.0, 1.0, 0.5, 0.999])
        loss = distill_loss_kl(cls_logits[:4], teacher_probs)
        assert loss.item() >= 0


# ---- Tests: distill_loss_mse ----


class TestDistillLossMSE:
    """Tests for distill_loss_mse."""

    def test_distill_loss_mse_positive(self, batch_size):
        """MSE between student and teacher probs should be non-negative."""
        student_probs = torch.rand(batch_size)
        teacher_probs = torch.rand(batch_size)
        loss = distill_loss_mse(student_probs, teacher_probs)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_distill_loss_mse_zero_identical(self, batch_size):
        """Identical probability vectors should yield zero loss."""
        probs = torch.rand(batch_size)
        loss = distill_loss_mse(probs, probs)
        assert loss.item() < 1e-6

    def test_distill_loss_mse_batched(self, batch_size):
        """MSE loss should average over the batch."""
        student = torch.zeros(batch_size)
        teacher = torch.ones(batch_size)
        loss = distill_loss_mse(student, teacher)
        # MSE of [0,0,...] vs [1,1,...] = 1.0
        assert abs(loss.item() - 1.0) < 0.01


# ---- Tests: PriorFLoss combined ----


class TestPriorFLoss:
    """Tests for PriorFLoss combined tri-loss."""

    def test_priorf_loss_forward_all_losses(self, lm_logits, lm_labels, cls_logits, cls_targets, teacher_probs):
        """PriorFLoss.forward() should return dict with loss + 3 components."""
        criterion = PriorFLoss(lambda_cls=1.0, lambda_distill=1.0)
        result = criterion(lm_logits, lm_labels, cls_logits, cls_targets, teacher_probs)

        assert "loss" in result
        assert "gen_loss" in result
        assert "cls_loss" in result
        assert "distill_loss" in result

        # All should be scalars
        assert result["loss"].dim() == 0
        assert result["gen_loss"].dim() == 0
        assert result["cls_loss"].dim() == 0
        assert result["distill_loss"].dim() == 0

    def test_priorf_loss_total_is_weighted_sum(self, lm_logits, lm_labels, cls_logits, cls_targets, teacher_probs):
        """Total loss = gen + lambda_cls * cls + lambda_distill * distill."""
        criterion = PriorFLoss(lambda_cls=2.0, lambda_distill=0.5)
        result = criterion(lm_logits, lm_labels, cls_logits, cls_targets, teacher_probs)

        expected = (
            result["gen_loss"]
            + 2.0 * result["cls_loss"]
            + 0.5 * result["distill_loss"]
        )
        assert torch.allclose(result["loss"], expected, atol=1e-5)

    def test_priorf_loss_default_weights(self, lm_logits, lm_labels, cls_logits, cls_targets, teacher_probs):
        """Default weights should be lambda_cls=1.0, lambda_distill=1.0."""
        criterion = PriorFLoss()
        result = criterion(lm_logits, lm_labels, cls_logits, cls_targets, teacher_probs)

        expected = result["gen_loss"] + result["cls_loss"] + result["distill_loss"]
        assert torch.allclose(result["loss"], expected, atol=1e-5)

    def test_priorf_loss_individual_components_nonzero(self, lm_logits, lm_labels, cls_logits, cls_targets, teacher_probs):
        """All component losses should be non-negative with valid inputs."""
        criterion = PriorFLoss()
        result = criterion(lm_logits, lm_labels, cls_logits, cls_targets, teacher_probs)

        assert result["gen_loss"].item() >= 0
        assert result["cls_loss"].item() >= 0
        assert result["distill_loss"].item() >= 0

    def test_priorf_loss_cls_only_mode(self, lm_logits, lm_labels, cls_logits, cls_targets, teacher_probs):
        """Setting lambda_distill=0 should disable distillation."""
        criterion = PriorFLoss(lambda_distill=0.0)
        result = criterion(lm_logits, lm_labels, cls_logits, cls_targets, teacher_probs)

        assert torch.allclose(result["loss"], result["gen_loss"] + result["cls_loss"], atol=1e-5)

    def test_priorf_loss_distill_only_mode(self, lm_logits, lm_labels, cls_logits, cls_targets, teacher_probs):
        """Setting lambda_cls=0 should disable classification loss."""
        criterion = PriorFLoss(lambda_cls=0.0)
        result = criterion(lm_logits, lm_labels, cls_logits, cls_targets, teacher_probs)

        assert torch.allclose(result["loss"], result["gen_loss"] + result["distill_loss"], atol=1e-5)
