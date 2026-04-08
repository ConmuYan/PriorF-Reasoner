"""Tests for llm/collators: SFTDataCollator and CoTrainDataCollator."""

from __future__ import annotations

import pytest
import torch

from priorf_reasoner_slm.llm.collators import (
    SFTDataCollator,
    CoTrainDataCollator,
)


# ---- Fixtures ----


@pytest.fixture
def sft_collator():
    return SFTDataCollator(pad_token_id=0, ignore_index=-100)


@pytest.fixture
def cotrain_collator():
    return CoTrainDataCollator(pad_token_id=0, ignore_index=-100)


@pytest.fixture
def sft_features():
    """Two samples with different sequence lengths."""
    return [
        {
            "input_ids": [1, 2, 3, 4, 5],
            "labels": [-100, -100, 10, 11, 12],
        },
        {
            "input_ids": [1, 2, 3],
            "labels": [-100, 20, 21],
        },
    ]


@pytest.fixture
def cotrain_features():
    """Two samples for co-training with teacher_probs and cls_targets."""
    return [
        {
            "input_ids": [1, 2, 3, 4, 5],
            "labels": [-100, -100, 10, 11, 12],
            "teacher_probs": 0.87,
            "cls_targets": 1.0,
        },
        {
            "input_ids": [1, 2, 3],
            "labels": [-100, 20, 21],
            "teacher_probs": 0.12,
            "cls_targets": 0.0,
        },
    ]


# ---- Tests: SFTDataCollator ----


class TestSFTDataCollator:
    """Tests for SFTDataCollator padding behavior."""

    def test_pads_to_max_length(self, sft_collator, sft_features):
        """Batch should be padded to the longest sequence in the batch."""
        result = sft_collator(sft_features)

        # max_len = 5 (first sample)
        assert result["input_ids"].shape == (2, 5)
        assert result["labels"].shape == (2, 5)
        assert result["attention_mask"].shape == (2, 5)

    def test_shorter_sequences_padded(self, sft_collator, sft_features):
        """Shorter sequences should be padded with pad_token_id."""
        result = sft_collator(sft_features)

        # Second sample (len=3) padded to 5
        # Position 3,4 should be pad_token_id=0
        assert result["input_ids"][1, 3] == 0
        assert result["input_ids"][1, 4] == 0

    def test_labels_padded_with_ignore_index(self, sft_collator, sft_features):
        """Padded label positions should use ignore_index=-100."""
        result = sft_collator(sft_features)

        # Second sample labels at positions 3,4 should be -100
        assert result["labels"][1, 3] == -100
        assert result["labels"][1, 4] == -100

    def test_attention_mask_ones_for_valid_tokens(self, sft_collator, sft_features):
        """Attention mask should be 1 for real tokens, 0 for padding."""
        result = sft_collator(sft_features)

        # First sample: all 5 positions are real tokens
        assert torch.all(result["attention_mask"][0] == 1)

        # Second sample: positions 0,1,2 are real, 3,4 are padded
        assert torch.all(result["attention_mask"][1, :3] == 1)
        assert torch.all(result["attention_mask"][1, 3:] == 0)

    def test_original_data_preserved(self, sft_collator, sft_features):
        """Original token IDs should be preserved (not reordered)."""
        result = sft_collator(sft_features)

        # First sample data should match exactly
        assert result["input_ids"][0].tolist()[:5] == [1, 2, 3, 4, 5]
        assert result["labels"][0].tolist()[:5] == [-100, -100, 10, 11, 12]

    def test_single_sample(self, sft_collator):
        """Single sample should not change shape (seq_len x 1)."""
        features = [{"input_ids": [1, 2, 3], "labels": [-100, 20, 21]}]
        result = sft_collator(features)

        assert result["input_ids"].shape == (1, 3)
        assert result["labels"].shape == (1, 3)
        assert result["attention_mask"].shape == (1, 3)

    def test_all_same_length(self, sft_collator):
        """When all samples have same length, no padding needed."""
        features = [
            {"input_ids": [1, 2, 3], "labels": [-100, 10, 11]},
            {"input_ids": [4, 5, 6], "labels": [-100, 20, 21]},
        ]
        result = sft_collator(features)

        assert result["input_ids"].shape == (2, 3)
        # No padding positions should be 0 (pad_token_id)
        assert result["input_ids"][0, 2] != 0 or result["input_ids"][1, 2] != 0


# ---- Tests: CoTrainDataCollator ----


class TestCoTrainDataCollator:
    """Tests for CoTrainDataCollator handling teacher_probs."""

    def test_returns_input_ids_labels_attention_mask(self, cotrain_collator, cotrain_features):
        """Should return standard SFT fields."""
        result = cotrain_collator(cotrain_features)

        assert "input_ids" in result
        assert "labels" in result
        assert "attention_mask" in result

    def test_returns_teacher_probs(self, cotrain_collator, cotrain_features):
        """Should return teacher_probs tensor."""
        result = cotrain_collator(cotrain_features)

        assert "teacher_probs" in result
        assert isinstance(result["teacher_probs"], torch.Tensor)
        assert result["teacher_probs"].shape == (2,)
        assert result["teacher_probs"][0] == 0.87
        assert result["teacher_probs"][1] == 0.12

    def test_returns_cls_targets(self, cotrain_collator, cotrain_features):
        """Should return cls_targets tensor."""
        result = cotrain_collator(cotrain_features)

        assert "cls_targets" in result
        assert isinstance(result["cls_targets"], torch.Tensor)
        assert result["cls_targets"].shape == (2,)
        assert result["cls_targets"][0] == 1.0
        assert result["cls_targets"][1] == 0.0

    def test_teacher_probs_correct_types(self, cotrain_collator, cotrain_features):
        """teacher_probs should be float tensor."""
        result = cotrain_collator(cotrain_features)

        assert result["teacher_probs"].dtype == torch.float32
        # All values should be in [0, 1]
        assert (result["teacher_probs"] >= 0).all()
        assert (result["teacher_probs"] <= 1).all()

    def test_cls_targets_float(self, cotrain_collator, cotrain_features):
        """cls_targets should be float tensor (even if originally int/bool)."""
        result = cotrain_collator(cotrain_features)

        assert result["cls_targets"].dtype == torch.float

    def test_pads_correctly_for_different_lengths(self, cotrain_collator, cotrain_features):
        """Should pad input_ids and labels correctly (like SFTDataCollator)."""
        result = cotrain_collator(cotrain_features)

        # Max len = 5, batch size = 2
        assert result["input_ids"].shape == (2, 5)
        assert result["labels"].shape == (2, 5)
        assert result["attention_mask"].shape == (2, 5)

        # Second sample padded positions should be pad_token_id=0
        assert result["input_ids"][1, 3] == 0
        assert result["input_ids"][1, 4] == 0

        # Labels padded with ignore_index=-100
        assert result["labels"][1, 3] == -100
        assert result["labels"][1, 4] == -100

    def test_single_sample_cotrain(self, cotrain_collator):
        """Single co-training sample should work."""
        features = [
            {
                "input_ids": [1, 2, 3],
                "labels": [-100, 20, 21],
                "teacher_probs": 0.75,
                "cls_targets": 1.0,
            }
        ]
        result = cotrain_collator(features)

        assert result["input_ids"].shape == (1, 3)
        assert result["teacher_probs"].shape == (1,)
        assert result["cls_targets"].shape == (1,)
        assert result["teacher_probs"][0] == 0.75

    def test_float_and_int_cls_targets(self, cotrain_collator):
        """cls_targets should accept both int (0/1) and float."""
        features = [
            {"input_ids": [1, 2], "labels": [-100, 10], "teacher_probs": 0.5, "cls_targets": 1},
            {"input_ids": [3, 4], "labels": [-100, 20], "teacher_probs": 0.3, "cls_targets": 0},
        ]
        result = cotrain_collator(features)

        # Should convert to float
        assert result["cls_targets"].dtype == torch.float
        assert result["cls_targets"][0] == 1.0
        assert result["cls_targets"][1] == 0.0
