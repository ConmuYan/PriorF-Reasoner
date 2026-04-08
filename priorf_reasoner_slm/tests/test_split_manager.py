"""Tests for graph_data.split_manager."""

from __future__ import annotations

import torch

from priorf_reasoner_slm.graph_data.split_manager import split_data


def test_split_data_tiny_dataset_still_produces_three_way_split():
    """The tiny-dataset fallback should keep train/val/test all non-empty."""
    y = torch.tensor([0, 1, 0], dtype=torch.long)

    result = split_data(num_nodes=3, y=y, seed=42)

    assert result["train_mask"].sum().item() == 1
    assert result["val_mask"].sum().item() == 1
    assert result["test_mask"].sum().item() == 1
    assert (result["train_mask"] & result["val_mask"]).sum().item() == 0
    assert (result["train_mask"] & result["test_mask"]).sum().item() == 0
    assert (result["val_mask"] & result["test_mask"]).sum().item() == 0
