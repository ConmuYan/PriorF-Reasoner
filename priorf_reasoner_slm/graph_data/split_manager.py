"""Dataset splitting with stratified sampling for reproducibility.

Standard split: 70% train / 10% val / 20% test, seed=717.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit


def split_data(
    num_nodes: int,
    y: torch.Tensor,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 717,
) -> dict[str, torch.Tensor]:
    """Split nodes into train/val/test with stratified sampling.

    Args:
        num_nodes: Total number of nodes.
        y: Node labels (N,).
        train_ratio: Training set ratio (default 0.7).
        val_ratio: Validation set ratio (default 0.1).
        test_ratio: Test set ratio (default 0.2).
        seed: Random seed (default 717).

    Returns:
        Dict with keys: train_mask, val_mask, test_mask (bool tensors).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1"
    assert num_nodes >= 3, "Need at least 3 nodes for a 3-way split"
    assert val_ratio > 0 and test_ratio > 0, "val and test ratios must be positive"

    y_np = y.cpu().numpy()
    indices = np.arange(num_nodes)

    # First split: Train vs (Val + Test)
    try:
        sss1 = StratifiedShuffleSplit(
            n_splits=1, test_size=(1 - train_ratio), random_state=seed
        )
        train_idx, remainder_idx = next(sss1.split(indices, y_np))
    except ValueError:
        # Fallback for tiny datasets
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(indices)
        # Reserve at least one node for both val and test.
        train_size = min(
            max(1, int(num_nodes * train_ratio)),
            num_nodes - 2,
        )
        train_idx = shuffled[:train_size]
        remainder_idx = shuffled[train_size:]

    # Second split: Val vs Test
    y_remainder = y_np[remainder_idx]
    test_relative = test_ratio / (val_ratio + test_ratio)

    try:
        sss2 = StratifiedShuffleSplit(
            n_splits=1, test_size=test_relative, random_state=seed
        )
        val_sub_idx, test_sub_idx = next(sss2.split(remainder_idx, y_remainder))
    except ValueError:
        rng = np.random.default_rng(seed)
        shuffled_rem = rng.permutation(len(remainder_idx))
        # Guarantee at least 1 val and 1 test node
        remainder_len = len(remainder_idx)
        if remainder_len < 2:
            raise ValueError(
                f"Cannot split {remainder_len} remainder nodes into val+test"
            )
        val_size = max(1, min(int(remainder_len * (1 - test_relative)), remainder_len - 1))
        val_sub_idx = shuffled_rem[:val_size]
        test_sub_idx = shuffled_rem[val_size:]

    val_idx = remainder_idx[val_sub_idx]
    test_idx = remainder_idx[test_sub_idx]

    # Create boolean masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return {
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "train_idx": torch.from_numpy(train_idx),
        "val_idx": torch.from_numpy(val_idx),
        "test_idx": torch.from_numpy(test_idx),
    }
