"""Data validation utilities for PriorF-Reasoner.

Checks: NaN, shape consistency, sparse graph symmetry, label proportions,
key completeness in .mat files.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Expected dimensions for standard benchmarks
EXPECTED_DIMS = {
    "amazon": {"feature_dim": 25, "num_relations": 3},
    "yelpchi": {"feature_dim": 32, "num_relations": 3},
}


class ValidationError(Exception):
    """Raised when data validation fails."""


def validate_data(
    graph_data: dict,
    dataset_name: str,
    strict: bool = True,
) -> list[str]:
    """Validate loaded graph data against expected schema.

    Args:
        graph_data: Output from mat_loader.load_mat_dataset.
        dataset_name: "amazon" or "yelpchi".
        strict: If True, raise on first error. If False, collect all warnings.

    Returns:
        List of warning/error messages (empty if all checks pass).
    """
    errors: list[str] = []

    def _check(condition: bool, msg: str) -> None:
        if not condition:
            errors.append(msg)
            if strict:
                raise ValidationError(msg)

    x = graph_data["x"]
    y = graph_data["y"]
    hsd = graph_data["hsd"]
    relations = graph_data["relations"]

    # 1. NaN check
    _check(not torch.isnan(x).any(), f"Features contain NaN values")
    _check(not torch.isnan(y).any(), f"Labels contain NaN values")
    _check(not torch.isnan(hsd).any(), f"HSD contains NaN values")

    # 2. Shape consistency
    n = x.size(0)
    _check(y.shape == (n,), f"Label shape mismatch: {y.shape} vs expected ({n},)")
    _check(hsd.shape == (n,), f"HSD shape mismatch: {hsd.shape} vs expected ({n},)")

    # 3. Feature dimensions
    expected = EXPECTED_DIMS.get(dataset_name, {})
    if "feature_dim" in expected:
        _check(
            x.size(1) == expected["feature_dim"],
            f"Feature dim mismatch: {x.size(1)} vs expected {expected['feature_dim']}",
        )

    # 4. Number of relations
    if "num_relations" in expected:
        _check(
            len(relations) == expected["num_relations"],
            f"Relation count mismatch: {len(relations)} vs expected {expected['num_relations']}",
        )

    # 5. Label proportions (must have both classes)
    unique_labels = torch.unique(y)
    _check(
        len(unique_labels) >= 2,
        f"Labels must have at least 2 classes, got {unique_labels.tolist()}",
    )

    fraud_ratio = y.float().mean().item()
    _check(
        0.01 < fraud_ratio < 0.99,
        f"Suspicious label ratio: {fraud_ratio:.4f} (expected between 0.01 and 0.99)",
    )

    # 6. Edge index sanity
    for rel_name, rel_data in relations.items():
        ei = rel_data["edge_index"]
        _check(
            ei.dim() == 2 and ei.size(0) == 2,
            f"Relation {rel_name}: edge_index must be (2, E), got {ei.shape}",
        )
        if ei.size(1) > 0:
            _check(
                ei.max() < n,
                f"Relation {rel_name}: edge_index contains node id >= {n}",
            )

    # 7. Graph symmetry check (undirected)
    homo = graph_data["homo"]
    if homo.size(1) > 0:
        row, col = homo[0], homo[1]
        edge_set = set(zip(row.tolist(), col.tolist()))
        sample_size = min(100, len(edge_set))
        sample_edges = list(edge_set)[:sample_size]
        symmetric_count = sum(1 for (u, v) in sample_edges if (v, u) in edge_set)
        sym_ratio = symmetric_count / max(sample_size, 1)
        if sym_ratio < 0.8:
            msg = (
                f"Graph may not be symmetric: {sym_ratio:.2%} of sampled edges "
                f"have reverse counterpart"
            )
            _check(False, msg)

    if errors:
        logger.warning("Validation completed with %d warnings", len(errors))
    else:
        logger.info("All data validation checks passed for %s", dataset_name)

    return errors
