"""Load CARE-GNN benchmark .mat datasets (YelpChi / Amazon).

Standardizes output to a uniform dict with keys:
    x, y, relations, homo, node_ids, hsd

Relations:
    YelpChi: RUR (same user), RTR (same month), RSR (same star rating)
    Amazon:  UPU (shared product), USU (same star user), UVU (top-5% text sim)
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch
from torch_scatter import scatter_mean


class RelationData(TypedDict):
    """Edge index and metadata for a single relation."""

    edge_index: torch.Tensor  # (2, E)
    name: str  # e.g. "RUR"
    num_edges: int


class GraphData(TypedDict):
    """Standardized graph data output."""

    x: torch.Tensor  # (N, D) node features WITHOUT HSD
    y: torch.Tensor  # (N,) binary labels
    relations: dict[str, RelationData]  # relation_name -> {edge_index, name, num_edges}
    homo: torch.Tensor  # (2, E_total) union of all edges
    node_ids: torch.Tensor  # (N,) integer node indices
    hsd: torch.Tensor  # (N,) HSD scores
    dataset_name: str  # "amazon" or "yelpchi"
    feature_dim: int  # original feature dim (25 for Amazon, 32 for YelpChi)


# Canonical 3-relation mapping
YELP_MAT_KEYS = ("net_rur", "net_rtr", "net_rsr")
YELP_REL_NAMES = ("RUR", "RTR", "RSR")

AMAZON_MAT_KEYS = ("net_upu", "net_usu", "net_uvu")
AMAZON_REL_NAMES = ("UPU", "USU", "UVU")


def _detect_dataset(mat: dict) -> str:
    """Detect dataset type from .mat keys."""
    if "net_rur" in mat:
        return "yelpchi"
    if "net_upu" in mat:
        return "amazon"
    raise ValueError(
        f"Cannot detect dataset type from .mat keys: "
        f"{[k for k in mat if not k.startswith('__')]}"
    )


def _sparse_to_edge_index(adj: sp.spmatrix) -> torch.Tensor:
    """Convert scipy sparse matrix to PyG edge_index (2, E), no self-loops."""
    coo = adj.tocoo()
    row = torch.from_numpy(coo.row.astype(np.int64))
    col = torch.from_numpy(coo.col.astype(np.int64))
    edge_index = torch.stack([row, col], dim=0)
    mask = edge_index[0] != edge_index[1]
    return edge_index[:, mask]


def compute_hsd(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Compute Heterophily-Induced Structural Discrepancy (HSD).

    For each node, HSD = mean L2 distance to its neighbors.
    """
    if edge_index.size(1) == 0:
        return torch.zeros(x.size(0))
    row, col = edge_index
    dist = torch.norm(x[row] - x[col], p=2, dim=1)
    hsd = scatter_mean(dist, row, dim=0, dim_size=x.size(0))
    hsd = torch.nan_to_num(hsd, nan=0.0)
    return hsd


def load_mat_dataset(mat_path: str | Path, hsd_invert: bool = False) -> GraphData:
    """Load a .mat benchmark dataset and return standardized GraphData.

    Args:
        mat_path: Path to YelpChi.mat or Amazon.mat.
        hsd_invert: If True, compute max - HSD (for homophilic fraud).

    Returns:
        GraphData dict with x, y, relations, homo, node_ids, hsd.
    """
    mat_path = Path(mat_path)
    if not mat_path.exists():
        raise FileNotFoundError(f".mat file not found: {mat_path}")

    mat = sio.loadmat(str(mat_path))
    ds_name = _detect_dataset(mat)

    if ds_name == "yelpchi":
        mat_keys, rel_names = YELP_MAT_KEYS, YELP_REL_NAMES
    else:
        mat_keys, rel_names = AMAZON_MAT_KEYS, AMAZON_REL_NAMES

    # --- Features ---
    features = mat["features"]
    if sp.issparse(features):
        features = features.toarray()
    features = features.astype(np.float32)
    x = torch.from_numpy(features)

    # --- Labels ---
    labels = mat["label"].flatten().astype(np.int64)
    y = torch.from_numpy(labels)

    # --- Relations ---
    relations: dict[str, RelationData] = {}
    all_edges: list[torch.Tensor] = []

    for mat_key, rel_name in zip(mat_keys, rel_names):
        adj = mat[mat_key]
        ei = _sparse_to_edge_index(adj)
        relations[rel_name] = {
            "edge_index": ei,
            "name": rel_name,
            "num_edges": ei.size(1),
        }
        all_edges.append(ei)

    # --- Homo (deduplicated union of all edges) ---
    homo = torch.cat(all_edges, dim=1)
    # Remove duplicate edges that appear across multiple relations
    if homo.size(1) > 0:
        n_nodes = x.size(0)
        edge_hashes = homo[0].long() * n_nodes + homo[1].long()
        # Find first occurrence of each unique edge hash
        sorted_hashes, sort_idx = torch.sort(edge_hashes)
        # Detect where value changes -> first occurrence in sorted order
        change_mask = torch.ones(sorted_hashes.size(0), dtype=torch.bool)
        change_mask[1:] = sorted_hashes[1:] != sorted_hashes[:-1]
        # Map back to original indices
        homo = homo[:, sort_idx[change_mask]]

    # --- HSD on union graph ---
    hsd = compute_hsd(x, homo)
    if hsd_invert and hsd.max() > 0:
        hsd = hsd.max() - hsd

    # --- Node IDs ---
    n = x.size(0)
    node_ids = torch.arange(n, dtype=torch.long)

    return GraphData(
        x=x,
        y=y,
        relations=relations,
        homo=homo,
        node_ids=node_ids,
        hsd=hsd,
        dataset_name=ds_name,
        feature_dim=features.shape[1],
    )
