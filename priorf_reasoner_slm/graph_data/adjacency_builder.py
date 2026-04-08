"""Build relation-specific adjacency structures for evidence extraction.

For each node, compute neighbor statistics across relations:
- Per-relation degree
- Suspicious-neighbor ratio (fraction of neighbors labeled fraud)
- Per-relation mean feature discrepancy (L2)
- Top-k neighbor count (capped by degree)
- Mean neighbor HSD

Returns batch arrays instead of per-node Python dicts for scalability
on YelpChi (~45k) and Amazon (~12k) datasets.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_scatter import scatter_add, scatter_mean


@dataclass
class NeighborStats:
    """Batch neighbor statistics arrays (one value per node)."""

    degrees_per_relation: dict[str, torch.Tensor]  # (N,) long
    suspicious_neighbor_ratio: torch.Tensor  # (N,) float
    topk_neighbors: torch.Tensor  # (N,) long
    neighbor_hsd_mean: torch.Tensor  # (N,) float
    relation_discrepancy: dict[str, torch.Tensor]  # (N,) float per relation


def build_neighbor_stats(
    relations: dict[str, dict],
    y: torch.Tensor,
    hsd: torch.Tensor,
    x: torch.Tensor,
    num_nodes: int,
    top_k: int = 6,
) -> NeighborStats:
    """Compute per-node neighbor statistics for evidence cards.

    Uses vectorized scatter operations on torch tensors. Returns a
    NeighborStats dataclass with batch arrays — no per-node Python loop
    or dict materialization.

    Args:
        relations: Dict mapping relation name to {"edge_index": (2, E)}.
        y: Node labels (N,).
        hsd: HSD scores (N,).
        x: Node features (N, D).
        num_nodes: Total node count.
        top_k: Cap for topk_neighbors field.

    Returns:
        NeighborStats with vectorized arrays.
    """
    device = x.device

    # --- Union all edges for global neighbor stats ---
    all_row_list: list[torch.Tensor] = []
    all_col_list: list[torch.Tensor] = []
    for rel_data in relations.values():
        ei = rel_data["edge_index"]
        all_row_list.append(ei[0])
        all_col_list.append(ei[1])

    all_row = torch.cat(all_row_list).to(device)
    all_col = torch.cat(all_col_list).to(device)

    # Global degree
    degree = torch.zeros(num_nodes, dtype=torch.long, device=device)
    scatter_add(torch.ones(all_row.size(0), dtype=torch.long, device=device), all_col, out=degree)

    # Suspicious neighbor ratio per node
    src_labels = y.to(device)[all_row].float()
    fraud_neighbors = torch.zeros(num_nodes, dtype=torch.float, device=device)
    scatter_add(src_labels, all_col, out=fraud_neighbors)
    total_neighbors = degree.float().clamp(min=1)
    susp_ratio = fraud_neighbors / total_neighbors

    # Neighbor HSD mean per node
    src_hsd = hsd.to(device)[all_row]
    neighbor_hsd_mean = scatter_mean(src_hsd, all_col, dim=0, dim_size=num_nodes)
    neighbor_hsd_mean = torch.nan_to_num(neighbor_hsd_mean, nan=0.0)

    # Top-k cap
    topk_neighbors = degree.clamp(max=top_k)

    # Per-relation degree and discrepancy
    rel_degrees: dict[str, torch.Tensor] = {}
    rel_discrepancy: dict[str, torch.Tensor] = {}
    for rel_name, rel_data in relations.items():
        ei = rel_data["edge_index"].to(device)
        row, col = ei

        # Degree per relation
        rel_deg = torch.zeros(num_nodes, dtype=torch.long, device=device)
        scatter_add(torch.ones(row.size(0), dtype=torch.long, device=device), col, out=rel_deg)
        rel_degrees[rel_name] = rel_deg

        # Mean feature discrepancy per relation per node
        diff = torch.norm(x.to(device)[row] - x.to(device)[col], p=2, dim=1)
        rel_disc = scatter_mean(diff, col, dim=0, dim_size=num_nodes)
        rel_discrepancy[rel_name] = torch.nan_to_num(rel_disc, nan=0.0)

    return NeighborStats(
        degrees_per_relation=rel_degrees,
        suspicious_neighbor_ratio=susp_ratio,
        topk_neighbors=topk_neighbors,
        neighbor_hsd_mean=neighbor_hsd_mean,
        relation_discrepancy=rel_discrepancy,
    )
