"""Export per-node teacher evidence from PriorF-GNN.

For each node, produces flat + structured evidence aligned with
the Evidence Card schema defined in read.md (M2).

Output columns are designed to feed directly into the evidence/serializer
which builds the nested JSON card.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from priorf_reasoner_slm.graph_data.adjacency_builder import NeighborStats, build_neighbor_stats
from priorf_reasoner_slm.priorf_teacher.export_hooks import (
    register_export_hooks,
    remove_hooks,
    extract_branch_logits,
)

logger = logging.getLogger(__name__)


@dataclass
class TeacherEvidence:
    """Per-node teacher evidence record."""

    node_id: int
    split: str  # "train", "val", "test"
    label: int
    teacher_prob: float
    teacher_logit: float
    hsd: float
    hsd_quantile: str  # e.g. "top_1_percent"
    asda_switch: float
    mlp_logit: float | None
    gnn_logit: float | None
    branch_gap: float
    high_hsd_flag: bool


# ---- Pre-computed quantile thresholds for scalability ----

def _precompute_quantile_thresholds(hsd: torch.Tensor) -> dict[str, float]:
    """Compute HSD quantile thresholds once for all nodes.

    Returns:
        Dict mapping label -> threshold value.
    """
    hsd_f = hsd.float()
    return {
        "top_1_percent": torch.quantile(hsd_f, 0.99).item(),
        "top_5_percent": torch.quantile(hsd_f, 0.95).item(),
        "top_10_percent": torch.quantile(hsd_f, 0.90).item(),
    }


def compute_hsd_quantile(
    hsd_val: float,
    thresholds: dict[str, float],
) -> str:
    """Map a single HSD value to a quantile bucket using pre-computed thresholds.

    Args:
        hsd_val: HSD value for one node.
        thresholds: Output of _precompute_quantile_thresholds.

    Returns:
        Quantile label string.
    """
    for label in ("top_1_percent", "top_5_percent", "top_10_percent"):
        if hsd_val >= thresholds[label]:
            return label
    return "normal"


def _compute_discrepancy_level(
    disc_val: float,
    low_thresh: float = 0.33,
    high_thresh: float = 0.66,
) -> str:
    """Map a discrepancy value to low/medium/high using relative rank.

    Uses quantile-based thresholds to be dataset-adaptive.
    """
    if disc_val >= high_thresh:
        return "high"
    elif disc_val >= low_thresh:
        return "medium"
    return "low"


def _determine_split(
    i: int,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
) -> str:
    """Determine which split a node belongs to."""
    if train_mask[i]:
        return "train"
    elif val_mask[i]:
        return "val"
    elif test_mask[i]:
        return "test"
    return "unused"


def _build_top_relations(
    relation_discrepancies: dict[str, float],
    top_n: int = 2,
) -> list[str]:
    """Return the highest-discrepancy relations for a node."""
    ranked = sorted(
        relation_discrepancies.items(),
        key=lambda item: (-item[1], item[0]),
    )
    return [rel for rel, _ in ranked[:top_n]]


def _build_neighbor_summary(
    degrees_per_relation: dict[str, int],
    relation_discrepancies: dict[str, float],
    suspicious_neighbor_ratio: float,
    top_relations: list[str],
) -> dict[str, object]:
    """Build the v1 neighbor summary required by read.md M1."""
    ranked = sorted(
        relation_discrepancies.items(),
        key=lambda item: (-item[1], item[0]),
    )
    discrepancy_rank = {
        rel: rank
        for rank, (rel, _) in enumerate(ranked, start=1)
    }
    return {
        "degrees_per_relation": degrees_per_relation,
        "top_relations_mean_discrepancy": {
            rel: relation_discrepancies[rel] for rel in top_relations
        },
        "suspicious_neighbor_ratio": suspicious_neighbor_ratio,
        "same_relation_discrepancy_rank": discrepancy_rank,
    }


def export_teacher_evidence(
    model: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    hsd: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    relations: dict[str, dict],
    dataset_name: str = "",
    device: str | torch.device = "cpu",
) -> pd.DataFrame:
    """Run teacher forward pass and export per-node evidence.

    Args:
        model: Trained PriorF-GNN model.
        x: Node features (N, D).
        edge_index: Edge indices (2, E).
        edge_type: Edge types (E,).
        hsd: HSD scores (N,).
        y: Labels (N,).
        train_mask, val_mask, test_mask: Boolean split masks.
        relations: Dict mapping relation name to {"edge_index": (2, E)}.
        dataset_name: Name of the dataset (for Evidence Card).
        device: Compute device.

    Returns:
        DataFrame with one row per node.
    """
    model.eval()
    device = torch.device(device)

    # Pre-compute HSD quantile thresholds (O(1) instead of O(N) per node)
    hsd_thresholds = _precompute_quantile_thresholds(hsd)
    high_hsd_thresh = hsd_thresholds["top_5_percent"]

    # Pre-compute per-relation discrepancy quantiles for level mapping
    neighbor_stats = build_neighbor_stats(
        relations=relations,
        y=y,
        hsd=hsd,
        x=x,
        num_nodes=x.size(0),
    )

    # Compute discrepancy level thresholds per relation
    disc_thresholds: dict[str, tuple[float, float]] = {}
    for rel_name in relations:
        disc_vals = neighbor_stats.relation_discrepancy[rel_name].cpu().float().numpy()
        disc_thresholds[rel_name] = (
            float(np.percentile(disc_vals, 33)),
            float(np.percentile(disc_vals, 66)),
        )

    # Register hooks with exception-safe cleanup
    handles, storage = register_export_hooks(model)
    relation_names = list(relations.keys())

    try:
        # Move data to device
        x_d = x.to(device)
        ei_d = edge_index.to(device)
        et_d = edge_type.to(device)
        hsd_d = hsd.to(device)

        # Forward pass
        with torch.no_grad():
            use_asda = hasattr(model, "asda") and model.asda is not None
            if use_asda:
                logits = model(x_d, ei_d, et_d, hsd_d)
            else:
                logits = model(x_d, ei_d, et_d)

        # Get probabilities
        probs = torch.sigmoid(logits).cpu()

        # Extract branch logits from hook storage
        mlp_emb = storage.get("mlp_embedding")
        rgcn2_out = storage.get("rgcn2_output")
        # Apply BN + ReLU to get gnn_embedding
        if rgcn2_out is not None:
            if hasattr(model, "bn2"):
                rgcn2_out = model.bn2(rgcn2_out)
            rgcn2_out = torch.relu(rgcn2_out)

        # Assert fusion layout assumption
        out_layer = getattr(model, "out", None)
        if out_layer is not None and mlp_emb is not None and rgcn2_out is not None:
            expected_dim = mlp_emb.size(-1) + rgcn2_out.size(-1)
            assert out_layer.in_features == expected_dim, (
                f"Fusion layout mismatch: out.in_features={out_layer.in_features}, "
                f"but mlp_dim({mlp_emb.size(-1)}) + gnn_dim({rgcn2_out.size(-1)}) = {expected_dim}"
            )

        branch_logits = extract_branch_logits(model, mlp_emb, rgcn2_out)

        # ASDA switch
        asda_switch_raw = storage.get("asda_switch")

        # Assemble per-node records
        n = x.size(0)
        records = []

        # Vectorized branch gap computation
        mlp_logits_arr = branch_logits["mlp_logit"]
        gnn_logits_arr = branch_logits["gnn_logit"]

        for i in range(n):
            split = _determine_split(i, train_mask, val_mask, test_mask)
            hsd_val = hsd[i].item()
            prob_val = probs[i].item()
            logit_val = logits[i].cpu().item()

            # ASDA switch
            asda_val = 0.0
            if asda_switch_raw is not None:
                if asda_switch_raw.dim() == 2:
                    asda_val = asda_switch_raw[i, 0].cpu().item()
                else:
                    asda_val = asda_switch_raw[i].cpu().item()

            # Branch logits
            mlp_l = mlp_logits_arr[i].item() if mlp_logits_arr is not None else None
            gnn_l = gnn_logits_arr[i].item() if gnn_logits_arr is not None else None

            # Branch gap
            if mlp_l is not None and gnn_l is not None:
                mlp_prob = torch.sigmoid(torch.tensor(mlp_l)).item()
                gnn_prob = torch.sigmoid(torch.tensor(gnn_l)).item()
                gap = abs(mlp_prob - gnn_prob)
            else:
                gap = 0.0

            # HSD quantile (using pre-computed thresholds)
            hsd_quant = compute_hsd_quantile(hsd_val, hsd_thresholds)

            # High HSD flag
            high_hsd = hsd_val >= high_hsd_thresh

            # Discrepancy levels per relation (aligned with Evidence Card schema)
            disc_levels = {}
            degrees_per_relation = {}
            relation_discrepancies = {}
            for rel_name in relations:
                degrees_per_relation[rel_name] = int(
                    neighbor_stats.degrees_per_relation[rel_name][i].cpu()
                )
                disc_val = neighbor_stats.relation_discrepancy[rel_name][i].item()
                relation_discrepancies[rel_name] = float(disc_val)
                lo, hi = disc_thresholds[rel_name]
                disc_levels[rel_name] = _compute_discrepancy_level(disc_val, lo, hi)

            suspicious_neighbor_ratio = float(
                neighbor_stats.suspicious_neighbor_ratio[i].cpu()
            )
            top_relations = _build_top_relations(relation_discrepancies)
            neighbor_summary = _build_neighbor_summary(
                degrees_per_relation=degrees_per_relation,
                relation_discrepancies=relation_discrepancies,
                suspicious_neighbor_ratio=suspicious_neighbor_ratio,
                top_relations=top_relations,
            )

            record = {
                "node_id": i,
                "dataset": dataset_name,
                "split": split,
                "label": int(y[i].item()) if split != "test" else -1,  # no test labels
                "teacher_prob": prob_val,
                "teacher_logit": logit_val,
                "hsd": hsd_val,
                "hsd_quantile": hsd_quant,
                "asda_switch": asda_val,
                "mlp_logit": mlp_l,
                "gnn_logit": gnn_l,
                "branch_gap": gap,
                "high_hsd_flag": bool(high_hsd),
                "suspicious_neighbor_ratio": suspicious_neighbor_ratio,
                "topk_neighbors": int(neighbor_stats.topk_neighbors[i].cpu()),
                "neighbor_hsd_mean": float(neighbor_stats.neighbor_hsd_mean[i].cpu()),
                "top_relations": json.dumps(top_relations),
                "neighbor_summary": json.dumps(neighbor_summary),
                **{
                    f"degree_{rel}": degrees_per_relation[rel]
                    for rel in relation_names
                },
                **{
                    f"disc_{rel}": relation_discrepancies[rel]
                    for rel in relation_names
                },
                **{
                    f"disc_level_{rel}": disc_levels[rel]
                    for rel in relation_names
                },
            }
            records.append(record)

        df = pd.DataFrame(records)
        from priorf_reasoner_slm.evidence.serializer import serialize_row_to_json

        df["evidence_card_json"] = [
            serialize_row_to_json(row, relations=relation_names, dataset_name=dataset_name)
            for _, row in df.iterrows()
        ]
        df["card_json"] = df["evidence_card_json"]
        logger.info("Exported teacher evidence for %d nodes", len(df))
        return df

    finally:
        # Exception-safe hook cleanup
        remove_hooks(handles)


def save_evidence(df: pd.DataFrame, path: str | Path) -> Path:
    """Save teacher evidence to parquet or CSV based on extension."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    logger.info("Saved teacher evidence to %s (%d rows)", path, len(df))
    return path
