"""Extract structured features from teacher-exported DataFrame rows.

Provides helper functions that pull structured fields out of the flat
per-node DataFrame produced by priorf_teacher.export_scores, and return
them in a form suitable for EvidenceCard construction.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from priorf_reasoner_slm.evidence.evidence_schema import (
    NeighborStats,
    RelationEntry,
    StructureEvidence,
    TeacherSummary,
)


def extract_teacher_summary(row: pd.Series) -> TeacherSummary:
    """Extract teacher summary fields from a flat DataFrame row.

    Required columns: teacher_prob, branch_gap
    """
    return TeacherSummary(
        teacher_prob=float(row["teacher_prob"]),
        branch_gap=float(row["branch_gap"]),
    )


def discover_relations(df: pd.DataFrame) -> list[str]:
    """Discover relation names from the DataFrame column names.

    Looks for columns matching the pattern ``disc_level_{REL}`` and returns
    the relation names sorted alphabetically for deterministic ordering.
    """
    relations: list[str] = []
    for col in df.columns:
        if col.startswith("disc_level_"):
            rel_name = col[len("disc_level_"):]
            relations.append(rel_name)
    return sorted(relations)


def discover_row_relations(row: pd.Series) -> list[str]:
    """Discover relations that actually exist (non-null) for a single row.

    Unlike ``discover_relations`` which scans all columns in a DataFrame,
    this only returns relations whose ``disc_level_{REL}`` value is present
    and non-null in the given row. Used when processing individual rows
    from a per-dataset export where only that dataset's relations exist.
    """
    relations: list[str] = []
    for col in row.index:
        if col.startswith("disc_level_"):
            val = row[col]
            # Skip None, NaN, and empty strings
            if val is None or (isinstance(val, float) and val != val):
                continue
            val_str = str(val).strip().lower()
            if val_str in ("low", "medium", "high"):
                rel_name = col[len("disc_level_"):]
                relations.append(rel_name)
    return sorted(relations)


def extract_relation_profile(row: pd.Series, relations: list[str]) -> list[RelationEntry]:
    """Build the relation profile list from a flat DataFrame row.

    For each relation, reads the ``disc_level_{REL}`` column.
    Skips relations whose column is missing or has a null/NaN value.

    Args:
        row: A single row from the teacher export DataFrame.
        relations: Ordered list of relation names to include.

    Returns:
        List of RelationEntry objects.
    """
    profile: list[RelationEntry] = []
    for rel in relations:
        col = f"disc_level_{rel}"
        if col not in row.index:
            continue
        val = row[col]
        # Skip None, NaN, and empty strings
        if val is None or (isinstance(val, float) and val != val):
            continue
        level = str(val).lower().strip()
        if level not in ("low", "medium", "high"):
            continue
        profile.append(RelationEntry(relation=rel, discrepancy_level=level))
    return profile


def extract_neighbor_stats(row: pd.Series) -> NeighborStats:
    """Extract neighbor statistics from a flat DataFrame row.

    Required columns: suspicious_neighbor_ratio, topk_neighbors
    """
    return NeighborStats(
        suspicious_neighbor_ratio=float(row["suspicious_neighbor_ratio"]),
        topk_neighbors=int(row["topk_neighbors"]),
    )


def extract_structure_evidence(
    row: pd.Series,
    relations: list[str],
) -> StructureEvidence:
    """Build the full StructureEvidence sub-object from a flat row.

    Required columns:
        hsd_quantile, asda_switch, high_hsd_flag,
        suspicious_neighbor_ratio, topk_neighbors,
        disc_level_{REL} for each relation.

    Args:
        row: A single row from the teacher export DataFrame.
        relations: Ordered list of relation names.

    Returns:
        StructureEvidence model instance.
    """
    return StructureEvidence(
        hsd_quantile=str(row["hsd_quantile"]),
        asda_switch=float(row["asda_switch"]),
        high_hsd_flag=bool(row["high_hsd_flag"]),
        relation_profile=extract_relation_profile(row, relations),
        neighbor_stats=extract_neighbor_stats(row),
    )


def extract_label(row: pd.Series) -> int:
    """Extract the ground-truth label from a row.

    Returns -1 for test rows where labels are masked.
    """
    return int(row["label"])


def extract_split(row: pd.Series) -> str:
    """Extract the split assignment from a row."""
    return str(row["split"])
