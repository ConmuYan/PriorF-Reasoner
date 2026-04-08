"""Serialize flat teacher-exported DataFrame rows into nested Evidence Cards.

The main entry point is ``build_evidence_card``, which takes a single row
from the teacher-exported parquet/CSV and returns a fully validated
EvidenceCard Pydantic model.

Also provides ``serialize_row_to_json`` for direct JSON string output,
and ``build_evidence_cards_from_dataframe`` for batch conversion.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd

from priorf_reasoner_slm.evidence.evidence_schema import (
    EVIDENCE_CARD_TASK,
    EvidenceCard,
)
from priorf_reasoner_slm.evidence.feature_extractors import (
    discover_relations,
    discover_row_relations,
    extract_neighbor_stats,
    extract_relation_profile,
    extract_structure_evidence,
    extract_teacher_summary,
)

logger = logging.getLogger(__name__)


def build_evidence_card(
    row: pd.Series,
    relations: list[str] | None = None,
    dataset_name: str | None = None,
) -> EvidenceCard:
    """Convert a single flat DataFrame row into an EvidenceCard model.

    Args:
        row: One row from the teacher-exported DataFrame.
        relations: Optional pre-discovered relation names. If None,
            they are inferred from the row's index/columns. When a
            plain pd.Series is passed (e.g. df.iterrows()), the
            index is used; for a dict-backed row, the caller should
            supply ``relations`` explicitly.
        dataset_name: Override for the dataset field. If None, reads
            from the row's ``dataset`` column.

    Returns:
        A validated EvidenceCard instance.
    """
    # Discover relations if not provided — use row-level discovery
    # to skip null/NaN columns that belong to other datasets.
    if relations is None:
        relations = discover_row_relations(row)

    dataset = dataset_name or str(row.get("dataset", "unknown"))

    card = EvidenceCard(
        dataset=dataset,
        node_id=int(row["node_id"]),
        teacher_summary=extract_teacher_summary(row),
        structure_evidence=extract_structure_evidence(row, relations),
        task=EVIDENCE_CARD_TASK,
    )
    return card


def serialize_row_to_json(
    row: pd.Series,
    relations: list[str] | None = None,
    dataset_name: str | None = None,
    indent: int | None = None,
) -> str:
    """Convert a flat DataFrame row to an Evidence Card JSON string.

    Args:
        row: One row from the teacher-exported DataFrame.
        relations: Optional pre-discovered relation names.
        dataset_name: Optional dataset name override.
        indent: JSON indentation level. None for compact output.

    Returns:
        JSON string of the Evidence Card.
    """
    card = build_evidence_card(row, relations=relations, dataset_name=dataset_name)
    return card.model_dump_json(indent=indent)


def build_evidence_cards_from_dataframe(
    df: pd.DataFrame,
    dataset_name: str | None = None,
) -> list[EvidenceCard]:
    """Convert an entire DataFrame into a list of EvidenceCard instances.

    Discovers relation names once from the DataFrame columns and applies
    them to every row for consistency and performance.

    Args:
        df: Teacher-exported DataFrame.
        dataset_name: Optional dataset name override for all cards.

    Returns:
        List of EvidenceCard instances, one per row.
    """
    relations = discover_relations(df)
    dataset = dataset_name  # let per-row override work if this is None

    cards: list[EvidenceCard] = []
    for _, row in df.iterrows():
        # Use row-level relation discovery to skip null columns
        row_relations = discover_row_relations(row) or relations
        card = build_evidence_card(
            row,
            relations=row_relations,
            dataset_name=dataset,
        )
        cards.append(card)

    logger.info("Built %d Evidence Cards from DataFrame (%d relations)", len(cards), len(relations))
    return cards
