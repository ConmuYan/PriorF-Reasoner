"""evidence: Evidence card construction, rationale generation, and SFT dataset building."""

from priorf_reasoner_slm.evidence.evidence_schema import (
    EvidenceCard,
    PredictionOutput,
    TeacherSummary,
    RelationEntry,
    NeighborStats,
    StructureEvidence,
    EVIDENCE_CARD_TASK,
    SYSTEM_PROMPT,
)
from priorf_reasoner_slm.evidence.feature_extractors import (
    discover_relations,
    discover_row_relations,
)
from priorf_reasoner_slm.evidence.serializer import (
    build_evidence_card,
    serialize_row_to_json,
    build_evidence_cards_from_dataframe,
)
from priorf_reasoner_slm.evidence.rationale_templates import generate_prediction
from priorf_reasoner_slm.evidence.dataset_builder import (
    build_prompt_completion_dataset,
    build_conversational_dataset,
    to_hf_dataset_prompt_completion,
    to_hf_dataset_conversational,
    save_jsonl,
    load_jsonl,
)

__all__ = [
    # Schema
    "EvidenceCard",
    "PredictionOutput",
    "TeacherSummary",
    "RelationEntry",
    "NeighborStats",
    "StructureEvidence",
    "EVIDENCE_CARD_TASK",
    "SYSTEM_PROMPT",
    # Feature extractors
    "discover_relations",
    "discover_row_relations",
    # Serializer
    "build_evidence_card",
    "serialize_row_to_json",
    "build_evidence_cards_from_dataframe",
    # Rationale
    "generate_prediction",
    # Dataset builder
    "build_prompt_completion_dataset",
    "build_conversational_dataset",
    "to_hf_dataset_prompt_completion",
    "to_hf_dataset_conversational",
    "save_jsonl",
    "load_jsonl",
]
