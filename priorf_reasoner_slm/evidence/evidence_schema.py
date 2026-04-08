"""Pydantic models for Evidence Card input and M3 prediction output.

Defines two schemas:
1. EvidenceCard  - the structured input card fed to the student LLM.
2. PredictionOutput - the M3 JSON the student must generate.

Both schemas enforce strict types so that serialization is deterministic
and round-trippable (json -> model -> json produces the same structure).
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# ---- Enum types for categorical validation ----


class DiscrepancyLevel(str, Enum):
    """Allowed discrepancy level values."""

    low = "low"
    medium = "medium"
    high = "high"


class HsdQuantile(str, Enum):
    """Allowed HSD quantile bucket values."""

    top_1_percent = "top_1_percent"
    top_5_percent = "top_5_percent"
    top_10_percent = "top_10_percent"
    normal = "normal"


class FraudLabel(str, Enum):
    """Allowed prediction label values."""

    fraud = "fraud"
    benign = "benign"


class PatternHint(str, Enum):
    """Allowed pattern hint values."""

    camouflage = "camouflage"
    co_attack = "co-attack"
    suspicious = "suspicious"


# ---- Nested sub-models for EvidenceCard ----


class TeacherSummary(BaseModel):
    """Teacher model summary statistics."""

    teacher_prob: float = Field(..., ge=0.0, le=1.0)
    branch_gap: float = Field(..., ge=0.0, le=1.0)


class RelationEntry(BaseModel):
    """A single relation in the relation profile."""

    relation: str
    discrepancy_level: DiscrepancyLevel  # constrained enum


class NeighborStats(BaseModel):
    """Neighborhood statistics for the node."""

    suspicious_neighbor_ratio: float = Field(..., ge=0.0, le=1.0)
    topk_neighbors: int = Field(..., ge=0)


class StructureEvidence(BaseModel):
    """Structural evidence extracted from the teacher model."""

    hsd_quantile: HsdQuantile  # constrained enum
    asda_switch: float = Field(..., ge=0.0, le=1.0)
    high_hsd_flag: bool
    relation_profile: list[RelationEntry]
    neighbor_stats: NeighborStats


class EvidenceCard(BaseModel):
    """Full Evidence Card (M2) fed as input to the student LLM.

    This schema matches the JSON structure defined in read.md.
    """

    dataset: str
    node_id: int = Field(..., ge=0)
    teacher_summary: TeacherSummary
    structure_evidence: StructureEvidence
    task: str = "Predict fraud label and explain using only the provided structural evidence."


# ---- M3 Prediction output schema ----


class PredictionOutput(BaseModel):
    """M3 prediction output that the student LLM must generate.

    The student is trained to produce JSON matching this schema.
    """

    label: FraudLabel  # constrained enum
    score: float = Field(..., ge=0.0, le=1.0)
    pattern_hint: PatternHint  # constrained enum
    evidence: list[str]
    rationale: str


# ---- Serialization helpers ----

EVIDENCE_CARD_TASK = (
    "Predict fraud label and explain using only the provided structural evidence."
)

SYSTEM_PROMPT = (
    "You are a fraud-reasoning assistant for graph-based anomaly detection. "
    "Given structured structural evidence about a node, predict whether it is "
    "fraudulent or benign, assign a confidence score, identify the behavioral "
    "pattern, list the key evidence, and provide a brief rationale. "
    "Respond only with valid JSON matching this exact schema: "
    '{"label":"fraud|benign","score":0.0,"pattern_hint":"camouflage|co-attack|suspicious","evidence":["..."],"rationale":"..."} '
    "Do not use alternative field names such as fraud_label, confidence, behavioral_pattern, or key_evidence."
)
