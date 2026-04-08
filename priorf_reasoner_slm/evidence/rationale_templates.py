"""Rule-based rationale generation using templates (NO external API).

Generates the M3 PredictionOutput (label, score, pattern_hint, evidence list,
rationale text) from the structured evidence card fields. Pure deterministic
logic -- no LLM or API calls.

Pattern hint rules (v1):
    - camouflage:  high_hsd_flag AND asda_switch > 0.5
    - co-attack:   suspicious_neighbor_ratio > 0.5 AND branch_gap > 0.2
    - suspicious:  default fallback

The fallback priority is: camouflage > co-attack > suspicious. A node can
satisfy multiple conditions; the first match wins.
"""

from __future__ import annotations

from priorf_reasoner_slm.evidence.evidence_schema import (
    EvidenceCard,
    PredictionOutput,
    StructureEvidence,
    TeacherSummary,
)


# ---- Pattern hint determination ----

def determine_pattern_hint(
    high_hsd_flag: bool,
    asda_switch: float,
    suspicious_neighbor_ratio: float,
    branch_gap: float,
) -> str:
    """Determine the behavioral pattern hint from structural flags.

    Priority: camouflage > co-attack > suspicious.

    Args:
        high_hsd_flag: Whether the node is in the top HSD quantile.
        asda_switch: ASDA routing switch value [0, 1].
        suspicious_neighbor_ratio: Fraction of neighbors labeled fraud.
        branch_gap: Absolute difference between MLP and GNN branch probs.

    Returns:
        One of "camouflage", "co-attack", "suspicious".
    """
    if high_hsd_flag and asda_switch > 0.5:
        return "camouflage"
    if suspicious_neighbor_ratio > 0.5 and branch_gap > 0.2:
        return "co-attack"
    return "suspicious"


# ---- Evidence list generation ----

def build_evidence_list(
    high_hsd_flag: bool,
    hsd_quantile: str,
    asda_switch: float,
    relation_profile: list[dict],
    suspicious_neighbor_ratio: float,
    high_discrepancy_threshold: float = 0.5,
) -> list[str]:
    """Build the list of evidence item strings from structural flags.

    Args:
        high_hsd_flag: Whether node has high HSD.
        hsd_quantile: HSD quantile label string.
        asda_switch: ASDA switch value.
        relation_profile: List of {"relation": str, "discrepancy_level": str}.
        suspicious_neighbor_ratio: Fraction of suspicious neighbors.
        high_discrepancy_threshold: Threshold for flagging high switch.

    Returns:
        Ordered list of evidence description strings.
    """
    items: list[str] = []

    # HSD evidence
    if high_hsd_flag:
        items.append(f"high HSD quantile ({hsd_quantile})")
    elif hsd_quantile == "top_10_percent":
        items.append("elevated HSD quantile")

    # ASDA switch evidence
    if asda_switch > 0.5:
        items.append("high routing switch")
    elif asda_switch > 0.3:
        items.append("moderate routing switch")

    # Relation discrepancy evidence
    high_disc_rels = [
        rp["relation"]
        for rp in relation_profile
        if rp["discrepancy_level"] == "high"
    ]
    if high_disc_rels:
        rel_str = " and ".join(high_disc_rels)
        items.append(f"relation discrepancy concentrated on {rel_str}")

    # Neighbor evidence
    if suspicious_neighbor_ratio > 0.5:
        items.append("high suspicious neighbor ratio")
    elif suspicious_neighbor_ratio > 0.3:
        items.append("moderate suspicious neighbor ratio")

    # Fallback if no evidence items
    if not items:
        items.append("low structural anomaly signals")

    return items


# ---- Rationale text templates ----

_CAMOUFLAGE_TEMPLATES = {
    "high_hsd": "The node deviates strongly from its neighborhood",
    "high_switch": "the routing mechanism preserves residual discrepancy instead of smoothing it away",
    "combined": (
        "The node deviates strongly from its neighborhood and the routing "
        "mechanism preserves residual discrepancy instead of smoothing it away."
    ),
}

_COATTACK_TEMPLATES = {
    "high_neighbor_ratio": (
        "A large fraction of neighboring nodes are also flagged as suspicious"
    ),
    "branch_divergence": (
        "the MLP and GNN branches diverge significantly, indicating the node "
        "behaves differently in local versus global context"
    ),
    "combined": (
        "A large fraction of neighboring nodes are also flagged as suspicious, "
        "and the MLP and GNN branches diverge significantly, indicating "
        "coordinated anomalous behavior."
    ),
}

_SUSPICIOUS_TEMPLATE = (
    "The node exhibits moderate structural anomaly signals that warrant "
    "further attention, though no single indicator is strongly conclusive."
)


# ---- Rationale text assembly ----

def build_rationale(
    pattern_hint: str,
    high_hsd_flag: bool,
    asda_switch: float,
    suspicious_neighbor_ratio: float,
    branch_gap: float,
    high_disc_rels: list[str],
) -> str:
    """Assemble rationale text from template fragments.

    Args:
        pattern_hint: One of "camouflage", "co-attack", "suspicious".
        high_hsd_flag: Whether node has high HSD.
        asda_switch: ASDA switch value.
        suspicious_neighbor_ratio: Fraction of suspicious neighbors.
        branch_gap: Branch gap value.
        high_disc_rels: Relations with high discrepancy.

    Returns:
        Rationale text string.
    """
    if pattern_hint == "camouflage":
        parts = []
        if high_hsd_flag:
            parts.append(_CAMOUFLAGE_TEMPLATES["high_hsd"])
        if asda_switch > 0.5:
            if parts:
                parts.append("and " + _CAMOUFLAGE_TEMPLATES["high_switch"])
            else:
                parts.append(_CAMOUFLAGE_TEMPLATES["high_switch"])
        # Add relation-specific evidence if available
        if high_disc_rels and not parts:
            rel_str = " and ".join(high_disc_rels)
            parts.append(
                f"Relation discrepancy is concentrated on {rel_str}, "
                f"preserving residual discrepancy instead of smoothing it away."
            )
        elif high_disc_rels:
            rel_str = " and ".join(high_disc_rels)
            parts.append(f"with discrepancy concentrated on {rel_str}.")
        if not parts:
            parts.append(_CAMOUFLAGE_TEMPLATES["combined"])
        return " ".join(parts) + "."

    elif pattern_hint == "co-attack":
        parts = []
        if suspicious_neighbor_ratio > 0.5:
            parts.append(_COATTACK_TEMPLATES["high_neighbor_ratio"])
        if branch_gap > 0.2:
            if parts:
                parts.append("and " + _COATTACK_TEMPLATES["branch_divergence"])
            else:
                parts.append(_COATTACK_TEMPLATES["branch_divergence"])
        # Add relation-specific evidence
        if high_disc_rels:
            rel_str = " and ".join(high_disc_rels)
            parts.append(f"Discrepancy is particularly high on {rel_str} relations.")
        if not parts:
            parts.append(_COATTACK_TEMPLATES["combined"])
        return " ".join(parts) + "."

    else:  # suspicious
        if high_disc_rels:
            rel_str = " and ".join(high_disc_rels)
            return (
                f"The node exhibits moderate structural anomaly signals that warrant "
                f"further attention. Notable discrepancy on {rel_str} relations."
            )
        return _SUSPICIOUS_TEMPLATE


# ---- Label and score determination ----

def determine_label(teacher_prob: float, threshold: float = 0.5) -> str:
    """Map teacher probability to a label string.

    Args:
        teacher_prob: Teacher model probability of fraud.
        threshold: Classification threshold.

    Returns:
        "fraud" or "benign".
    """
    return "fraud" if teacher_prob >= threshold else "benign"


def compute_score(
    teacher_prob: float,
    branch_gap: float,
    pattern_hint: str,
) -> float:
    """Compute a confidence score.

    Uses the teacher probability with a small adjustment based on
    pattern hint strength. The score is clamped to [0, 1].

    Args:
        teacher_prob: Teacher model probability.
        branch_gap: Branch gap value.
        pattern_hint: Pattern hint string.

    Returns:
        Confidence score in [0, 1].
    """
    score = teacher_prob
    # Boost slightly for strong patterns, dampen for weak ones
    if pattern_hint == "camouflage":
        score = min(1.0, score + 0.02)
    elif pattern_hint == "co-attack":
        score = min(1.0, score + 0.01)
    else:
        score = max(0.0, score - 0.01)
    return round(score, 4)


# ---- Main generation function ----

def generate_prediction(card: EvidenceCard) -> PredictionOutput:
    """Generate a full M3 PredictionOutput from an EvidenceCard.

    This is the primary entry point for rationale generation. It
    extracts all needed fields from the card, applies the rule-based
    logic, and returns a validated PredictionOutput.

    Args:
        card: A validated EvidenceCard instance.

    Returns:
        A validated PredictionOutput instance.
    """
    se = card.structure_evidence
    ts = card.teacher_summary

    # Determine pattern hint
    pattern_hint = determine_pattern_hint(
        high_hsd_flag=se.high_hsd_flag,
        asda_switch=se.asda_switch,
        suspicious_neighbor_ratio=se.neighbor_stats.suspicious_neighbor_ratio,
        branch_gap=ts.branch_gap,
    )

    # Build evidence list
    relation_profile_dicts = [rp.model_dump() for rp in se.relation_profile]
    evidence_list = build_evidence_list(
        high_hsd_flag=se.high_hsd_flag,
        hsd_quantile=se.hsd_quantile,
        asda_switch=se.asda_switch,
        relation_profile=relation_profile_dicts,
        suspicious_neighbor_ratio=se.neighbor_stats.suspicious_neighbor_ratio,
    )

    # Build rationale text
    high_disc_rels = [
        rp.relation for rp in se.relation_profile
        if rp.discrepancy_level == "high"
    ]
    rationale = build_rationale(
        pattern_hint=pattern_hint,
        high_hsd_flag=se.high_hsd_flag,
        asda_switch=se.asda_switch,
        suspicious_neighbor_ratio=se.neighbor_stats.suspicious_neighbor_ratio,
        branch_gap=ts.branch_gap,
        high_disc_rels=high_disc_rels,
    )

    # Determine label and score
    label = determine_label(ts.teacher_prob)
    score = compute_score(
        teacher_prob=ts.teacher_prob,
        branch_gap=ts.branch_gap,
        pattern_hint=pattern_hint,
    )

    return PredictionOutput(
        label=label,
        score=score,
        pattern_hint=pattern_hint,
        evidence=evidence_list,
        rationale=rationale,
    )


def generate_prediction_from_row(row) -> dict:
    """Convenience: generate M3 output dict from a flat DataFrame row.

    Builds an EvidenceCard from the row, generates the prediction,
    and returns the output as a plain dict suitable for JSON serialization.

    Args:
        row: A pandas Series from the teacher-exported DataFrame.

    Returns:
        Dict matching the M3 PredictionOutput schema.
    """
    from priorf_reasoner_slm.evidence.serializer import build_evidence_card

    card = build_evidence_card(row)
    prediction = generate_prediction(card)
    return prediction.model_dump()
