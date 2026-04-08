"""Tests for the evidence module: schema, serializer, rationale generation."""

import json

import pandas as pd
import pytest

from priorf_reasoner_slm.evidence.evidence_schema import (
    EVIDENCE_CARD_TASK,
    SYSTEM_PROMPT,
    EvidenceCard,
    NeighborStats,
    PredictionOutput,
    RelationEntry,
    StructureEvidence,
    TeacherSummary,
)
from priorf_reasoner_slm.evidence.feature_extractors import (
    discover_relations,
    extract_neighbor_stats,
    extract_relation_profile,
    extract_structure_evidence,
    extract_teacher_summary,
)
from priorf_reasoner_slm.evidence.serializer import (
    build_evidence_card,
    build_evidence_cards_from_dataframe,
    serialize_row_to_json,
)
from priorf_reasoner_slm.evidence.rationale_templates import (
    build_evidence_list,
    build_rationale,
    compute_score,
    determine_label,
    determine_pattern_hint,
    generate_prediction,
    generate_prediction_from_row,
)


# ---- Fixtures ----

def _make_row(**overrides) -> pd.Series:
    """Build a minimal valid teacher-export row with sensible defaults."""
    defaults = {
        "node_id": 1832,
        "dataset": "YelpChi",
        "split": "train",
        "label": 1,
        "teacher_prob": 0.914,
        "teacher_logit": 2.35,
        "hsd": 1.82,
        "hsd_quantile": "top_5_percent",
        "asda_switch": 0.81,
        "mlp_logit": 2.0,
        "gnn_logit": 0.5,
        "branch_gap": 0.231,
        "high_hsd_flag": True,
        "suspicious_neighbor_ratio": 0.67,
        "topk_neighbors": 6,
        "neighbor_hsd_mean": 0.45,
        "degree_RUR": 12,
        "degree_RTR": 8,
        "degree_RSR": 5,
        "disc_RUR": 0.15,
        "disc_RTR": 0.82,
        "disc_RSR": 0.71,
        "disc_level_RUR": "low",
        "disc_level_RTR": "high",
        "disc_level_RSR": "high",
    }
    defaults.update(overrides)
    return pd.Series(defaults)


def _make_dataframe(n: int = 5, **row_overrides) -> pd.DataFrame:
    """Build a minimal valid teacher-export DataFrame with n rows."""
    rows = []
    for i in range(n):
        row = _make_row(node_id=i, **row_overrides)
        rows.append(row)
    return pd.DataFrame(rows)


RELATIONS = ["RSR", "RTR", "RUR"]  # sorted


# ============================================================
# Test: Pydantic Schema validation
# ============================================================


class TestTeacherSummary:
    def test_valid(self):
        ts = TeacherSummary(teacher_prob=0.914, branch_gap=0.231)
        assert ts.teacher_prob == 0.914
        assert ts.branch_gap == 0.231

    def test_invalid_prob_range(self):
        with pytest.raises(Exception):
            TeacherSummary(teacher_prob=1.5, branch_gap=0.1)

    def test_invalid_gap_range(self):
        with pytest.raises(Exception):
            TeacherSummary(teacher_prob=0.5, branch_gap=-0.1)


class TestRelationEntry:
    def test_valid(self):
        re = RelationEntry(relation="RUR", discrepancy_level="high")
        assert re.relation == "RUR"
        assert re.discrepancy_level == "high"


class TestNeighborStats:
    def test_valid(self):
        ns = NeighborStats(suspicious_neighbor_ratio=0.67, topk_neighbors=6)
        assert ns.suspicious_neighbor_ratio == 0.67
        assert ns.topk_neighbors == 6

    def test_invalid_ratio(self):
        with pytest.raises(Exception):
            NeighborStats(suspicious_neighbor_ratio=1.5, topk_neighbors=6)


class TestStructureEvidence:
    def test_valid(self):
        se = StructureEvidence(
            hsd_quantile="top_5_percent",
            asda_switch=0.81,
            high_hsd_flag=True,
            relation_profile=[
                RelationEntry(relation="RUR", discrepancy_level="low"),
                RelationEntry(relation="RTR", discrepancy_level="high"),
            ],
            neighbor_stats=NeighborStats(suspicious_neighbor_ratio=0.5, topk_neighbors=3),
        )
        assert se.high_hsd_flag is True
        assert len(se.relation_profile) == 2


class TestEvidenceCard:
    def test_valid_card(self):
        card = EvidenceCard(
            dataset="YelpChi",
            node_id=1832,
            teacher_summary=TeacherSummary(teacher_prob=0.914, branch_gap=0.231),
            structure_evidence=StructureEvidence(
                hsd_quantile="top_5_percent",
                asda_switch=0.81,
                high_hsd_flag=True,
                relation_profile=[
                    RelationEntry(relation="RUR", discrepancy_level="low"),
                    RelationEntry(relation="RTR", discrepancy_level="high"),
                ],
                neighbor_stats=NeighborStats(suspicious_neighbor_ratio=0.67, topk_neighbors=6),
            ),
        )
        assert card.dataset == "YelpChi"
        assert card.node_id == 1832
        assert card.task == EVIDENCE_CARD_TASK

    def test_json_roundtrip(self):
        card = EvidenceCard(
            dataset="Amazon",
            node_id=0,
            teacher_summary=TeacherSummary(teacher_prob=0.5, branch_gap=0.1),
            structure_evidence=StructureEvidence(
                hsd_quantile="normal",
                asda_switch=0.3,
                high_hsd_flag=False,
                relation_profile=[],
                neighbor_stats=NeighborStats(suspicious_neighbor_ratio=0.1, topk_neighbors=0),
            ),
        )
        json_str = card.model_dump_json()
        parsed = json.loads(json_str)
        card2 = EvidenceCard.model_validate(parsed)
        assert card2.dataset == card.dataset
        assert card2.node_id == card.node_id
        assert card2.teacher_summary.teacher_prob == card.teacher_summary.teacher_prob


class TestPredictionOutput:
    def test_valid_output(self):
        out = PredictionOutput(
            label="fraud",
            score=0.93,
            pattern_hint="camouflage",
            evidence=["high HSD quantile", "high routing switch"],
            rationale="The node deviates strongly from its neighborhood.",
        )
        assert out.label == "fraud"
        assert out.pattern_hint == "camouflage"

    def test_json_roundtrip(self):
        out = PredictionOutput(
            label="benign",
            score=0.12,
            pattern_hint="suspicious",
            evidence=["low structural anomaly signals"],
            rationale="Moderate signals.",
        )
        json_str = out.model_dump_json()
        parsed = json.loads(json_str)
        out2 = PredictionOutput.model_validate(parsed)
        assert out2.label == out.label
        assert out2.score == out.score


# ============================================================
# Test: Feature extractors
# ============================================================


class TestDiscoverRelations:
    def test_discovers_from_dataframe(self):
        df = _make_dataframe()
        rels = discover_relations(df)
        assert sorted(rels) == ["RSR", "RTR", "RUR"]

    def test_empty_dataframe(self):
        df = pd.DataFrame({"node_id": [0]})
        rels = discover_relations(df)
        assert rels == []


class TestExtractTeacherSummary:
    def test_extracts_correctly(self):
        row = _make_row()
        ts = extract_teacher_summary(row)
        assert ts.teacher_prob == 0.914
        assert ts.branch_gap == 0.231


class TestExtractRelationProfile:
    def test_extracts_all_relations(self):
        row = _make_row()
        profile = extract_relation_profile(row, RELATIONS)
        assert len(profile) == 3
        # RUR=low, RTR=high, RSR=high (from _make_row defaults)
        levels = {rp.relation: rp.discrepancy_level for rp in profile}
        assert levels["RUR"] == "low"
        assert levels["RTR"] == "high"
        assert levels["RSR"] == "high"


class TestExtractNeighborStats:
    def test_extracts_correctly(self):
        row = _make_row()
        ns = extract_neighbor_stats(row)
        assert ns.suspicious_neighbor_ratio == 0.67
        assert ns.topk_neighbors == 6


class TestExtractStructureEvidence:
    def test_extracts_full_evidence(self):
        row = _make_row()
        se = extract_structure_evidence(row, RELATIONS)
        assert se.hsd_quantile == "top_5_percent"
        assert se.asda_switch == 0.81
        assert se.high_hsd_flag is True
        assert len(se.relation_profile) == 3


# ============================================================
# Test: Serializer
# ============================================================


class TestBuildEvidenceCard:
    def test_builds_from_row(self):
        row = _make_row()
        card = build_evidence_card(row)
        assert card.dataset == "YelpChi"
        assert card.node_id == 1832
        assert card.teacher_summary.teacher_prob == 0.914
        assert card.structure_evidence.high_hsd_flag is True
        assert len(card.structure_evidence.relation_profile) == 3

    def test_auto_discovers_relations(self):
        row = _make_row()
        card = build_evidence_card(row)  # No explicit relations
        assert len(card.structure_evidence.relation_profile) == 3

    def test_dataset_name_override(self):
        row = _make_row()
        card = build_evidence_card(row, dataset_name="CustomDS")
        assert card.dataset == "CustomDS"


class TestSerializeRowToJson:
    def test_produces_valid_json(self):
        row = _make_row()
        json_str = serialize_row_to_json(row)
        parsed = json.loads(json_str)
        assert parsed["node_id"] == 1832
        assert parsed["teacher_summary"]["teacher_prob"] == 0.914

    def test_compact_mode(self):
        row = _make_row()
        json_str = serialize_row_to_json(row, indent=None)
        assert "\n" not in json_str


class TestBuildEvidenceCardsFromDataFrame:
    def test_builds_all_rows(self):
        df = _make_dataframe(n=5)
        cards = build_evidence_cards_from_dataframe(df)
        assert len(cards) == 5
        assert all(isinstance(c, EvidenceCard) for c in cards)

    def test_consistent_relations_across_cards(self):
        df = _make_dataframe(n=3)
        cards = build_evidence_cards_from_dataframe(df)
        for card in cards:
            rel_names = [rp.relation for rp in card.structure_evidence.relation_profile]
            assert rel_names == sorted(rel_names)


# ============================================================
# Test: Rationale templates
# ============================================================


class TestDeterminePatternHint:
    def test_camouflage(self):
        hint = determine_pattern_hint(
            high_hsd_flag=True, asda_switch=0.81,
            suspicious_neighbor_ratio=0.1, branch_gap=0.1,
        )
        assert hint == "camouflage"

    def test_camouflage_requires_both(self):
        # high_hsd_flag True but asda_switch <= 0.5 => not camouflage
        hint = determine_pattern_hint(
            high_hsd_flag=True, asda_switch=0.3,
            suspicious_neighbor_ratio=0.1, branch_gap=0.1,
        )
        assert hint != "camouflage"

    def test_co_attack(self):
        hint = determine_pattern_hint(
            high_hsd_flag=False, asda_switch=0.3,
            suspicious_neighbor_ratio=0.67, branch_gap=0.3,
        )
        assert hint == "co-attack"

    def test_co_attack_requires_both(self):
        # ratio > 0.5 but gap <= 0.2 => not co-attack
        hint = determine_pattern_hint(
            high_hsd_flag=False, asda_switch=0.1,
            suspicious_neighbor_ratio=0.67, branch_gap=0.1,
        )
        assert hint == "suspicious"

    def test_suspicious_fallback(self):
        hint = determine_pattern_hint(
            high_hsd_flag=False, asda_switch=0.1,
            suspicious_neighbor_ratio=0.1, branch_gap=0.1,
        )
        assert hint == "suspicious"

    def test_camouflage_priority_over_coattack(self):
        """Camouflage should win when both conditions are met."""
        hint = determine_pattern_hint(
            high_hsd_flag=True, asda_switch=0.81,
            suspicious_neighbor_ratio=0.67, branch_gap=0.3,
        )
        assert hint == "camouflage"


class TestBuildEvidenceList:
    def test_camouflage_evidence(self):
        items = build_evidence_list(
            high_hsd_flag=True,
            hsd_quantile="top_5_percent",
            asda_switch=0.81,
            relation_profile=[
                {"relation": "RTR", "discrepancy_level": "high"},
            ],
            suspicious_neighbor_ratio=0.2,
        )
        assert any("HSD" in item for item in items)
        assert any("routing switch" in item for item in items)
        assert any("RTR" in item for item in items)

    def test_low_signals(self):
        items = build_evidence_list(
            high_hsd_flag=False,
            hsd_quantile="normal",
            asda_switch=0.1,
            relation_profile=[],
            suspicious_neighbor_ratio=0.05,
        )
        assert items == ["low structural anomaly signals"]

    def test_high_neighbor_ratio(self):
        items = build_evidence_list(
            high_hsd_flag=False,
            hsd_quantile="normal",
            asda_switch=0.1,
            relation_profile=[],
            suspicious_neighbor_ratio=0.6,
        )
        assert any("suspicious neighbor" in item for item in items)


class TestBuildRationale:
    def test_camouflage_rationale(self):
        text = build_rationale(
            pattern_hint="camouflage",
            high_hsd_flag=True,
            asda_switch=0.81,
            suspicious_neighbor_ratio=0.1,
            branch_gap=0.1,
            high_disc_rels=["RTR"],
        )
        assert "deviates strongly" in text
        assert "routing mechanism" in text

    def test_co_attack_rationale(self):
        text = build_rationale(
            pattern_hint="co-attack",
            high_hsd_flag=False,
            asda_switch=0.3,
            suspicious_neighbor_ratio=0.67,
            branch_gap=0.3,
            high_disc_rels=[],
        )
        assert "neighboring nodes" in text
        assert "diverge" in text

    def test_suspicious_rationale(self):
        text = build_rationale(
            pattern_hint="suspicious",
            high_hsd_flag=False,
            asda_switch=0.1,
            suspicious_neighbor_ratio=0.1,
            branch_gap=0.1,
            high_disc_rels=[],
        )
        assert "moderate" in text.lower() or "structural" in text.lower()


class TestDetermineLabel:
    def test_fraud(self):
        assert determine_label(0.914) == "fraud"

    def test_benign(self):
        assert determine_label(0.3) == "benign"

    def test_boundary(self):
        assert determine_label(0.5) == "fraud"
        assert determine_label(0.4999) == "benign"


class TestComputeScore:
    def test_score_clamped(self):
        assert compute_score(0.99, 0.1, "camouflage") <= 1.0
        assert compute_score(0.01, 0.1, "suspicious") >= 0.0

    def test_camouflage_boosted(self):
        base = compute_score(0.5, 0.1, "camouflage")
        suspicious = compute_score(0.5, 0.1, "suspicious")
        assert base >= suspicious


class TestGeneratePrediction:
    def test_generates_valid_output(self):
        row = _make_row()
        card = build_evidence_card(row)
        prediction = generate_prediction(card)
        assert isinstance(prediction, PredictionOutput)
        assert prediction.label == "fraud"
        assert prediction.pattern_hint == "camouflage"
        assert len(prediction.evidence) > 0
        assert len(prediction.rationale) > 0

    def test_benign_node(self):
        row = _make_row(
            teacher_prob=0.2,
            branch_gap=0.05,
            hsd_quantile="normal",
            high_hsd_flag=False,
            asda_switch=0.1,
            suspicious_neighbor_ratio=0.05,
        )
        card = build_evidence_card(row)
        prediction = generate_prediction(card)
        assert prediction.label == "benign"
        assert prediction.pattern_hint == "suspicious"

    def test_prediction_json_roundtrip(self):
        row = _make_row()
        card = build_evidence_card(row)
        prediction = generate_prediction(card)
        json_str = prediction.model_dump_json()
        parsed = json.loads(json_str)
        prediction2 = PredictionOutput.model_validate(parsed)
        assert prediction2.label == prediction.label
        assert prediction2.score == prediction.score
        assert prediction2.pattern_hint == prediction.pattern_hint
        assert prediction2.evidence == prediction.evidence


class TestGeneratePredictionFromRow:
    def test_generates_dict(self):
        row = _make_row()
        result = generate_prediction_from_row(row)
        assert isinstance(result, dict)
        assert "label" in result
        assert "score" in result
        assert "pattern_hint" in result
        assert "evidence" in result
        assert "rationale" in result

    def test_result_is_serializable(self):
        row = _make_row()
        result = generate_prediction_from_row(row)
        json_str = json.dumps(result)
        assert isinstance(json_str, str)
