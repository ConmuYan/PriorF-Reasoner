"""Tests for dataset_builder: prompt-completion and conversational formats."""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from priorf_reasoner_slm.evidence.dataset_builder import (
    build_conversational_dataset,
    build_conversational_sample,
    build_prompt_completion_dataset,
    build_prompt_completion_sample,
    evidence_card_to_prompt,
    load_jsonl,
    prediction_to_completion,
    save_jsonl,
    to_hf_dataset_conversational,
    to_hf_dataset_prompt_completion,
)
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
from priorf_reasoner_slm.evidence.rationale_templates import generate_prediction
from priorf_reasoner_slm.evidence.serializer import build_evidence_card


# ---- Fixtures ----


def _make_row(**overrides) -> pd.Series:
    """Build a minimal valid teacher-export row."""
    defaults = {
        "node_id": 0,
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


def _make_card(**row_overrides) -> EvidenceCard:
    """Build a card from a row."""
    row = _make_row(**row_overrides)
    return build_evidence_card(row)


def _make_dataframe(n: int = 5, split: str = "train") -> pd.DataFrame:
    """Build a minimal teacher-export DataFrame."""
    rows = []
    for i in range(n):
        row = _make_row(node_id=i, split=split)
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
# Test: Single sample builders
# ============================================================


class TestEvidenceCardToPrompt:
    def test_produces_json_string(self):
        card = _make_card()
        prompt = evidence_card_to_prompt(card)
        assert isinstance(prompt, str)
        parsed = json.loads(prompt)
        assert "dataset" in parsed
        assert "structure_evidence" in parsed

    def test_includes_task(self):
        card = _make_card()
        prompt = evidence_card_to_prompt(card)
        parsed = json.loads(prompt)
        assert parsed["task"] == EVIDENCE_CARD_TASK

    def test_compact_mode(self):
        card = _make_card()
        prompt = evidence_card_to_prompt(card, indent=None)
        assert "\n" not in prompt


class TestPredictionToCompletion:
    def test_produces_compact_json(self):
        card = _make_card()
        pred = generate_prediction(card)
        completion = prediction_to_completion(pred)
        assert isinstance(completion, str)
        parsed = json.loads(completion)
        assert "label" in parsed
        assert "score" in parsed
        assert "pattern_hint" in parsed
        assert "evidence" in parsed
        assert "rationale" in parsed


class TestBuildPromptCompletionSample:
    def test_structure(self):
        card = _make_card()
        sample = build_prompt_completion_sample(card)
        assert "prompt" in sample
        assert "completion" in sample
        assert isinstance(sample["prompt"], str)
        assert isinstance(sample["completion"], str)

    def test_completion_is_parseable(self):
        card = _make_card()
        sample = build_prompt_completion_sample(card)
        completion = json.loads(sample["completion"])
        assert "label" in completion
        assert "rationale" in completion

    def test_custom_prediction(self):
        card = _make_card()
        custom_pred = PredictionOutput(
            label="fraud",
            score=0.99,
            pattern_hint="camouflage",
            evidence=["custom evidence"],
            rationale="Custom rationale.",
        )
        sample = build_prompt_completion_sample(card, prediction=custom_pred)
        completion = json.loads(sample["completion"])
        assert completion["score"] == 0.99
        assert completion["evidence"] == ["custom evidence"]


class TestBuildConversationalSample:
    def test_structure(self):
        card = _make_card()
        sample = build_conversational_sample(card)
        assert "messages" in sample
        assert len(sample["messages"]) == 3

    def test_roles(self):
        card = _make_card()
        sample = build_conversational_sample(card)
        roles = [m["role"] for m in sample["messages"]]
        assert roles == ["system", "user", "assistant"]

    def test_system_prompt(self):
        card = _make_card()
        sample = build_conversational_sample(card)
        assert sample["messages"][0]["content"] == SYSTEM_PROMPT

    def test_user_is_card_json(self):
        card = _make_card()
        sample = build_conversational_sample(card)
        parsed = json.loads(sample["messages"][1]["content"])
        assert parsed["node_id"] == 0

    def test_assistant_is_prediction_json(self):
        card = _make_card()
        sample = build_conversational_sample(card)
        parsed = json.loads(sample["messages"][2]["content"])
        assert "label" in parsed
        assert "score" in parsed


# ============================================================
# Test: Batch dataset builders
# ============================================================


class TestBuildPromptCompletionDataset:
    def test_builds_from_dataframe(self):
        df = _make_dataframe(n=5)
        samples = build_prompt_completion_dataset(df)
        assert len(samples) == 5
        for s in samples:
            assert "prompt" in s
            assert "completion" in s

    def test_split_filter(self):
        df = pd.DataFrame([
            _make_row(node_id=0, split="train"),
            _make_row(node_id=1, split="train"),
            _make_row(node_id=2, split="test"),
        ])
        samples = build_prompt_completion_dataset(df, splits=["train"])
        assert len(samples) == 2

    def test_split_filter_multiple(self):
        df = pd.DataFrame([
            _make_row(node_id=0, split="train"),
            _make_row(node_id=1, split="val"),
            _make_row(node_id=2, split="test"),
        ])
        samples = build_prompt_completion_dataset(df, splits=["train", "val"])
        assert len(samples) == 2

    def test_all_splits_by_default(self):
        df = pd.DataFrame([
            _make_row(node_id=0, split="train"),
            _make_row(node_id=1, split="val"),
            _make_row(node_id=2, split="test"),
        ])
        samples = build_prompt_completion_dataset(df)
        assert len(samples) == 3


class TestBuildConversationalDataset:
    def test_builds_from_dataframe(self):
        df = _make_dataframe(n=3)
        samples = build_conversational_dataset(df)
        assert len(samples) == 3
        for s in samples:
            assert len(s["messages"]) == 3

    def test_split_filter(self):
        df = pd.DataFrame([
            _make_row(node_id=0, split="train"),
            _make_row(node_id=1, split="test"),
        ])
        samples = build_conversational_dataset(df, splits=["train"])
        assert len(samples) == 1


# ============================================================
# Test: HuggingFace Dataset conversion
# ============================================================


class TestToHfDatasetPromptCompletion:
    def test_creates_dataset(self):
        from datasets import Dataset

        df = _make_dataframe(n=3)
        samples = build_prompt_completion_dataset(df)
        ds = to_hf_dataset_prompt_completion(samples)
        assert isinstance(ds, Dataset)
        assert len(ds) == 3
        assert "prompt" in ds.column_names
        assert "completion" in ds.column_names

    def test_content_preserved(self):
        df = _make_dataframe(n=2)
        samples = build_prompt_completion_dataset(df)
        ds = to_hf_dataset_prompt_completion(samples)
        for i in range(2):
            prompt = ds[i]["prompt"]
            completion = ds[i]["completion"]
            assert isinstance(prompt, str)
            assert isinstance(completion, str)
            json.loads(prompt)  # Must be valid JSON
            json.loads(completion)  # Must be valid JSON


class TestToHfDatasetConversational:
    def test_creates_dataset(self):
        from datasets import Dataset

        df = _make_dataframe(n=3)
        samples = build_conversational_dataset(df)
        ds = to_hf_dataset_conversational(samples)
        assert isinstance(ds, Dataset)
        assert len(ds) == 3
        assert "messages" in ds.column_names

    def test_messages_structure(self):
        df = _make_dataframe(n=2)
        samples = build_conversational_dataset(df)
        ds = to_hf_dataset_conversational(samples)
        for i in range(2):
            messages = ds[i]["messages"]
            assert len(messages) == 3
            roles = [m["role"] for m in messages]
            assert roles == ["system", "user", "assistant"]


# ============================================================
# Test: JSONL I/O
# ============================================================


class TestJsonlIO:
    def test_save_and_load_roundtrip(self, tmp_path):
        df = _make_dataframe(n=4)
        samples = build_prompt_completion_dataset(df)
        path = save_jsonl(samples, tmp_path / "test.jsonl")
        assert path.exists()

        loaded = load_jsonl(path)
        assert len(loaded) == 4
        for orig, load in zip(samples, loaded):
            assert orig["prompt"] == load["prompt"]
            assert orig["completion"] == load["completion"]

    def test_conversational_jsonl(self, tmp_path):
        df = _make_dataframe(n=2)
        samples = build_conversational_dataset(df)
        path = save_jsonl(samples, tmp_path / "conv.jsonl")
        loaded = load_jsonl(path)
        assert len(loaded) == 2
        for item in loaded:
            assert "messages" in item
            assert len(item["messages"]) == 3

    def test_creates_parent_dirs(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c" / "data.jsonl"
        save_jsonl([{"test": 1}], nested)
        assert nested.exists()

    def test_empty_file(self, tmp_path):
        path = save_jsonl([], tmp_path / "empty.jsonl")
        loaded = load_jsonl(path)
        assert loaded == []


# ============================================================
# Test: End-to-end pipeline
# ============================================================


class TestEndToEnd:
    def test_full_pipeline_prompt_completion(self):
        """Simulate: DataFrame -> Evidence Cards -> predictions -> dataset."""
        df = _make_dataframe(n=3)

        # Build dataset
        samples = build_prompt_completion_dataset(df, dataset_name="TestDS")

        # Validate structure
        for sample in samples:
            prompt = json.loads(sample["prompt"])
            completion = json.loads(sample["completion"])

            # Prompt must have all required fields
            assert "dataset" in prompt
            assert prompt["dataset"] == "TestDS"
            assert "node_id" in prompt
            assert "teacher_summary" in prompt
            assert "structure_evidence" in prompt
            assert "task" in prompt

            # Completion must be valid M3
            assert completion["label"] in ("fraud", "benign")
            assert 0 <= completion["score"] <= 1
            assert completion["pattern_hint"] in ("camouflage", "co-attack", "suspicious")
            assert isinstance(completion["evidence"], list)
            assert isinstance(completion["rationale"], str)

    def test_full_pipeline_conversational(self):
        """Simulate: DataFrame -> conversational dataset -> HF Dataset."""
        from datasets import Dataset

        df = _make_dataframe(n=3)
        samples = build_conversational_dataset(df)
        ds = to_hf_dataset_conversational(samples)

        assert isinstance(ds, Dataset)
        assert len(ds) == 3

        # Verify each sample has proper messages
        for i in range(3):
            messages = ds[i]["messages"]
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"
            assert messages[2]["role"] == "assistant"

            # Verify content is valid JSON
            user_content = json.loads(messages[1]["content"])
            assert "structure_evidence" in user_content

            assistant_content = json.loads(messages[2]["content"])
            assert "label" in assistant_content

    def test_full_pipeline_with_file_roundtrip(self, tmp_path):
        """DataFrame -> build -> save JSONL -> load JSONL -> all parseable."""
        df = _make_dataframe(n=5)
        samples = build_prompt_completion_dataset(df)
        path = save_jsonl(samples, tmp_path / "train.jsonl")
        loaded = load_jsonl(path)

        assert len(loaded) == 5
        for sample in loaded:
            completion = json.loads(sample["completion"])
            PredictionOutput.model_validate(completion)  # Must validate
