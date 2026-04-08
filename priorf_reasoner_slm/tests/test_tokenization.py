"""Tests for tokenizer_utils: tokenizer loading, chat formatting, and label masking."""

from __future__ import annotations

import json
import pytest

from priorf_reasoner_slm.llm.tokenizer_utils import (
    load_tokenizer,
    format_evidence_card,
    apply_chat_template,
    QWEN3_THINKING_DISABLED,
)


# ---- Fixtures ----


@pytest.fixture(scope="module")
def tokenizer():
    """Load the Qwen3-4B tokenizer (or skip if not available)."""
    try:
        tok = load_tokenizer("Qwen/Qwen3-4B", max_length=512)
    except Exception as exc:
        pytest.skip(f"Qwen3-4B tokenizer not available: {exc}")
    return tok


@pytest.fixture
def sample_card():
    """A minimal Evidence Card JSON string."""
    return json.dumps(
        {
            "dataset": "YelpChi",
            "node_id": 42,
            "teacher_summary": {"teacher_prob": 0.87, "branch_gap": 0.32},
            "structure_evidence": {
                "hsd_quantile": "top_5_percent",
                "asda_switch": 0.65,
                "high_hsd_flag": True,
                "relation_profile": [
                    {"relation": "r_u_u", "discrepancy_level": "high"},
                ],
                "neighbor_stats": {
                    "suspicious_neighbor_ratio": 0.45,
                    "topk_neighbors": 10,
                },
            },
            "task": "Predict fraud label and explain using only the provided structural evidence.",
        },
        ensure_ascii=False,
    )


@pytest.fixture
def sample_completion():
    """A minimal M3 prediction JSON string."""
    return json.dumps(
        {
            "label": "fraud",
            "score": 0.89,
            "pattern_hint": "camouflage",
            "evidence": ["high HSD", "suspicious neighbors"],
            "rationale": "Node shows high structural anomaly.",
        },
        ensure_ascii=False,
    )


# ---- Tests ----


class TestFormatEvidenceCard:
    """Tests for format_evidence_card."""

    def test_inference_mode_returns_two_messages(self, sample_card):
        """Without completion, only system + user messages."""
        msgs = format_evidence_card(sample_card)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_training_mode_returns_three_messages(self, sample_card, sample_completion):
        """With completion, system + user + assistant messages."""
        msgs = format_evidence_card(sample_card, completion_json=sample_completion)
        assert len(msgs) == 3
        assert msgs[2]["role"] == "assistant"
        assert msgs[2]["content"] == sample_completion

    def test_system_prompt_is_fraud_assistant(self, sample_card):
        """System prompt should mention fraud detection."""
        msgs = format_evidence_card(sample_card)
        assert "fraud" in msgs[0]["content"].lower()

    def test_user_content_is_card_json(self, sample_card):
        """User content should be the raw card JSON."""
        msgs = format_evidence_card(sample_card)
        # Should be valid JSON
        parsed = json.loads(msgs[1]["content"])
        assert parsed["node_id"] == 42

    def test_dict_input_serialized(self):
        """Passing a dict should auto-serialize to JSON."""
        card_dict = {"dataset": "test", "node_id": 0, "dummy": True}
        msgs = format_evidence_card(card_dict)
        parsed = json.loads(msgs[1]["content"])
        assert parsed["dataset"] == "test"


class TestApplyChatTemplate:
    """Tests for apply_chat_template."""

    def test_training_produces_input_ids_and_labels(self, tokenizer, sample_card, sample_completion):
        """Training mode returns both input_ids and labels."""
        msgs = format_evidence_card(sample_card, completion_json=sample_completion)
        result = apply_chat_template(tokenizer, msgs)
        assert "input_ids" in result
        assert "labels" in result
        assert len(result["input_ids"]) == len(result["labels"])

    def test_prompt_labels_are_masked(self, tokenizer, sample_card, sample_completion):
        """Labels for prompt tokens should be -100."""
        msgs = format_evidence_card(sample_card, completion_json=sample_completion)
        result = apply_chat_template(tokenizer, msgs)

        labels = result["labels"]
        # First token should be -100 (part of prompt)
        assert labels[0] == -100

        # Count -100 tokens (prompt portion)
        prompt_mask_count = sum(1 for l in labels if l == -100)
        # There should be at least some masked tokens
        assert prompt_mask_count > 0

        # There should be non-masked tokens (completion portion)
        completion_count = len(labels) - prompt_mask_count
        assert completion_count > 0

    def test_inference_produces_only_input_ids(self, tokenizer, sample_card):
        """Inference mode returns only input_ids (no labels)."""
        msgs = format_evidence_card(sample_card)
        result = apply_chat_template(
            tokenizer, msgs, add_generation_prompt=True
        )
        assert "input_ids" in result
        assert "labels" not in result

    def test_thinking_disabled_prefix(self, tokenizer, sample_card, sample_completion):
        """With disable_thinking, assistant content gets /no_think prefix."""
        msgs = format_evidence_card(sample_card, completion_json=sample_completion)
        result = apply_chat_template(tokenizer, msgs, disable_thinking=True)

        # The input_ids should be valid
        assert len(result["input_ids"]) > 0

        # Verify /no_think was injected into assistant message
        assistant_msgs = [m for m in msgs if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0]["content"].startswith(QWEN3_THINKING_DISABLED)

    def test_no_truncation_under_max_length(self, tokenizer, sample_card, sample_completion):
        """Result should not exceed max_length."""
        msgs = format_evidence_card(sample_card, completion_json=sample_completion)
        result = apply_chat_template(tokenizer, msgs, max_length=512)
        assert len(result["input_ids"]) <= 512
