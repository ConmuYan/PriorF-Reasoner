"""Tests for llm/tokenizer_utils: mocked tokenizer loading and key contract tests."""

from __future__ import annotations

import json
import pytest
from unittest.mock import patch, MagicMock

from priorf_reasoner_slm.llm.tokenizer_utils import (
    load_tokenizer,
    format_evidence_card,
    apply_chat_template,
    QWEN3_THINKING_DISABLED,
)


# ---- Fixtures ----


@pytest.fixture
def mock_tokenizer():
    """A mock tokenizer that simulates Qwen3-4B without downloading anything."""
    mock_tok = MagicMock()
    mock_tok.pad_token = None
    mock_tok.eos_token = "<|endoftext|>"
    mock_tok.eos_token_id = 151643
    mock_tok.vocab_size = 151936
    mock_tok.model_max_length = 4096
    mock_tok.padding_side = "right"

    # Consistent vocabulary for word-based tokenization
    _NEXT_ID = [200]  # closure mutable counter for fresh stable IDs

    def _get_word_id(word: str) -> int:
        """Map a word to a stable token ID (same word -> same ID)."""
        if not hasattr(_get_word_id, "_cache"):
            _get_word_id._cache = {}
        if word not in _get_word_id._cache:
            _get_word_id._cache[word] = _NEXT_ID[0]
            _NEXT_ID[0] += 1
        return _get_word_id._cache[word]

    def _tokenize_text(text: str) -> list[int]:
        words = text.replace("<|im_start|>", "").replace("<|im_end|>", "").split()
        return [_get_word_id(w) for w in words]

    def apply_chat_template_side_effect(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=False,
    ):
        # Simulate what Qwen3 chat template produces (simplified)
        text = ""
        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "system":
                text += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                text += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        if tokenize:
            token_ids = _tokenize_text(text)
            if return_dict:
                return {"input_ids": token_ids}
            return token_ids
        return text

    mock_tok.apply_chat_template.side_effect = apply_chat_template_side_effect

    def encode_side_effect(text, add_special_tokens=True):
        return _tokenize_text(text)

    mock_tok.encode.side_effect = encode_side_effect

    return mock_tok


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
            "task": "Predict fraud label.",
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
            "evidence": ["high HSD"],
            "rationale": "Node shows anomaly.",
        },
        ensure_ascii=False,
    )


# ---- Tests ----


class TestLoadTokenizer:
    """Tests for load_tokenizer with mocking."""

    @patch("priorf_reasoner_slm.llm.tokenizer_utils.AutoTokenizer")
    def test_load_tokenizer_returns_tokenizer(self, mock_autotokenizer):
        """load_tokenizer should return an AutoTokenizer instance."""
        mock_tok_instance = MagicMock()
        mock_tok_instance.pad_token = None
        mock_tok_instance.eos_token = "<|endoftext|>"
        mock_tok_instance.eos_token_id = 151643
        mock_tok_instance.vocab_size = 151936
        mock_tok_instance.model_max_length = 4096
        mock_autotokenizer.from_pretrained.return_value = mock_tok_instance

        tok = load_tokenizer("Qwen/Qwen3-4B", max_length=512)

        mock_autotokenizer.from_pretrained.assert_called_once_with(
            "Qwen/Qwen3-4B",
            trust_remote_code=True,
            padding_side="right",
        )
        assert tok.pad_token == tok.eos_token
        assert tok.pad_token_id == tok.eos_token_id
        assert tok.model_max_length == 512

    @patch("priorf_reasoner_slm.llm.tokenizer_utils.AutoTokenizer")
    def test_load_tokenizer_no_download(self, mock_autotokenizer):
        """load_tokenizer should NOT trigger actual model download (uses AutoTokenizer)."""
        mock_tok_instance = MagicMock()
        mock_tok_instance.pad_token = None
        mock_tok_instance.eos_token = "<|endoftext|>"
        mock_tok_instance.eos_token_id = 151643
        mock_tok_instance.vocab_size = 151936
        mock_tok_instance.model_max_length = 4096
        mock_autotokenizer.from_pretrained.return_value = mock_tok_instance

        load_tokenizer("Qwen/Qwen3-4B")

        # Verify only AutoTokenizer.from_pretrained was called (no download)
        mock_autotokenizer.from_pretrained.assert_called_once()


class TestFormatEvidenceCard:
    """Tests for format_evidence_card formatting."""

    def test_inference_mode_two_messages(self, sample_card):
        """Without completion: system + user."""
        msgs = format_evidence_card(sample_card)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_training_mode_three_messages(self, sample_card, sample_completion):
        """With completion: system + user + assistant."""
        msgs = format_evidence_card(sample_card, completion_json=sample_completion)
        assert len(msgs) == 3
        assert msgs[2]["role"] == "assistant"
        assert msgs[2]["content"] == sample_completion

    def test_dict_input_auto_serialized(self):
        """Dict input should be auto-serialized to JSON string."""
        card_dict = {"dataset": "test", "node_id": 99, "dummy": True}
        msgs = format_evidence_card(card_dict)
        parsed = json.loads(msgs[1]["content"])
        assert parsed["node_id"] == 99

    def test_system_message_contains_fraud(self, sample_card):
        """System prompt should mention fraud."""
        msgs = format_evidence_card(sample_card)
        assert "fraud" in msgs[0]["content"].lower()

    def test_user_content_is_card_json(self, sample_card):
        """User content should be the card JSON string."""
        msgs = format_evidence_card(sample_card)
        parsed = json.loads(msgs[1]["content"])
        assert parsed["dataset"] == "YelpChi"


class TestApplyChatTemplate:
    """Tests for apply_chat_template returns correct keys."""

    def test_training_returns_input_ids_and_labels(self, mock_tokenizer, sample_card, sample_completion):
        """Training mode (with assistant) must return both input_ids and labels."""
        msgs = format_evidence_card(sample_card, completion_json=sample_completion)
        result = apply_chat_template(mock_tokenizer, msgs)

        assert "input_ids" in result
        assert "labels" in result
        assert isinstance(result["input_ids"], list)
        assert isinstance(result["labels"], list)
        assert len(result["input_ids"]) == len(result["labels"])

    def test_inference_returns_only_input_ids(self, mock_tokenizer, sample_card):
        """Inference mode (no assistant) must return only input_ids."""
        msgs = format_evidence_card(sample_card)
        result = apply_chat_template(mock_tokenizer, msgs, add_generation_prompt=True)

        assert "input_ids" in result
        assert "labels" not in result

    def test_labels_have_masked_prompt_tokens(self, mock_tokenizer, sample_card, sample_completion):
        """Labels should have -100 for prompt tokens (first N tokens)."""
        msgs = format_evidence_card(sample_card, completion_json=sample_completion)
        result = apply_chat_template(mock_tokenizer, msgs)

        labels = result["labels"]
        # First few tokens should be masked (prompt)
        assert labels[0] == -100

        # Some tokens should be unmasked (completion)
        assert any(l != -100 for l in labels)

    def test_disable_thinking_prefix_injects_no_think(self, mock_tokenizer, sample_card, sample_completion):
        """With disable_thinking=True, assistant message gets /no_think prefix."""
        msgs = format_evidence_card(sample_card, completion_json=sample_completion)
        result = apply_chat_template(mock_tokenizer, msgs, disable_thinking=True)

        # The assistant message in msgs should have /no_think prefix
        assistant_msgs = [m for m in msgs if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0]["content"].startswith(QWEN3_THINKING_DISABLED)

    def test_max_length_truncates(self, mock_tokenizer, sample_card, sample_completion):
        """Results should be truncated to max_length."""
        msgs = format_evidence_card(sample_card, completion_json=sample_completion)
        result = apply_chat_template(mock_tokenizer, msgs, max_length=50)

        assert len(result["input_ids"]) <= 50
        assert len(result["labels"]) <= 50

    def test_training_mask_uses_prefix_alignment_when_template_mismatches(self):
        """Masking should use aligned prefix, not raw prompt token length."""
        tok = MagicMock()
        tok.model_max_length = 4096

        def apply_template(messages, tokenize=False, add_generation_prompt=False):
            has_assistant = any(m["role"] == "assistant" for m in messages)
            if has_assistant:
                return "FULL"
            if add_generation_prompt:
                return "PROMPT_WITH_GEN"
            return "PROMPT_NO_GEN"

        tok.apply_chat_template.side_effect = apply_template

        token_map = {
            "PROMPT_NO_GEN": [10],
            "PROMPT_WITH_GEN": [10, 11, 12],
            "FULL": [10, 20, 21],
        }
        tok.encode.side_effect = lambda text, add_special_tokens=False: token_map[text]

        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": '{"label":"fraud"}'},
        ]
        result = apply_chat_template(tok, messages, disable_thinking=False)

        assert result["input_ids"] == [10, 20, 21]
        assert result["labels"] == [-100, 20, 21]
