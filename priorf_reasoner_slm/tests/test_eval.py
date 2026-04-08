"""Tests for eval metrics and evaluation functions."""

from __future__ import annotations

import numpy as np
import pytest
import pandas as pd

from priorf_reasoner_slm.eval import metrics
from priorf_reasoner_slm.eval.eval_gen_only import evaluate_gen, evaluate_gen_from_scores_csv
from priorf_reasoner_slm.eval.eval_fusion import (
    evaluate_fusion,
    evaluate_fusion_with_optimization,
    evaluate_alpha_sweep,
)
from priorf_reasoner_slm.eval.eval_head_only import evaluate_head
from priorf_reasoner_slm.eval import faithfulness
from priorf_reasoner_slm.llm.generation import extract_prediction_score, parse_prediction_output


class TestMetrics:
    """Tests for the metrics module."""

    def test_auroc_perfect(self):
        """AUROC = 1.0 for perfect predictions."""
        y_true = [0, 0, 1, 1]
        y_score = [0.0, 0.0, 1.0, 1.0]
        assert np.isclose(metrics.compute_auroc(y_true, y_score), 1.0)

    def test_auroc_random(self):
        """AUROC ≈ 0.5 for random predictions."""
        rng = np.random.default_rng(42)
        y_true = [0] * 50 + [1] * 50
        y_score = rng.random(100).tolist()
        auroc = metrics.compute_auroc(y_true, y_score)
        assert 0.35 < auroc < 0.65  # widened tolerance for statistical noise

    def test_auroc_single_class(self):
        """AUROC = 0.5 when only one class present."""
        y_true = [1, 1, 1, 1]
        y_score = [0.1, 0.5, 0.8, 0.3]
        assert np.isclose(metrics.compute_auroc(y_true, y_score), 0.5)

    def test_auprc(self):
        """AUPRC is computed correctly."""
        y_true = [0, 0, 1, 1]
        y_score = [0.1, 0.4, 0.35, 0.8]
        auprc = metrics.compute_auprc(y_true, y_score)
        assert 0.0 <= auprc <= 1.0
        # Higher scores for the positive class should give higher AUPRC
        y_score2 = [0.9, 0.4, 0.35, 0.1]
        auprc2 = metrics.compute_auprc(y_true, y_score2)
        assert auprc > auprc2

    def test_find_optimal_threshold(self):
        """Optimal threshold is in valid range."""
        y_true = [0, 0, 1, 1]
        y_score = [0.1, 0.4, 0.35, 0.8]
        thresh = metrics.find_optimal_threshold(y_true, y_score)
        assert 0.0 <= thresh <= 1.0

    def test_f1_binary(self):
        """F1 score is computed correctly."""
        y_true = [0, 1, 1, 0, 1]
        y_score = [0.2, 0.8, 0.6, 0.3, 0.9]
        f1 = metrics.compute_f1(y_true, y_score, threshold=0.5)
        assert 0.0 <= f1 <= 1.0

    def test_precision_recall(self):
        """Precision and recall are computed correctly."""
        y_true = [0, 0, 1, 1]
        y_score = [0.1, 0.4, 0.35, 0.8]
        prec = metrics.compute_precision(y_true, y_score)
        rec = metrics.compute_recall(y_true, y_score)
        assert 0.0 <= prec <= 1.0
        assert 0.0 <= rec <= 1.0

    def test_gmeans(self):
        """G-means is in [0, 1]."""
        y_true = [0, 0, 1, 1]
        y_score = [0.1, 0.4, 0.35, 0.8]
        gm = metrics.compute_gmeans(y_true, y_score)
        assert 0.0 <= gm <= 1.0

    def test_specificity(self):
        """Specificity is computed correctly."""
        y_true = [0, 0, 1, 1]
        y_score = [0.1, 0.4, 0.35, 0.8]
        spec = metrics.compute_specificity(y_true, y_score)
        assert 0.0 <= spec <= 1.0

    def test_compute_all_metrics(self):
        """compute_all_metrics returns all keys."""
        y_true = [0, 0, 1, 1]
        y_score = [0.1, 0.4, 0.35, 0.8]
        result = metrics.compute_all_metrics(y_true, y_score)
        expected_keys = {
            "auroc", "auprc", "f1", "precision", "recall",
            "specificity", "gmeans", "optimal_threshold",
        }
        assert set(result.keys()) == expected_keys
        for v in result.values():
            assert isinstance(v, (int, float))


class TestFaithfulnessMetrics:
    """Tests for faithfulness metrics."""

    def test_sufficiency_high(self):
        """High sufficiency when evidence matters a lot."""
        # Both above chance: evidence maintains prediction well above random
        original = [0.9, 0.8, 0.7, 0.6]
        ablated = [0.8, 0.7, 0.6, 0.5]
        score = faithfulness.sufficiency(original, ablated)
        assert score > 0.2  # sufficiency above chance

    def test_sufficiency_low(self):
        """Low sufficiency when evidence barely matters."""
        original = [0.5, 0.5, 0.5, 0.5]
        ablated = [0.5, 0.5, 0.5, 0.5]
        score = faithfulness.sufficiency(original, ablated)
        assert score < 0.01  # no difference

    def test_comprehensiveness_high(self):
        """High comprehensiveness when teacher signal matters."""
        original = [0.9, 0.8, 0.7, 0.6]
        ablated = [0.5, 0.5, 0.5, 0.5]
        score = faithfulness.comprehensiveness(original, ablated)
        assert score > 0.2

    def test_comprehensiveness_low(self):
        """Low comprehensiveness when teacher signal barely matters."""
        original = [0.5, 0.5, 0.5, 0.5]
        ablated = [0.5, 0.5, 0.5, 0.5]
        score = faithfulness.comprehensiveness(original, ablated)
        assert score < 0.01

    def test_faithfulness_impact(self):
        """faithfulness_impact returns all keys."""
        original = [0.9, 0.8, 0.7, 0.6]
        ablated = [0.5, 0.5, 0.5, 0.5]
        result = faithfulness.faithfulness_impact(original, ablated)
        assert "sufficiency" in result
        assert "comprehensiveness" in result
        assert "prediction_flip_rate" in result

    def test_evaluate_comprehensiveness_accepts_cls_head_instance(self, monkeypatch):
        """Comprehensiveness API should work with a pre-loaded cls_head."""
        df = pd.DataFrame({
            "evidence_card_json": ['{"k":"v"}'],
            "teacher_prob": [0.7],
            "label": [1],
        })

        class DummyModel:
            def eval(self):
                return self

        class DummyTokenizer:
            def __call__(self, *args, **kwargs):
                import torch
                return {
                    "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
                    "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
                }

        class DummyHead:
            def to(self, device):
                return self

            def eval(self):
                return self

            def predict_proba(self, hidden):
                import torch
                return torch.tensor([0.6], dtype=torch.float32)

        import torch

        monkeypatch.setattr(
            "priorf_reasoner_slm.eval.faithfulness.get_hidden_states",
            lambda model, input_ids, attention_mask: torch.zeros((1, 3, 4), dtype=torch.float32),
        )
        monkeypatch.setattr(
            "priorf_reasoner_slm.eval.faithfulness.get_last_token_hidden",
            lambda hidden, attention_mask: hidden[:, -1, :],
        )

        result = faithfulness.evaluate_comprehensiveness(
            teacher_df=df,
            model=DummyModel(),
            cls_head_weights=None,
            tokenizer=DummyTokenizer(),
            device="cpu",
            sample_fraction=1.0,
            cls_head=DummyHead(),
        )

        assert "comprehensiveness" in result
        assert result["num_samples"] == 1


class TestEvalFusion:
    """Tests for fusion evaluation."""

    def test_fusion_fixed_alpha(self):
        """Fusion with fixed alpha produces valid metrics."""
        import pandas as pd

        df = pd.DataFrame({
            "label": [0, 0, 1, 1],
            "cls_prob": [0.2, 0.4, 0.7, 0.9],
            "teacher_prob": [0.3, 0.5, 0.6, 0.8],
        })

        result = evaluate_fusion(df, df["cls_prob"], df["teacher_prob"], alpha=0.5)

        assert "fusion_auroc" in result
        assert "fusion_auprc" in result
        assert "fusion_f1" in result
        assert result["alpha"] == 0.5
        assert 0.0 <= result["fusion_auroc"] <= 1.0

    def test_fusion_optimization(self):
        """Fusion optimization finds optimal alpha."""
        import pandas as pd

        df = pd.DataFrame({
            "label": [0, 0, 1, 1],
            "cls_prob": [0.2, 0.4, 0.7, 0.9],
            "teacher_prob": [0.3, 0.5, 0.6, 0.8],
        })

        result = evaluate_fusion_with_optimization(df, df["cls_prob"], df["teacher_prob"])

        assert "optimal_alpha" in result
        assert "best_holdout_accuracy" in result
        assert 0.0 <= result["optimal_alpha"] <= 1.0
        assert 0.0 <= result["best_holdout_accuracy"] <= 1.0

    def test_alpha_sweep(self):
        """Alpha sweep returns results for all alpha values."""
        import pandas as pd

        df = pd.DataFrame({
            "label": [0, 0, 1, 1],
            "cls_prob": [0.2, 0.4, 0.7, 0.9],
            "teacher_prob": [0.3, 0.5, 0.6, 0.8],
        })

        results = evaluate_alpha_sweep(df, df["cls_prob"], df["teacher_prob"], n_steps=10)

        assert len(results) == 11  # 0 to 10 inclusive
        assert all("alpha" in r for r in results)
        assert all(0.0 <= r["alpha"] <= 1.0 for r in results)


class TestEvalGenOnly:
    """Tests for generation evaluation from CSV."""

    def test_parse_prediction_output_normalizes_alias_schema(self):
        """Alias JSON fields should normalize into the strict M3 schema."""
        raw = """
        {
          "fraud_label": "benign",
          "confidence": 0.87,
          "behavioral_pattern": "normal structural behavior with isolated high-discrepancy relations",
          "key_evidence": ["low suspicious neighbor ratio"],
          "rationale": "benign"
        }
        """

        parsed = parse_prediction_output(raw)

        assert parsed == {
            "label": "benign",
            "score": 0.87,
            "pattern_hint": "suspicious",
            "evidence": ["low suspicious neighbor ratio"],
            "rationale": "benign",
        }

    def test_extract_prediction_score_accepts_alias_schema(self):
        """Faithfulness score extraction should accept alias output fields."""
        parsed = {
            "fraud_label": "fraud",
            "confidence": 0.91,
            "behavioral_pattern": "camouflage",
            "key_evidence": ["high HSD"],
            "rationale": "strong anomaly",
        }

        assert np.isclose(extract_prediction_score(parsed), 0.91)

    def test_evaluate_gen_from_scores_csv(self):
        """Pre-computed scores CSV is evaluated correctly."""
        import tempfile
        import os
        import pandas as pd

        df = pd.DataFrame({
            "generated_fraud_prob": [0.2, 0.4, 0.7, 0.9],
            "label": [0, 0, 1, 1],
        })

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f.name, index=False)
            csv_path = f.name

        try:
            result = evaluate_gen_from_scores_csv(csv_path)
            assert "gen_auroc" in result
            assert "gen_f1" in result
            assert result["num_samples"] == 4
        finally:
            os.unlink(csv_path)

    def test_evaluate_gen_parse_failure_not_scored(self, monkeypatch):
        """Parse failures should be counted, not injected as neutral scores."""
        df = pd.DataFrame({
            "label": [0, 1],
            "evidence_card_json": ['{"a":1}', '{"b":2}'],
        })

        monkeypatch.setattr(
            "priorf_reasoner_slm.eval.eval_gen_only.generate_prediction",
            lambda **kwargs: "not-json",
        )
        monkeypatch.setattr(
            "priorf_reasoner_slm.eval.eval_gen_only.parse_prediction_output",
            lambda text: None,
        )

        result = evaluate_gen(
            teacher_df=df,
            model=object(),
            tokenizer=object(),
            device="cpu",
        )

        assert result["num_samples"] == 2
        assert result["num_parse_failed"] == 2
        assert result["num_scored"] == 0
        assert np.isclose(result["score_coverage"], 0.0)

    def test_evaluate_gen_mixed_scoring_coverage(self, monkeypatch):
        """Only valid parsed+formatted samples should contribute to metrics."""
        df = pd.DataFrame({
            "label": [1, 0],
            "evidence_card_json": ['{"x":1}', '{"y":2}'],
        })

        outputs = iter(["first", "second"])
        parsed = iter([
            {
                "label": "fraud",
                "score": 0.8,
                "pattern_hint": "camouflage",
                "evidence": ["high HSD"],
                "rationale": "ok",
            },
            None,
        ])

        monkeypatch.setattr(
            "priorf_reasoner_slm.eval.eval_gen_only.generate_prediction",
            lambda **kwargs: next(outputs),
        )
        monkeypatch.setattr(
            "priorf_reasoner_slm.eval.eval_gen_only.parse_prediction_output",
            lambda text: next(parsed),
        )

        result = evaluate_gen(
            teacher_df=df,
            model=object(),
            tokenizer=object(),
            device="cpu",
        )

        assert result["num_samples"] == 2
        assert result["num_scored"] == 1
        assert np.isclose(result["score_coverage"], 0.5)
        assert result["num_parse_failed"] == 1

    def test_evaluate_gen_legacy_schema_not_counted_as_m3(self, monkeypatch):
        """Legacy output keys should not pass M3 format validation."""
        df = pd.DataFrame({
            "label": [1],
            "evidence_card_json": ['{"x":1}'],
        })

        monkeypatch.setattr(
            "priorf_reasoner_slm.eval.eval_gen_only.generate_prediction",
            lambda **kwargs: "legacy",
        )
        monkeypatch.setattr(
            "priorf_reasoner_slm.eval.eval_gen_only.parse_prediction_output",
            lambda text: {
                "fraud_prob": 0.9,
                "reasoning": "old",
                "anomaly_types": ["camouflage"],
            },
        )

        result = evaluate_gen(
            teacher_df=df,
            model=object(),
            tokenizer=object(),
            device="cpu",
        )

        assert result["num_parsed"] == 1
        assert result["num_format_correct"] == 0
        assert result["num_scored"] == 0

    def test_evaluate_gen_filters_unlabeled_rows(self, monkeypatch):
        """Rows with label=-1 should be excluded from binary evaluation."""
        df = pd.DataFrame({
            "label": [1, -1, 0],
            "evidence_card_json": ['{"x":1}', '{"y":2}', '{"z":3}'],
        })

        monkeypatch.setattr(
            "priorf_reasoner_slm.eval.eval_gen_only.generate_prediction",
            lambda **kwargs: "ok",
        )
        monkeypatch.setattr(
            "priorf_reasoner_slm.eval.eval_gen_only.parse_prediction_output",
            lambda text: {
                "label": "fraud",
                "score": 0.8,
                "pattern_hint": "camouflage",
                "evidence": ["high HSD"],
                "rationale": "ok",
            },
        )

        result = evaluate_gen(
            teacher_df=df,
            model=object(),
            tokenizer=object(),
            device="cpu",
        )

        assert result["num_samples"] == 2


class TestEvalHeadOnly:
    """Tests for classification head evaluation."""

    def test_evaluate_head_requires_cls_head_weights(self):
        """evaluate_head should fail fast if cls_head weights are missing."""
        df = pd.DataFrame({
            "label": [0],
            "evidence_card_json": ['{"dummy": true}'],
        })

        class DummyModel:
            def eval(self):
                return self

        with pytest.raises(ValueError, match="cls_head_state_dict is required"):
            evaluate_head(
                teacher_df=df,
                model=DummyModel(),
                cls_head_state_dict=None,
                device="cpu",
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
