"""Tests for priorf_teacher: model loading, hooks, and evidence export."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from priorf_reasoner_slm.priorf_teacher.load_teacher import load_teacher_model
from priorf_reasoner_slm.priorf_teacher.export_hooks import (
    register_export_hooks,
    remove_hooks,
    extract_branch_logits,
)
from priorf_reasoner_slm.priorf_teacher.export_scores import (
    save_evidence,
    compute_hsd_quantile,
    _precompute_quantile_thresholds,
    export_teacher_evidence,
)


# ---- Minimal mock teacher model for testing ----
class MockTeacher(nn.Module):
    """Minimal 2-branch model mimicking LGHGCLNetV2 interface."""

    def __init__(self, in_dim: int = 5, hidden: int = 8, use_asda: bool = True):
        super().__init__()
        self.use_mlp = True
        self.use_gnn = True
        self.use_asda = use_asda

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
        )
        if use_asda:
            self.asda = nn.Module()
            self.asda.node_switch = nn.Sequential(
                nn.Linear(1, 4), nn.ReLU(), nn.Linear(4, 1), nn.Sigmoid()
            )

        self.rgcn2 = nn.Linear(in_dim, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.out = nn.Linear(hidden * 2, 1)
        self.emb = None

    def forward(self, x, edge_index=None, edge_type=None, hsd=None):
        z_mlp = self.mlp(x)
        h = self.rgcn2(x)
        z_gnn = torch.relu(self.bn2(h))
        if hasattr(self, "asda") and hasattr(self.asda, "node_switch"):
            inp = hsd.unsqueeze(-1) if hsd is not None else x[:, :1]
            _ = self.asda.node_switch(inp)
        self.emb = (z_mlp.detach(), z_gnn.detach())
        z = torch.cat([z_mlp, z_gnn], dim=-1)
        return self.out(z).squeeze(-1)


class TestLoadTeacher:
    def test_loads_from_checkpoint(self, tmp_path):
        model = MockTeacher(in_dim=5, hidden=8)
        ckpt_path = tmp_path / "best_model.pt"
        torch.save({"model_state_dict": model.state_dict(), "epoch": 10}, ckpt_path)

        loaded = load_teacher_model(ckpt_path, MockTeacher, {"in_dim": 5, "hidden": 8})
        assert isinstance(loaded, MockTeacher)
        for p1, p2 in zip(model.parameters(), loaded.parameters()):
            assert torch.equal(p1, p2)

    def test_ignores_unsupported_kwargs(self, tmp_path):
        model = MockTeacher(in_dim=5, hidden=8)
        ckpt_path = tmp_path / "best_model.pt"
        torch.save({"model_state_dict": model.state_dict(), "epoch": 10}, ckpt_path)

        loaded = load_teacher_model(
            ckpt_path,
            MockTeacher,
            {"in_dim": 5, "hidden": 8, "focal_alpha": 0.25, "lambda_sdcl": 0.1},
        )
        assert isinstance(loaded, MockTeacher)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_teacher_model("/nonexistent/ckpt.pt", MockTeacher, {"in_dim": 5})


class TestExportHooks:
    def test_hooks_capture_embeddings(self):
        model = MockTeacher(in_dim=5, hidden=8)
        handles, storage = register_export_hooks(model)

        x = torch.randn(10, 5)
        ei = torch.randint(0, 10, (2, 20))
        et = torch.zeros(20, dtype=torch.long)
        hsd = torch.randn(10)

        with torch.no_grad():
            model(x, ei, et, hsd)

        assert "mlp_embedding" in storage
        assert storage["mlp_embedding"].shape == (10, 8)
        assert "asda_switch" in storage
        assert storage["asda_switch"].shape == (10, 1)

        remove_hooks(handles)

    def test_no_hooks_when_no_asda(self):
        model = MockTeacher(in_dim=5, hidden=8, use_asda=False)
        handles, storage = register_export_hooks(model)

        x = torch.randn(10, 5)
        ei = torch.randint(0, 10, (2, 20))
        et = torch.zeros(20, dtype=torch.long)

        with torch.no_grad():
            model(x, ei, et)

        assert "asda_switch" not in storage
        remove_hooks(handles)


class TestExtractBranchLogits:
    def test_extracts_both_branches(self):
        model = MockTeacher(in_dim=5, hidden=8)
        mlp_emb = torch.randn(10, 8)
        gnn_emb = torch.randn(10, 8)

        result = extract_branch_logits(model, mlp_emb, gnn_emb)
        assert result["mlp_logit"] is not None
        assert result["gnn_logit"] is not None
        assert result["mlp_logit"].shape == (10,)
        assert result["gnn_logit"].shape == (10,)

    def test_fusion_layout_assertion(self):
        model = MockTeacher(in_dim=5, hidden=8)  # out = Linear(16, 1)
        mlp_emb = torch.randn(10, 4)  # Wrong dim
        gnn_emb = torch.randn(10, 4)
        with pytest.raises(AssertionError, match="Fusion layout"):
            extract_branch_logits(model, mlp_emb, gnn_emb)


class TestSaveEvidence:
    def test_save_csv(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
        path = save_evidence(df, tmp_path / "test.csv")
        assert path.exists()
        loaded = pd.read_csv(path)
        assert len(loaded) == 2


class TestExportTeacherEvidence:
    def test_export_contains_summary_and_card_json(self):
        model = MockTeacher(in_dim=5, hidden=8)
        x = torch.randn(4, 5)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
        edge_type = torch.zeros(edge_index.size(1), dtype=torch.long)
        hsd = torch.tensor([0.1, 0.8, 0.4, 0.7], dtype=torch.float32)
        y = torch.tensor([0, 1, 0, 1], dtype=torch.long)
        train_mask = torch.tensor([True, True, False, False])
        val_mask = torch.tensor([False, False, True, False])
        test_mask = torch.tensor([False, False, False, True])
        relations = {
            "RUR": {"edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long)},
            "RTR": {"edge_index": torch.tensor([[2, 3], [3, 2]], dtype=torch.long)},
        }

        df = export_teacher_evidence(
            model=model,
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            hsd=hsd,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            relations=relations,
            dataset_name="YelpChi",
            device="cpu",
        )

        assert "top_relations" in df.columns
        assert "neighbor_summary" in df.columns
        assert "evidence_card_json" in df.columns
        assert "card_json" in df.columns

        top_relations = json.loads(df.loc[0, "top_relations"])
        neighbor_summary = json.loads(df.loc[0, "neighbor_summary"])
        card = json.loads(df.loc[0, "evidence_card_json"])

        assert isinstance(top_relations, list)
        assert "degrees_per_relation" in neighbor_summary
        assert "same_relation_discrepancy_rank" in neighbor_summary
        assert card["dataset"] == "YelpChi"
        assert "teacher_summary" in card
        assert "structure_evidence" in card


class TestHsdQuantile:
    def test_quantile_buckets(self):
        hsd = torch.linspace(0, 10, 100)
        thresholds = _precompute_quantile_thresholds(hsd)
        assert compute_hsd_quantile(hsd[99].item(), thresholds) == "top_1_percent"
        assert compute_hsd_quantile(hsd[50].item(), thresholds) == "normal"
