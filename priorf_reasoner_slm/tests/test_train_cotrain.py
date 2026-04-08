"""Tests for train_cotrain checkpointing behavior."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

from priorf_reasoner_slm.train import train_cotrain as tc


class _DummyLM(torch.nn.Module):
    """Minimal LM stub for fast co-train loop tests."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.1))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        batch, seq_len = input_ids.shape
        # Keep vocab size small but valid for our sample labels.
        logits = torch.zeros((batch, seq_len, 8), dtype=torch.float32, device=input_ids.device)
        logits = logits + self.weight
        return SimpleNamespace(logits=logits)

    def save_pretrained(self, path: str) -> None:
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        (out / "dummy.bin").write_text("ok", encoding="utf-8")


def test_train_cotrain_saves_intermediate_and_final_checkpoints(tmp_path, monkeypatch):
    """save_steps should trigger intermediate checkpointing in co-training."""
    monkeypatch.setattr(tc, "get_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(tc, "load_tokenizer", lambda model_name: SimpleNamespace(pad_token_id=0))
    monkeypatch.setattr(tc, "load_student_model", lambda **kwargs: _DummyLM())
    monkeypatch.setattr(tc, "load_adapter", lambda model, adapter_path, device: model)
    monkeypatch.setattr(
        tc,
        "build_cotrain_dataset",
        lambda df, tokenizer, max_seq_length=2048: [
            {
                "input_ids": [1, 2, 3],
                "labels": [-100, 2, 3],
                "teacher_probs": 0.8,
                "cls_targets": 1,
            },
            {
                "input_ids": [1, 2, 3],
                "labels": [-100, 2, 3],
                "teacher_probs": 0.2,
                "cls_targets": 0,
            },
        ],
    )
    monkeypatch.setattr(
        tc,
        "get_hidden_states",
        lambda model, input_ids, attention_mask: torch.zeros(
            (input_ids.size(0), input_ids.size(1), tc.HIDDEN_DIM),
            dtype=torch.float32,
            device=input_ids.device,
        ),
    )

    teacher_df = pd.DataFrame({"dummy": [1, 2]})
    out_dir = tmp_path / "outputs"

    final_path = tc.train_cotrain(
        teacher_df=teacher_df,
        adapter_path=str(tmp_path / "adapter"),
        output_dir=out_dir,
        num_train_epochs=1,
        per_device_batch_size=1,
        gradient_accumulation_steps=1,
        training_args={"save_steps": 1},
    )

    assert final_path == out_dir / "final_cotrain"
    assert (out_dir / "checkpoint-1" / "adapter" / "dummy.bin").exists()
    assert (out_dir / "checkpoint-1" / "cls_head.pt").exists()
    assert (out_dir / "final_cotrain" / "adapter" / "dummy.bin").exists()
    assert (out_dir / "final_cotrain" / "cls_head.pt").exists()
