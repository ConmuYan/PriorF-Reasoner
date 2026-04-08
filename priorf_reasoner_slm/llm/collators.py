"""Data collators for SFT and co-training of the student model.

Provides two collators:

- ``SFTDataCollator``: Standard SFT collator that pads input_ids,
  labels, and builds attention masks. Prompt tokens (label == -100)
  are padded with -100.

- ``CoTrainDataCollator``: Extends SFTDataCollator to also pad
  ``teacher_probs`` for the combined generation + cls + distill loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class SFTDataCollator:
    """Data collator for supervised fine-tuning.

    Pads ``input_ids`` and ``labels`` to the maximum sequence length
    in the batch. Builds ``attention_mask`` from the padded input_ids.

    Expected input format (list of dicts):
        - ``input_ids``: list[int]
        - ``labels``: list[int]  (with -100 for prompt tokens)
    """

    pad_token_id: int = 0
    ignore_index: int = -100

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collate a batch of feature dicts into padded tensors.

        Args:
            features: List of dicts, each containing ``input_ids`` and
                ``labels`` as lists of ints.

        Returns:
            Dict with padded ``input_ids``, ``labels``, and
            ``attention_mask`` tensors.
        """
        max_len = max(len(f["input_ids"]) for f in features)
        batch_size = len(features)

        input_ids = torch.full(
            (batch_size, max_len), self.pad_token_id, dtype=torch.long
        )
        labels = torch.full(
            (batch_size, max_len), self.ignore_index, dtype=torch.long
        )
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)

        for i, f in enumerate(features):
            ids = f["input_ids"]
            seq_len = len(ids)
            input_ids[i, :seq_len] = torch.tensor(ids, dtype=torch.long)
            labels[i, :seq_len] = torch.tensor(f["labels"], dtype=torch.long)
            attention_mask[i, :seq_len] = 1

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


@dataclass
class CoTrainDataCollator:
    """Data collator for co-training with generation + cls + distill losses.

    Extends the SFT collator to also handle ``teacher_probs`` and
    ``cls_targets`` fields needed for the classification and
    distillation losses.

    Expected input format (list of dicts):
        - ``input_ids``: list[int]
        - ``labels``: list[int]
        - ``teacher_probs``: float (scalar per sample)
        - ``cls_targets``: int or float (binary label per sample)
    """

    pad_token_id: int = 0
    ignore_index: int = -100

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collate a batch of feature dicts into padded tensors.

        Args:
            features: List of dicts with input_ids, labels,
                teacher_probs, and cls_targets.

        Returns:
            Dict with padded text tensors plus teacher_probs and
            cls_targets tensors.
        """
        max_len = max(len(f["input_ids"]) for f in features)
        batch_size = len(features)

        input_ids = torch.full(
            (batch_size, max_len), self.pad_token_id, dtype=torch.long
        )
        labels = torch.full(
            (batch_size, max_len), self.ignore_index, dtype=torch.long
        )
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)

        teacher_probs = torch.zeros(batch_size, dtype=torch.float)
        cls_targets = torch.zeros(batch_size, dtype=torch.float)

        for i, f in enumerate(features):
            ids = f["input_ids"]
            seq_len = len(ids)
            input_ids[i, :seq_len] = torch.tensor(ids, dtype=torch.long)
            labels[i, :seq_len] = torch.tensor(f["labels"], dtype=torch.long)
            attention_mask[i, :seq_len] = 1
            teacher_probs[i] = float(f["teacher_probs"])
            cls_targets[i] = float(f["cls_targets"])

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "teacher_probs": teacher_probs,
            "cls_targets": cls_targets,
        }
