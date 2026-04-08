"""Stage 2 co-training: joint generation + classification + distillation loss.

Loads the Stage 1 LoRA adapter and fine-tunes with the tri-loss:

    L = L_gen + lambda1 * L_cls + lambda2 * L_distill

where:
    - L_gen:     Causal LM cross-entropy on completion tokens
    - L_cls:     Binary BCE from the classification head
    - L_distill: KL divergence between student and teacher probabilities
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from priorf_reasoner_slm.llm.cls_head import BinaryClsHead
from priorf_reasoner_slm.llm.collators import CoTrainDataCollator
from priorf_reasoner_slm.llm.losses import PriorFLoss
from priorf_reasoner_slm.llm.model_wrapper import (
    DEFAULT_LORA_CONFIG,
    get_hidden_states,
    get_last_token_hidden,
    load_student_model,
)
from priorf_reasoner_slm.llm.tokenizer_utils import load_tokenizer
from priorf_reasoner_slm.train.train_utils import format_log, get_device, load_adapter

logger = logging.getLogger(__name__)

# ---- Default co-training hyperparameters ----
COTRAIN_CONFIG: dict[str, Any] = {
    "output_dir": "outputs/cotrain",
    "learning_rate": 5e-5,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "num_train_epochs": 3,
    "warmup_ratio": 0.1,
    "logging_steps": 10,
    "save_steps": 500,
    "save_total_limit": 2,
    "fp16": False,
    "bf16": True,
    "lr_scheduler_type": "cosine",
    "report_to": ["tensorboard"],
    "remove_unused_columns": False,
    "max_seq_length": 2048,
}

# ---- Loss weights ----
LAMBDA_CLS: float = 0.1
LAMBDA_DISTILL: float = 0.1
DISTILL_TEMP: float = 2.0

# ---- Model config ----
DEFAULT_MODEL_NAME = "Qwen/Qwen3-4B"
HIDDEN_DIM: int = 2560  # Fallback hidden size for Qwen3-4B-Instruct-2507
LORA_CONFIG: dict[str, Any] = DEFAULT_LORA_CONFIG.copy()


def _save_cotrain_checkpoint(
    model: nn.Module,
    cls_head: nn.Module,
    checkpoint_dir: Path,
) -> Path:
    """Save adapter + cls head into a checkpoint directory."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(checkpoint_dir / "adapter"))
    torch.save(cls_head.state_dict(), str(checkpoint_dir / "cls_head.pt"))
    return checkpoint_dir


class CoTrainer(nn.Module):
    """Co-training wrapper that combines LLM + cls head with tri-loss.

    Args:
        model: The student causal LM (LoRA-wrapped or full).
        cls_head: Binary classification head.
        lambda_cls: Weight for classification loss.
        lambda_distill: Weight for distillation loss.
        distill_temp: Temperature for KL distillation.
    """

    def __init__(
        self,
        model: nn.Module,
        cls_head: nn.Module,
        lambda_cls: float = LAMBDA_CLS,
        lambda_distill: float = LAMBDA_DISTILL,
        distill_temp: float = DISTILL_TEMP,
    ) -> None:
        super().__init__()
        self.model = model
        self.cls_head = cls_head
        self.loss_fn = PriorFLoss(
            lambda_cls=lambda_cls,
            lambda_distill=lambda_distill,
            distill_temperature=distill_temp,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        teacher_probs: torch.Tensor,
        cls_targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute the tri-loss forward pass.

        Args:
            input_ids: Token IDs ``(batch, seq_len)``.
            attention_mask: Attention mask ``(batch, seq_len)``.
            labels: Token labels with -100 for prompt ``(batch, seq_len)``.
            teacher_probs: Teacher fraud probabilities ``(batch,)``.
            cls_targets: Ground-truth binary labels ``(batch,)``.

        Returns:
            Dict with ``loss``, ``gen_loss``, ``cls_loss``, ``distill_loss``.
        """
        # LM forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        lm_logits = outputs.logits

        # Classification head: extract last-token hidden state
        hidden = get_hidden_states(self.model, input_ids, attention_mask)
        last_hidden = get_last_token_hidden(hidden, attention_mask)
        cls_logits = self.cls_head(last_hidden)

        # Tri-loss
        losses = self.loss_fn(
            lm_logits=lm_logits,
            labels=labels,
            cls_logits=cls_logits,
            cls_targets=cls_targets,
            teacher_probs=teacher_probs,
        )

        return losses


def build_cotrain_dataset(
    df: pd.DataFrame,
    tokenizer: Any,
    max_seq_length: int = 2048,
) -> list[dict[str, Any]]:
    """Build co-training samples from a teacher export DataFrame.

    Each sample includes input_ids, labels, teacher_probs, and cls_targets.

    Args:
        df: Teacher-exported DataFrame.
        tokenizer: Tokenizer for encoding.
        max_seq_length: Maximum sequence length.

    Returns:
        List of sample dicts ready for CoTrainDataCollator.
    """
    from priorf_reasoner_slm.evidence.dataset_builder import (
        build_evidence_card,
    )
    from priorf_reasoner_slm.evidence.rationale_templates import generate_prediction

    samples: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        card = build_evidence_card(row)

        # Build prompt
        prompt_text = card.model_dump_json(indent=2)
        prediction = generate_prediction(card)
        completion_text = prediction.model_dump_json()

        # Full text = prompt + completion
        full_text = prompt_text + "\n" + completion_text

        # Tokenize with labels: prompt tokens = -100, completion tokens = actual ids
        encoded = tokenizer(
            full_text,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            return_tensors=None,
        )
        input_ids = encoded["input_ids"]
        labels = list(input_ids)  # Start with same ids

        # Find where the completion starts (after prompt + newline)
        prompt_encoded = tokenizer(
            prompt_text + "\n",
            truncation=True,
            max_length=max_seq_length,
            return_tensors=None,
        )
        prompt_len = len(prompt_encoded["input_ids"])

        # Mark prompt tokens as -100
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100

        samples.append({
            "input_ids": input_ids,
            "labels": labels,
            "teacher_probs": float(row.get("teacher_prob", 0.5)),
            "cls_targets": int(row.get("label", 0)),
        })

    logger.info("Built %d co-training samples", len(samples))
    return samples


def train_cotrain(
    teacher_df: pd.DataFrame,
    adapter_path: str | Path,
    output_dir: str | Path = "outputs/cotrain",
    model_name: str = DEFAULT_MODEL_NAME,
    lambda_cls: float = LAMBDA_CLS,
    lambda_distill: float = LAMBDA_DISTILL,
    distill_temp: float = DISTILL_TEMP,
    training_args: dict[str, Any] | None = None,
    max_seq_length: int = 2048,
    per_device_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    num_train_epochs: int = 3,
    learning_rate: float = 5e-5,
) -> Path:
    """Run Stage 2 co-training.

    Args:
        teacher_df: DataFrame of teacher evidence exports.
        adapter_path: Path to the Stage 1 LoRA adapter.
        output_dir: Directory to save checkpoints and logs.
        model_name: HuggingFace model identifier.
        lambda_cls: Weight for classification loss.
        lambda_distill: Weight for distillation loss.
        distill_temp: Temperature for KL distillation.
        training_args: TrainingArguments overrides.
        max_seq_length: Maximum sequence length.
        per_device_batch_size: Batch size per device.
        gradient_accumulation_steps: Gradient accumulation steps.
        num_train_epochs: Number of epochs.
        learning_rate: Learning rate.

    Returns:
        Path to the final model checkpoint.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()

    logger.info("Loading tokenizer and model on %s", device)
    tokenizer = load_tokenizer(model_name)

    # Load model with LoRA from Stage 1
    base_model = load_student_model(
        model_name=model_name,
        lora_config=LORA_CONFIG,
        device=device,
    )
    model = load_adapter(base_model, str(adapter_path), device=device)

    # Classification head
    model_dtype = next(model.parameters()).dtype
    hidden_dim = int(getattr(getattr(model, "config", None), "hidden_size", HIDDEN_DIM))
    cls_head = BinaryClsHead(hidden_dim=hidden_dim, dropout=0.1).to(
        device=device,
        dtype=model_dtype,
    )

    # Co-trainer
    cotrainer = CoTrainer(
        model=model,
        cls_head=cls_head,
        lambda_cls=lambda_cls,
        lambda_distill=lambda_distill,
        distill_temp=distill_temp,
    )

    # Build dataset
    logger.info("Building co-training dataset from %d samples", len(teacher_df))
    samples = build_cotrain_dataset(
        teacher_df, tokenizer, max_seq_length=max_seq_length
    )

    # DataLoader with CoTrainDataCollator
    collator = CoTrainDataCollator(
        pad_token_id=tokenizer.pad_token_id or 0,
        ignore_index=-100,
    )
    dataloader = DataLoader(
        samples,
        batch_size=per_device_batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(cls_head.parameters()),
        lr=learning_rate,
    )

    # Training loop
    logger.info(
        "Starting co-training: lr=%.0e, epochs=%d, batch_size=%d, "
        "lambda_cls=%.2f, lambda_distill=%.2f, distill_temp=%.1f",
        learning_rate, num_train_epochs, per_device_batch_size,
        lambda_cls, lambda_distill, distill_temp,
    )

    total_steps = len(dataloader) * num_train_epochs // gradient_accumulation_steps
    warmup_steps = int(total_steps * 0.1)
    save_steps = int(
        (training_args or {}).get("save_steps", COTRAIN_CONFIG["save_steps"])
    )

    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    model.train()
    global_step = 0

    for epoch in range(num_train_epochs):
        epoch_losses = []
        epoch_gen = []
        epoch_cls = []
        epoch_dist = []

        for step, batch in enumerate(dataloader):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass
            losses = cotrainer(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                teacher_probs=batch["teacher_probs"],
                cls_targets=batch["cls_targets"],
            )

            loss = losses["loss"] / gradient_accumulation_steps
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(cls_head.parameters()),
                    max_norm=1.0,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if save_steps > 0 and global_step % save_steps == 0:
                    ckpt_path = output_dir / f"checkpoint-{global_step}"
                    _save_cotrain_checkpoint(model, cls_head, ckpt_path)
                    logger.info("Saved intermediate checkpoint: %s", ckpt_path)

            epoch_losses.append(loss.item() * gradient_accumulation_steps)
            epoch_gen.append(losses["gen_loss"].item())
            epoch_cls.append(losses["cls_loss"].item())
            epoch_dist.append(losses["distill_loss"].item())

            if (global_step + 1) % 10 == 0:
                logger.info(
                    "  Step %d | loss=%.4f | gen=%.4f | cls=%.4f | dist=%.4f",
                    global_step + 1,
                    losses["loss"].item(),
                    losses["gen_loss"].item(),
                    losses["cls_loss"].item(),
                    losses["distill_loss"].item(),
                )

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_gen = sum(epoch_gen) / len(epoch_gen)
        avg_cls = sum(epoch_cls) / len(epoch_cls)
        avg_dist = sum(epoch_dist) / len(epoch_dist)
        logger.info(
            "Epoch %d/%d | loss=%.4f | gen=%.4f | cls=%.4f | dist=%.4f",
            epoch + 1, num_train_epochs, avg_loss, avg_gen, avg_cls, avg_dist,
        )

    # Save final checkpoint
    final_path = output_dir / "final_cotrain"
    _save_cotrain_checkpoint(model, cls_head, final_path)

    logger.info("Co-training complete. Checkpoint saved to: %s", final_path)
    return final_path


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Stage 2 co-training for PriorF-Reasoner student")
    parser.add_argument("--teacher_csv", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to Stage 1 LoRA adapter")
    parser.add_argument("--output_dir", type=str, default="outputs/cotrain")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--lambda_cls", type=float, default=LAMBDA_CLS)
    parser.add_argument("--lambda_distill", type=float, default=LAMBDA_DISTILL)
    parser.add_argument("--distill_temp", type=float, default=DISTILL_TEMP)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    args = parser.parse_args()

    teacher_path = Path(args.teacher_csv)
    if teacher_path.suffix == ".parquet":
        df = pd.read_parquet(teacher_path)
    else:
        df = pd.read_csv(teacher_path)
    train_cotrain(
        teacher_df=df,
        adapter_path=args.adapter_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        lambda_cls=args.lambda_cls,
        lambda_distill=args.lambda_distill,
        distill_temp=args.distill_temp,
        max_seq_length=args.max_seq_length,
    )
