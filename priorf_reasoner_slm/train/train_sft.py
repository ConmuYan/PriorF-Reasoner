"""Stage 1 SFT training using TRL's SFTTrainer.

Trains a Qwen3-4B student model with LoRA adapters on teacher-generated
evidence cards using supervised fine-tuning.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from priorf_reasoner_slm.evidence.dataset_builder import (
    build_conversational_dataset,
    build_prompt_completion_dataset,
    to_hf_dataset_conversational,
    to_hf_dataset_prompt_completion,
)
from priorf_reasoner_slm.llm.model_wrapper import DEFAULT_LORA_CONFIG, load_student_model
from priorf_reasoner_slm.llm.tokenizer_utils import load_tokenizer
from priorf_reasoner_slm.train.train_utils import format_log, get_device

logger = logging.getLogger(__name__)

# ---- Default SFT hyperparameters ----
SFT_CONFIG: dict[str, Any] = {
    "output_dir": "outputs/sft",
    "learning_rate": 1e-4,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "num_train_epochs": 3,
    "warmup_ratio": 0.1,
    "logging_steps": 10,
    "save_steps": 500,
    "save_total_limit": 2,
    "fp16": False,  # Set True for GPU with ampere+ hardware
    "bf16": True,   # Use bfloat16 on modern hardware
    "lr_scheduler_type": "cosine",
    "report_to": ["tensorboard"],
    "remove_unused_columns": False,
    "dataset_text_field": None,  # Set per dataset format
    "max_seq_length": 2048,
}

# LoRA config for Stage 1
LORA_CONFIG: dict[str, Any] = {
    "r": 16,
    "lora_alpha": 32,
    # Qwen3-4B-Instruct-2507 exposes standard attention projection modules.
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

# Model name or path
DEFAULT_MODEL_NAME = "Qwen/Qwen3-4B"


def build_sft_dataset_from_dataframe(
    df: pd.DataFrame,
    format: str = "prompt_completion",
    dataset_name: str | None = None,
) -> Any:
    """Build a HuggingFace Dataset for SFT from a teacher export DataFrame.

    Args:
        df: Teacher-exported DataFrame with evidence cards.
        format: Either "prompt_completion" or "conversational".
        dataset_name: Optional name for the dataset.

    Returns:
        A HuggingFace Dataset ready for SFTTrainer.
    """
    if format == "prompt_completion":
        samples = build_prompt_completion_dataset(df, dataset_name=dataset_name)
        return to_hf_dataset_prompt_completion(samples)
    elif format == "conversational":
        samples = build_conversational_dataset(df, dataset_name=dataset_name)
        return to_hf_dataset_conversational(samples)
    else:
        raise ValueError(f"Unknown format: {format!r}. Use 'prompt_completion' or 'conversational'.")


def train_sft(
    teacher_df: pd.DataFrame,
    output_dir: str | Path = "outputs/sft",
    model_name: str = DEFAULT_MODEL_NAME,
    dataset_format: str = "prompt_completion",
    lora_config: dict[str, Any] | None = None,
    training_args: dict[str, Any] | None = None,
    max_seq_length: int = 2048,
    formatting_prompt: str | None = None,
) -> Path:
    """Run Stage 1 SFT training.

    Args:
        teacher_df: DataFrame of teacher evidence exports.
        output_dir: Directory to save checkpoints and logs.
        model_name: HuggingFace model identifier.
        dataset_format: "prompt_completion" or "conversational".
        lora_config: LoRA configuration dict. None uses defaults.
        training_args: TrainingArguments overrides. None uses defaults.
        max_seq_length: Maximum sequence length for tokenization.
        formatting_prompt: Optional prompt template for formatting.

    Returns:
        Path to the final LoRA adapter checkpoint.
    """
    from trl import SFTConfig, SFTTrainer
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build dataset
    logger.info("Building SFT dataset from %d teacher samples", len(teacher_df))
    dataset = build_sft_dataset_from_dataframe(
        teacher_df, format=dataset_format
    )
    logger.info("Dataset size: %d", len(dataset))

    # Load tokenizer and model
    device = get_device()
    logger.info("Loading tokenizer and model on %s", device)
    tokenizer = load_tokenizer(model_name)
    model = load_student_model(
        model_name=model_name,
        lora_config=lora_config if lora_config is not None else LORA_CONFIG,
        device=device,
    )

    # Merge effective config
    effective_lora = DEFAULT_LORA_CONFIG.copy()
    if lora_config:
        effective_lora.update(lora_config)

    # Training arguments
    args = SFT_CONFIG.copy()
    args["output_dir"] = str(output_dir)
    if training_args:
        args.update(training_args)

    # TRL 1.0 uses SFTConfig for dataset/text-specific options such as
    # `dataset_text_field` and `max_length`.
    args["max_length"] = max_seq_length
    args.pop("max_seq_length", None)
    training_args_obj = SFTConfig(**args)

    # TRL's collator understands completion_mask and keeps loss on the
    # completion tokens only for prompt-completion datasets.
    collator = DataCollatorForLanguageModeling(
        pad_token_id=tokenizer.pad_token_id or 0,
        max_length=max_seq_length,
        completion_only_loss=True,
    )

    # SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args_obj,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=tokenizer,
        formatting_func=formatting_prompt,
    )

    logger.info(
        "Starting SFT training: lr=%.0e, epochs=%d, batch_size=%d (grad_accum=%d)",
        args["learning_rate"],
        args["num_train_epochs"],
        args["per_device_train_batch_size"],
        args["gradient_accumulation_steps"],
    )

    train_result = trainer.train()

    # Save adapter
    final_path = output_dir / "final_adapter"
    trainer.save_model(str(final_path))
    logger.info("SFT training complete. Final adapter saved to: %s", final_path)

    # Log metrics
    for key, value in train_result.metrics.items():
        logger.info("  %s: %s", key, format_log(value))

    return final_path


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Stage 1 SFT training for PriorF-Reasoner student")
    parser.add_argument("--teacher_csv", type=str, required=True, help="Path to teacher export CSV")
    parser.add_argument("--output_dir", type=str, default="outputs/sft", help="Output directory")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--dataset_format", type=str, default="prompt_completion", choices=["prompt_completion", "conversational"])
    parser.add_argument("--max_seq_length", type=int, default=2048)
    args = parser.parse_args()

    teacher_path = Path(args.teacher_csv)
    if teacher_path.suffix == ".parquet":
        df = pd.read_parquet(teacher_path)
    else:
        df = pd.read_csv(teacher_path)
    train_sft(
        teacher_df=df,
        output_dir=args.output_dir,
        model_name=args.model_name,
        dataset_format=args.dataset_format,
        max_seq_length=args.max_seq_length,
    )
